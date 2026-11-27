// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Memory helpers for C FFI boundaries.

use core::alloc::{AllocError, Allocator, Layout};
use core::mem::MaybeUninit;
use core::ptr::NonNull;

/// Overwrites `*dst` with `src` without reading or dropping the previous value.
///
/// Use this at C FFI `init` entry points that receive a partially-initialized `#[repr(C)]` struct
/// whose owned field (such as `Option<CBox<State>>` or `Option<Box<State>>`) contains uninitialized
/// or garbage bytes. A plain assignment (`*dst = src`) would run `Drop` on the uninitialized field,
/// and `core::mem::forget(core::mem::replace(dst, src))` would read it by value; both are undefined
/// behavior.
///
/// If `*dst` already holds an initialized value, that value is leaked without running its
/// destructor (just like [`core::mem::forget`]).
///
/// Note that forming a `&mut T` pointing to uninitialized memory relies on shallow reference
/// validity ([UCG#414](https://github.com/rust-lang/unsafe-code-guidelines/issues/414)).
///
/// # Example
///
/// ```
/// use core::mem::MaybeUninit;
/// use safer_cffi::overwrite_without_drop;
///
/// #[repr(C)]
/// struct Stream {
///     flags: u32,
///     state: Option<Box<u64>>,
/// }
///
/// // Simulate C passing a partially-initialized struct (`flags` set, `state` uninitialized).
/// let mut raw = MaybeUninit::<Stream>::uninit();
/// let ptr = raw.as_mut_ptr();
/// // SAFETY: `ptr` points to writable storage for `Stream`; writing `flags` touches no other field.
/// unsafe { (&raw mut (*ptr).flags).write(7) };
/// // SAFETY: `ptr` is non-null, aligned, dereferenceable for `Stream`, and uniquely accessed.
/// // Under UCG#414, `&mut Stream` is valid while `state` is uninitialized provided `state` is
/// // never read or dropped before initialization.
/// let stream: &mut Stream = unsafe { &mut *ptr };
///
/// overwrite_without_drop(&mut stream.state, Some(Box::new(42)));
/// assert_eq!(stream.flags, 7);
/// assert_eq!(stream.state.as_deref(), Some(&42));
/// // SAFETY: Both fields of `raw` are now initialized, and `stream` is no longer used.
/// unsafe { raw.assume_init_drop() };
/// ```
#[inline]
pub const fn overwrite_without_drop<T>(dst: &mut T, src: T) {
    // SAFETY: `core::ptr::write` requires `dst` to be valid for writes and properly aligned for
    // `T` (even when `T` is a ZST). Both hold because `dst` is a `&mut T` reference (Rust
    // Reference validity rules). Per `core::ptr::write`'s docs, it overwrites `*dst` with `src`
    // without reading or dropping the previous contents; skipping `Drop` on an unpinned `&mut T`
    // is sound in safe Rust (equivalent to `core::mem::forget(core::mem::replace(dst, src))` on
    // initialized memory), and on uninitialized memory (UCG#414) `*dst` is never read.
    unsafe { core::ptr::write(dst, src) };
}

/// Fallibly allocates memory in `alloc` and places `value` into it.
///
/// This is a stable-API equivalent of [`Box::try_new_in`] for constructing a [`Box`] whose
/// allocation failure must be reported as [`Err(AllocError)`] rather than aborting or panicking.
///
/// This does not allocate if `T` is zero-sized.
///
/// # Errors
///
/// Returns `Err(AllocError)` if `alloc.allocate` fails.
///
/// # Example
///
/// ```
/// use safer_cffi::{box_try_new_in, LibcAlloc};
///
/// let boxed = box_try_new_in(42u32, LibcAlloc)?;
/// assert_eq!(*boxed, 42);
/// # Ok::<(), safer_cffi::AllocError>(())
/// ```
#[inline]
pub fn box_try_new_in<T, A: Allocator>(value: T, alloc: A) -> Result<Box<T, A>, AllocError> {
    let ptr: NonNull<MaybeUninit<T>> = if size_of::<T>() == 0 {
        NonNull::dangling()
    } else {
        alloc.allocate(Layout::new::<MaybeUninit<T>>())?.cast()
    };
    // SAFETY:
    // Contract from `Box::from_raw_in` (see `alloc::boxed`'s "Memory layout" section):
    // - If `size_of_val` of the pointee (`size_of::<MaybeUninit<T>>() == size_of::<T>()`) is
    //   non-zero, `ptr` must point to a block of memory currently allocated by `alloc` such that
    //   `Layout::for_value` fits that block, and `ptr` must not already be owned by another `Box`.
    // - If `size_of_val` of the pointee is zero, `ptr` must be a non-null pointer sufficiently
    //   aligned for `MaybeUninit<T>`.
    // - The pointee must be a valid instance of `MaybeUninit<T>`.
    // Evidence:
    // - If `size_of::<T>() == 0`, `NonNull::<MaybeUninit<T>>::dangling()` is non-null and aligned
    //   to `align_of::<MaybeUninit<T>>()`.
    // - If `size_of::<T>() != 0`, `alloc.allocate(Layout::new::<MaybeUninit<T>>())` succeeded and
    //   returned an unaliased block currently allocated by `alloc`; by the `Allocator` trait safety
    //   invariant, `.cast()` is properly aligned and `Layout::for_value` (which equals
    //   `Layout::new::<MaybeUninit<T>>()` for `Sized` `T`) fits the allocated block. Moving `alloc`
    //   into `Box::from_raw_in` pairs the block with the same allocator that allocated it.
    // - Any byte pattern (including uninitialized bytes) is a valid `MaybeUninit<T>`.
    let uninit = unsafe { Box::from_raw_in(ptr.as_ptr(), alloc) };
    Ok(Box::write(uninit, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{CBox, LibcAlloc};
    use crate::testing::TrackingAlloc;
    use core::cell::Cell;
    use core::mem::MaybeUninit;
    use core::sync::atomic::Ordering;
    use googletest::prelude::*;

    #[repr(C)]
    struct PartialStream {
        flags: u32,
        state: Option<Box<u64>>,
    }

    struct DropCounter<'a>(&'a Cell<usize>);

    impl Drop for DropCounter<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[gtest]
    fn initializes_uninit_drop_field() {
        let mut raw = MaybeUninit::<PartialStream>::uninit();
        let ptr = raw.as_mut_ptr();
        // SAFETY: `ptr` points to writable `PartialStream` storage; writing `flags` does not read
        // other fields.
        unsafe { (&raw mut (*ptr).flags).write(123) };
        // SAFETY: `ptr` is non-null, aligned, dereferenceable for `PartialStream`, and uniquely
        // accessed; `state` is overwritten via `overwrite_without_drop` before any read or drop
        // (UCG#414).
        let stream = unsafe { &mut *ptr };

        overwrite_without_drop(&mut stream.state, None);
        expect_that!(stream.flags, eq(123));
        expect_that!(stream.state, none());

        stream.state = Some(Box::new(42));
        expect_that!(stream.state.as_deref(), some(eq(&42)));

        // SAFETY: All fields of `raw` are initialized and `stream` is no longer used.
        unsafe { raw.assume_init_drop() };
    }

    #[gtest]
    fn overwrites_garbage_pointer_bytes_without_dropping() {
        let mut raw = MaybeUninit::<PartialStream>::uninit();
        let ptr = raw.as_mut_ptr();
        // Fill the entire struct with non-zero misaligned garbage bytes (`0xAA`), simulating an
        // uninitialized C stack allocation where dropping `state` would dereference/free garbage.
        // SAFETY: `ptr` points to writable storage of `size_of::<PartialStream>()` bytes.
        unsafe { ptr.cast::<u8>().write_bytes(0xAA, core::mem::size_of::<PartialStream>()) };
        // SAFETY: `ptr` points to writable `PartialStream` storage.
        unsafe { (&raw mut (*ptr).flags).write(9) };
        // SAFETY: `ptr` is non-null, aligned, dereferenceable for `PartialStream`, and uniquely
        // accessed; `state` is overwritten via `overwrite_without_drop` before any read or drop
        // (UCG#414).
        let stream = unsafe { &mut *ptr };

        overwrite_without_drop(&mut stream.state, Some(Box::new(77)));
        expect_that!(stream.flags, eq(9));
        expect_that!(stream.state.as_deref(), some(eq(&77)));

        // SAFETY: All fields of `raw` are initialized and `stream` is no longer used.
        unsafe { raw.assume_init_drop() };
    }

    #[gtest]
    fn skips_drop_on_initialized_value() {
        let drops = Cell::new(0);
        {
            let mut slot = DropCounter(&drops);
            overwrite_without_drop(&mut slot, DropCounter(&drops));
            expect_that!(drops.get(), eq(0));
        }
        // Only the replacement value drops when `slot` leaves scope.
        expect_that!(drops.get(), eq(1));
    }

    #[gtest]
    fn works_with_zst() {
        static DROPS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        struct ZstDrop;
        impl Drop for ZstDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
        }

        let before = DROPS.load(core::sync::atomic::Ordering::Relaxed);
        {
            let mut slot = ZstDrop;
            overwrite_without_drop(&mut slot, ZstDrop);
            expect_that!(DROPS.load(core::sync::atomic::Ordering::Relaxed), eq(before));
        }
        expect_that!(DROPS.load(core::sync::atomic::Ordering::Relaxed), eq(before + 1));
    }

    #[gtest]
    fn works_in_const() {
        const VAL: u32 = {
            let mut raw = MaybeUninit::<u32>::uninit();
            // SAFETY: `raw` is non-null, aligned, dereferenceable for `u32`, and uniquely accessed;
            // `overwrite_without_drop` writes without reading (UCG#414).
            let slot = unsafe { &mut *raw.as_mut_ptr() };
            overwrite_without_drop(slot, 99);
            *slot
        };
        expect_that!(VAL, eq(99));
    }

    struct FailingAlloc;

    // SAFETY: `allocate` always fails without returning a block, so `deallocate` is unreachable for
    // valid blocks (`Allocator::deallocate` preconditions cannot be met).
    unsafe impl Allocator for FailingAlloc {
        fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            Err(AllocError)
        }

        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
    }

    #[gtest]
    fn box_try_new_in_allocates_and_drops_value() {
        let tracker = TrackingAlloc::default();
        let drops = Cell::new(0);
        {
            let boxed: CBox<(u32, DropCounter<'_>), &TrackingAlloc> =
                box_try_new_in((42, DropCounter(&drops)), &tracker).expect("allocation succeeds");
            expect_that!(boxed.0, eq(42));
            expect_that!(tracker.alloc_count.load(Ordering::SeqCst), eq(1));
            expect_that!(tracker.dealloc_count.load(Ordering::SeqCst), eq(0));
            expect_that!(drops.get(), eq(0));
        }
        expect_that!(tracker.dealloc_count.load(Ordering::SeqCst), eq(1));
        expect_that!(drops.get(), eq(1));
    }

    #[gtest]
    fn box_try_new_in_libc_alloc_succeeds() {
        let boxed = box_try_new_in(123u64, LibcAlloc).expect("LibcAlloc succeeds");
        expect_that!(*boxed, eq(123));
    }

    #[gtest]
    fn box_try_new_in_propagates_alloc_error_and_drops_value() {
        struct NonZstDrop<'a>(#[allow(dead_code)] u64, &'a Cell<usize>);
        impl Drop for NonZstDrop<'_> {
            fn drop(&mut self) {
                self.1.set(self.1.get() + 1);
            }
        }
        assert!(size_of::<NonZstDrop<'_>>() > 0);

        let drops = Cell::new(0);
        let res = box_try_new_in(NonZstDrop(7, &drops), FailingAlloc);
        expect_true!(res.is_err());
        expect_that!(drops.get(), eq(1));
    }

    #[gtest]
    fn box_try_new_in_handles_zst_without_allocating() {
        #[repr(align(64))]
        struct AlignedZst;
        assert_eq!(size_of::<AlignedZst>(), 0);

        let tracker = TrackingAlloc::default();
        let aligned =
            box_try_new_in(AlignedZst, &tracker).expect("aligned ZST should not allocate");
        expect_that!(core::ptr::from_ref(&*aligned) as usize % 64, eq(0));
        drop(aligned);
        expect_that!(tracker.alloc_count.load(Ordering::SeqCst), eq(0));
        expect_that!(tracker.dealloc_count.load(Ordering::SeqCst), eq(0));

        // Ensure ZST path bypasses `FailingAlloc` and still drops `value`.
        static ZST_DROPS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        struct DroppingZst;
        impl Drop for DroppingZst {
            fn drop(&mut self) {
                ZST_DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        assert_eq!(size_of::<DroppingZst>(), 0);

        let before = ZST_DROPS.load(Ordering::SeqCst);
        let zst_boxed =
            box_try_new_in(DroppingZst, FailingAlloc).expect("Drop ZST should not allocate");
        expect_that!(ZST_DROPS.load(Ordering::SeqCst), eq(before));
        drop(zst_boxed);
        expect_that!(ZST_DROPS.load(Ordering::SeqCst), eq(before + 1));
    }
}

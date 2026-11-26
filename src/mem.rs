// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Memory helpers for C FFI boundaries.

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

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use core::mem::MaybeUninit;
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
}

// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Memory allocator implementations for C FFI interoperability.

use allocator_api2::alloc::{AllocError, Allocator, Layout};
use allocator_api2::boxed::Box;
use allocator_api2::vec::Vec;
use core::ptr::NonNull;

/// A `Box` that uses the C allocator (`LibcAlloc`).
pub type CBox<T> = Box<T, LibcAlloc>;

/// A `Vec` that uses the C allocator (`LibcAlloc`).
pub type CVec<T> = Vec<T, LibcAlloc>;

/// The maximum alignment guaranteed by standard `malloc`: that of C's `max_align_t`, taken from
/// `libc` on the targets where it defines it.
#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    all(target_os = "android", not(target_arch = "arm")),
    all(target_os = "windows", target_env = "gnu"),
))]
const MALLOC_ALIGN: usize = core::mem::align_of::<libc::max_align_t>();

/// The maximum alignment guaranteed by standard `malloc`.
///
/// `libc` does not define `max_align_t` for this target, so approximate it with a `repr(C)`
/// union of the fundamental types. Erring high would be unsound: a layout `malloc` cannot
/// satisfy would get through. Erring low only refuses layouts it could have served. The
/// `libc_alloc_results_are_aligned` test checks the result at runtime.
#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    all(target_os = "android", not(target_arch = "arm")),
    all(target_os = "windows", target_env = "gnu"),
)))]
const MALLOC_ALIGN: usize = {
    #[repr(C)]
    union MaxAlign {
        _f64: f64,
        // `u128` pulls this up to 16 bytes where the C ABI gives `long double` or `__int128`
        // that alignment. Rust aligns it to 16 on 32-bit x86 too, where `malloc` guarantees
        // 8, and on aarch64 Apple targets `long double` is `double`, so leave it out on both.
        #[cfg(all(
            target_pointer_width = "64",
            not(all(target_vendor = "apple", target_arch = "aarch64"))
        ))]
        _u128: u128,
        _u64: u64,
        _ptr: *const (),
    }
    core::mem::align_of::<MaxAlign>()
};

/// Returns a non-null, dangling slice pointer aligned to `layout.align()` with length 0.
///
/// This is used to satisfy the [`Allocator`] contract for zero-sized allocations
/// (`layout.size() == 0`) without calling the system allocator (`malloc(0)`).
#[inline]
fn dangling_slice(layout: Layout) -> NonNull<[u8]> {
    NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0)
}

/// Number of bytes to request from `malloc` for `layout`, or `None` if `malloc` cannot satisfy
/// its alignment.
///
/// C23 (7.24.3, via N2293) only promises that `malloc(n)` is aligned for types of size at most
/// `n`, so `malloc(1)` may return a 1-aligned pointer. WG14 treats this as a clarification of
/// earlier wording, and Miri's `malloc` implements it. Requesting at least `layout.align()`
/// bytes brings the guarantee up to the alignment asked for. `std::alloc::System` avoids the
/// same hole by using plain `malloc` only when `align <= size`.
///
/// # Safety-usable invariant
///
/// If this returns `Some(n)`, then `n >= layout.size()` and a non-null result of `malloc(n)`,
/// `calloc(n, 1)`, or `realloc(_, n)` is aligned to `layout.align()`. The second part follows
/// from C23 7.24.3 given that `layout.align() <= MALLOC_ALIGN` and that on every supported
/// target each power of two up to `alignof(max_align_t)` is the alignment of some
/// fundamental type no larger than itself (`char`, `short`, `int`, `long long` or a pointer,
/// `long double` or `__int128`). `libc_alloc_results_are_aligned` checks this at runtime.
#[inline]
fn malloc_size(layout: Layout) -> Option<usize> {
    (layout.align() <= MALLOC_ALIGN).then(|| layout.size().max(layout.align()))
}

/// Zero-sized allocator backed by standard C library `malloc`, `calloc`, `realloc`, and `free`,
/// for memory whose ownership crosses the FFI boundary.
///
/// Beyond the [`Allocator`] contract, unsafe code may rely on the following:
///
/// - Every non-zero-sized block comes straight from `malloc`, `calloc`, or `realloc`, with no
///   header or offset. C code may `free` or `realloc` it. Conversely, `deallocate`, `grow`, and
///   `shrink` use only `ptr` and the layout sizes, never `old_layout.align()`, so a block from C's
///   `malloc` may be passed to them with any layout whose size is within the block.
/// - A block is aligned to its layout even when the layout is smaller than its alignment, which
///   plain `malloc(size)` does not guarantee (see `malloc_size`).
/// - A layout with alignment above `MALLOC_ALIGN` is refused with `AllocError`, never served
///   misaligned.
/// - Zero-sized layouts never reach `malloc`; the dangling pointer they get must not be passed to
///   `free`. `deallocate` handles this itself; code that frees through C must check the size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibcAlloc;

// SAFETY: `LibcAlloc` delegates to standard C allocation functions whose memory is safe
// to free with `libc::free`. Copying, cloning, or moving this allocator is not invalidating memory
// blocks returned from it, since the state lives in the allocator. Returned blocks meet the
// requested layout: `malloc_size` (see its safety-usable invariant) picks a request size that is
// at least `layout.size()` and makes the C result aligned to `layout.align()`.
unsafe impl Allocator for LibcAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }
        let size = malloc_size(layout).ok_or(AllocError)?;

        // SAFETY: `malloc` has no preconditions. If the result is non-null it is a fresh block,
        // valid for reads and writes of `size` bytes, and by `malloc_size`'s safety-usable
        // invariant `size >= layout.size()` and the block is aligned to `layout.align()`.
        let ptr = unsafe { libc::malloc(size) as *mut u8 };
        NonNull::new(ptr).map(|p| NonNull::slice_from_raw_parts(p, layout.size())).ok_or(AllocError)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }
        let size = malloc_size(layout).ok_or(AllocError)?;

        // SAFETY: `calloc` has no preconditions. If the result is non-null it is a fresh block of
        // `size` zeroed bytes, and by `malloc_size`'s safety-usable invariant `size >=
        // layout.size()` and the block is aligned to `layout.align()`.
        let ptr = unsafe { libc::calloc(size, 1) as *mut u8 };

        let non_null = NonNull::new(ptr).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(non_null, layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: The caller guarantees `ptr` was allocated by this allocator with `layout`.
            unsafe { libc::free(ptr.as_ptr().cast::<core::ffi::c_void>()) };
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if new_layout.size() == 0 {
            // SAFETY: The caller ensures `ptr` was allocated with `old_layout`.
            unsafe { self.deallocate(ptr, old_layout) };
            return Ok(dangling_slice(new_layout));
        }
        if old_layout.size() == 0 {
            return self.allocate(new_layout);
        }
        let size = malloc_size(new_layout).ok_or(AllocError)?;

        // SAFETY: The caller ensures that `ptr` was allocated by this allocator and has not been
        // deallocated yet. If the result is non-null it is valid for reads and writes of `size`
        // bytes with the first `min(old_layout.size(), size)` bytes preserved, and by
        // `malloc_size`'s safety-usable invariant `size >= new_layout.size()` and the block is
        // aligned to `new_layout.align()`. If it is null, `ptr` is still allocated and unchanged.
        let new_ptr =
            unsafe { libc::realloc(ptr.cast::<core::ffi::c_void>().as_ptr(), size) as *mut u8 };
        NonNull::new(new_ptr)
            .map(|p| NonNull::slice_from_raw_parts(p, new_layout.size()))
            .ok_or(AllocError)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: The caller must ensure that `ptr` was allocated by this allocator
        // and has not been deallocated yet.
        unsafe { self.grow(ptr, old_layout, new_layout) }
    }
}

/// An allocator that supports freeing memory using only the pointer.
///
/// This trait is implemented by allocators (such as [`LibcAlloc`]) that track allocation
/// metadata internally and can deallocate non-zero-sized allocations without requiring
/// the original [`Layout`].
///
/// Note that [`deallocate_by_ptr`](Self::deallocate_by_ptr) does not take a `&self` receiver
/// because pointer wrappers like [`OwnedCBufPtr`](crate::OwnedCBufPtr) are `#[repr(transparent)]`
/// around a single raw pointer and cannot store an allocator instance, so they must invoke
/// deallocation statically during `Drop`.
pub trait DropByPtrAllocator: Allocator {
    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// - `ptr` denotes a block of memory currently allocated via this allocator with a non-zero
    ///   size (`layout.size() > 0`).
    /// - No active references (`&T` or `&mut T`) or other aliases to the memory block exist, and
    ///   the memory referenced by `ptr` is not accessed after this call.
    unsafe fn deallocate_by_ptr(ptr: NonNull<u8>);
}

// The libc allocator supports freeing with only the pointer as an argument (`libc::free`).
impl DropByPtrAllocator for LibcAlloc {
    #[inline]
    unsafe fn deallocate_by_ptr(ptr: NonNull<u8>) {
        // SAFETY: The caller guarantees `ptr` was allocated by this allocator with a non-zero
        // layout and is exclusively owned.
        unsafe { libc::free(ptr.as_ptr().cast::<core::ffi::c_void>()) };
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]
    use super::*;
    use googletest::prelude::*;

    /// Checks that `allocate`, `allocate_zeroed`, `grow` and `shrink` serve every alignment
    /// `malloc` guarantees, return pointers aligned to the layout, and refuse anything stricter.
    /// Layouts smaller than their alignment are included: plain `malloc(size)` would not
    /// guarantee those.
    ///
    /// Under Miri this is exact, since its `malloc` aligns only as much as C requires. On a real
    /// libc a pass is evidence, not proof: allocators tend to over-align small blocks.
    #[gtest]
    fn libc_alloc_results_are_aligned() {
        let alloc = LibcAlloc;
        let is_aligned = |slice: NonNull<[u8]>, align: usize| {
            (slice.as_ptr() as *mut u8).addr().is_multiple_of(align)
        };

        let mut align = 1;
        while align <= MALLOC_ALIGN {
            for size in 1..=4 * MALLOC_ALIGN {
                let layout = Layout::from_size_align(size, align).unwrap();
                let larger = Layout::from_size_align(size + 1, align).unwrap();

                let slice = alloc.allocate(layout).unwrap();
                assert_that!(is_aligned(slice, align), eq(true), "allocate({size}, {align})");
                let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
                // SAFETY: `ptr` was allocated with `layout`, and `larger` is not smaller.
                let slice = unsafe { alloc.grow(ptr, layout, larger) }.unwrap();
                assert_that!(is_aligned(slice, align), eq(true), "grow({size}, {align})");
                let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
                // SAFETY: `ptr` was returned by `grow` with `larger`.
                unsafe { alloc.deallocate(ptr, larger) };

                let slice = alloc.allocate_zeroed(layout).unwrap();
                assert_that!(
                    is_aligned(slice, align),
                    eq(true),
                    "allocate_zeroed({size}, {align})"
                );
                let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
                // SAFETY: `ptr` was allocated with `layout`.
                unsafe { alloc.deallocate(ptr, layout) };
            }

            // Shrinking to below the alignment is the `realloc` form of the same case.
            let big = Layout::from_size_align(4 * MALLOC_ALIGN, align).unwrap();
            let one = Layout::from_size_align(1, align).unwrap();
            let ptr = NonNull::new(alloc.allocate(big).unwrap().as_ptr() as *mut u8).unwrap();
            // SAFETY: `ptr` was allocated with `big`, and `one` is not larger.
            let slice = unsafe { alloc.shrink(ptr, big, one) }.unwrap();
            assert_that!(is_aligned(slice, align), eq(true), "shrink(1, {align})");
            let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
            // SAFETY: `ptr` was returned by `shrink` with `one`.
            unsafe { alloc.deallocate(ptr, one) };

            align *= 2;
        }

        let too_strict = Layout::from_size_align(1, 2 * MALLOC_ALIGN).unwrap();
        assert_that!(alloc.allocate(too_strict), err(anything()));
        assert_that!(alloc.allocate_zeroed(too_strict), err(anything()));

        let small = Layout::from_size_align(1, 1).unwrap();
        let ptr = NonNull::new(alloc.allocate(small).unwrap().as_ptr() as *mut u8).unwrap();
        // SAFETY: `ptr` was allocated with `small`, and `too_strict` is not smaller.
        assert_that!(unsafe { alloc.grow(ptr, small, too_strict) }, err(anything()));
        // SAFETY: a failed `grow` leaves `ptr` allocated with `small`.
        unsafe { alloc.deallocate(ptr, small) };
    }

    #[gtest]
    fn libc_alloc_zero_size() {
        let alloc = LibcAlloc;
        let layout0 = Layout::from_size_align(0, 8).unwrap();
        let layout16 = Layout::from_size_align(16, 8).unwrap();

        let res = alloc.allocate(layout0);
        assert_that!(res, ok(anything()));
        let slice = res.unwrap();
        assert_that!(slice.len(), eq(0));
        assert_that!((slice.as_ptr() as *const u8 as usize) % layout0.align(), eq(0));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();

        // Grow 0 -> 0
        let grown0 = unsafe { alloc.grow(non_null, layout0, layout0) }.unwrap();
        assert_that!(grown0.len(), eq(0));
        assert_that!((grown0.as_ptr() as *const u8 as usize) % layout0.align(), eq(0));
        let non_null0 = NonNull::new(grown0.as_ptr() as *mut u8).unwrap();

        // Grow 0 -> 16
        let grown16 = unsafe { alloc.grow(non_null0, layout0, layout16) }.unwrap();
        assert_that!(grown16.len(), eq(16));
        let non_null16 = NonNull::new(grown16.as_ptr() as *mut u8).unwrap();

        // Shrink 16 -> 0
        let shrunk0 = unsafe { alloc.shrink(non_null16, layout16, layout0) }.unwrap();
        assert_that!(shrunk0.len(), eq(0));
        assert_that!((shrunk0.as_ptr() as *const u8 as usize) % layout0.align(), eq(0));
        let non_null_final = NonNull::new(shrunk0.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null_final` was returned by `shrink` with `layout0`.
        unsafe { alloc.deallocate(non_null_final, layout0) };
    }

    #[gtest]
    fn libc_alloc_allocate_and_deallocate() {
        let alloc = LibcAlloc;
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(16));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` was allocated with `layout`.
        unsafe { alloc.deallocate(non_null, layout) };
    }

    #[gtest]
    fn libc_alloc_deallocate_by_ptr() {
        let alloc = LibcAlloc;
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(16));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` was allocated by `LibcAlloc` with a non-zero layout,
        // no references or aliases exist, and ownership is transferred to `deallocate_by_ptr`.
        unsafe { LibcAlloc::deallocate_by_ptr(non_null) };
    }

    #[gtest]
    fn libc_alloc_allocate_zeroed() {
        let alloc = LibcAlloc;
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate_zeroed(layout).expect("allocate_zeroed succeeds");
        assert_that!(slice.len(), eq(16));
        // SAFETY: `slice` has 16 bytes initialized to zero.
        let bytes = unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u8, 16) };
        assert!(bytes.iter().all(|&b| b == 0));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` was allocated with `layout`.
        unsafe { alloc.deallocate(non_null, layout) };
    }

    #[gtest]
    fn libc_alloc_grow_and_shrink() {
        let alloc = LibcAlloc;
        let l1 = Layout::array::<u32>(2).unwrap();
        let l2 = Layout::array::<u32>(8).unwrap();
        let slice = alloc.allocate(l1).expect("allocation succeeds");
        let non_null1 = NonNull::new(slice.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null1` was allocated with `l1`, and `l2.size() >= l1.size()`.
        let grown = unsafe { alloc.grow(non_null1, l1, l2) }.expect("grow succeeds");
        assert_that!(grown.len(), eq(32));
        let non_null2 = NonNull::new(grown.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null2` was resized to `l2`, and `l1.size() <= l2.size()`.
        let shrunk = unsafe { alloc.shrink(non_null2, l2, l1) }.expect("shrink succeeds");
        assert_that!(shrunk.len(), eq(8));
        let non_null3 = NonNull::new(shrunk.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null3` has layout `l1`.
        unsafe { alloc.deallocate(non_null3, l1) };
    }

    #[gtest]
    fn cbox_usage() {
        let b = CBox::new_in(12345i32, LibcAlloc);
        assert_that!(*b, eq(12345));
    }
}

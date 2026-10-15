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

/// The maximum alignment guaranteed by standard `malloc`.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows", target_os = "ios"))]
const MALLOC_ALIGN: usize = core::mem::align_of::<libc::max_align_t>();

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "windows",
    target_os = "ios"
)))]
const MALLOC_ALIGN: usize = {
    // Fallback for weird platforms: Do an approximation at compile time,
    // but also check the alignment at runtime below to avoid UB.
    #[repr(C)]
    union MallocAlignProxy {
        _a: f64,
        _b: u64,
        _c: *const (),
    }
    core::mem::align_of::<MallocAlignProxy>()
};

/// Returns a non-null, dangling slice pointer aligned to `layout.align()` with length 0.
///
/// This is used to satisfy the [`Allocator`] contract for zero-sized allocations
/// (`layout.size() == 0`) without calling the system allocator (`malloc(0)`).
#[inline]
fn dangling_slice(layout: Layout) -> NonNull<[u8]> {
    NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0)
}

/// Zero-sized allocator backed by standard C library `malloc`, `calloc`, `realloc`, and `free`.
///
/// managed by C standard library functions across the FFI boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibcAlloc;

impl LibcAlloc {
    /// Reallocates `ptr` from `old_layout` to `new_layout` (supporting both growing and
    /// shrinking).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is currently allocated via `LibcAlloc` with layout `old_layout`.
    /// - `new_layout` is a valid [`Layout`].
    #[inline]
    unsafe fn realloc_impl(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if new_layout.size() == 0 {
            // SAFETY: By this function's `# Safety` contract, `ptr` is currently allocated by
            // `self` with `old_layout`, satisfying `Allocator::deallocate`.
            unsafe { self.deallocate(ptr, old_layout) };
            return Ok(dangling_slice(new_layout));
        }
        if old_layout.size() == 0 {
            return self.allocate(new_layout);
        }
        if new_layout.align() > MALLOC_ALIGN {
            return Err(AllocError);
        }

        // SAFETY: By this function's `# Safety` contract, `ptr` was allocated by `LibcAlloc` with a
        // non-zero `old_layout` (`old_layout.size() > 0` checked above) via `malloc`, `calloc`, or
        // `realloc` and has not been deallocated, and `new_layout.size() > 0` is checked above.
        // Standard C `realloc` guarantees alignment up to `MALLOC_ALIGN` (`>= new_layout.align()`).
        let new_ptr = unsafe {
            libc::realloc(ptr.cast::<core::ffi::c_void>().as_ptr(), new_layout.size()) as *mut u8
        };
        NonNull::new(new_ptr)
            .map(|p| NonNull::slice_from_raw_parts(p, new_layout.size()))
            .ok_or(AllocError)
    }
}

// SAFETY: `LibcAlloc` implements `Allocator` by delegating non-zero-sized allocations with
// `layout.align() <= MALLOC_ALIGN` to the C standard library's `malloc`, `calloc`, `realloc`, and
// `free` (returning `Err(AllocError)` when `layout.align() > MALLOC_ALIGN`), and returning a
// non-null aligned dangling pointer for zero-sized layouts without calling `malloc` or `free`.
// Because `LibcAlloc` is a zero-sized unit struct, cloning, copying, or moving it never invalidates
// any memory blocks allocated through it.
unsafe impl Allocator for LibcAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }

        if layout.align() > MALLOC_ALIGN {
            return Err(AllocError);
        }
        // SAFETY: `libc::malloc` has no memory-safety preconditions when called with a non-zero
        // byte size (`layout.size() > 0`), and returns either null or a pointer aligned to at least
        // `MALLOC_ALIGN` (`>= layout.align()`).
        let ptr = unsafe { libc::malloc(layout.size()) as *mut u8 };
        NonNull::new(ptr).map(|p| NonNull::slice_from_raw_parts(p, layout.size())).ok_or(AllocError)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }

        if layout.align() > MALLOC_ALIGN {
            return Err(AllocError);
        }

        // SAFETY: `libc::calloc` has no memory-safety preconditions when called with
        // `layout.size() > 0` and element size `1`, and returns either null or a zero-initialized
        // allocation aligned to at least `MALLOC_ALIGN` (`>= layout.align()`).
        let ptr = unsafe { libc::calloc(layout.size(), 1) as *mut u8 };

        let non_null = NonNull::new(ptr).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(non_null, layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: By `Allocator::deallocate`'s `# Safety` contract, `ptr` denotes a block of
            // memory currently allocated via `LibcAlloc` with `layout`. Because `layout.size() !=
            // 0`, `ptr` was obtained from `libc::malloc`, `libc::calloc`, or
            // `libc::realloc` and has not yet been freed, satisfying `libc::free`.
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
        // SAFETY: By `Allocator::grow`'s `# Safety` contract, `ptr` is currently allocated via
        // `LibcAlloc` with `old_layout`, satisfying `realloc_impl`'s preconditions.
        unsafe { self.realloc_impl(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: By `Allocator::shrink`'s `# Safety` contract, `ptr` is currently allocated via
        // `LibcAlloc` with `old_layout`, satisfying `realloc_impl`'s preconditions.
        unsafe { self.realloc_impl(ptr, old_layout, new_layout) }
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
    /// The caller must ensure that:
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
        // SAFETY: By `DropByPtrAllocator::deallocate_by_ptr`'s `# Safety` contract, `ptr` denotes a
        // currently allocated non-zero-sized block obtained from `LibcAlloc` (`malloc`, `calloc`,
        // or `realloc`) with no remaining active references or uses, satisfying
        // `libc::free`.
        unsafe { libc::free(ptr.as_ptr().cast::<core::ffi::c_void>()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

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
        // SAFETY: `non_null` was returned by `alloc.allocate(layout0)` and `layout0.size() >=
        // layout0.size()`.
        let grown0 = unsafe { alloc.grow(non_null, layout0, layout0) }.unwrap();
        assert_that!(grown0.len(), eq(0));
        assert_that!((grown0.as_ptr() as *const u8 as usize) % layout0.align(), eq(0));
        let non_null0 = NonNull::new(grown0.as_ptr() as *mut u8).unwrap();

        // Grow 0 -> 16
        // SAFETY: `non_null0` was returned by `alloc.grow` with `layout0`, and `layout16.size() >=
        // layout0.size()`.
        let grown16 = unsafe { alloc.grow(non_null0, layout0, layout16) }.unwrap();
        assert_that!(grown16.len(), eq(16));
        let non_null16 = NonNull::new(grown16.as_ptr() as *mut u8).unwrap();

        // Shrink 16 -> 0
        // SAFETY: `non_null16` was returned by `alloc.grow` with `layout16`, and `layout0.size() <=
        // layout16.size()`.
        let shrunk0 = unsafe { alloc.shrink(non_null16, layout16, layout0) }.unwrap();
        assert_that!(shrunk0.len(), eq(0));
        assert_that!((shrunk0.as_ptr() as *const u8 as usize) % layout0.align(), eq(0));
        let non_null_final = NonNull::new(shrunk0.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null_final` was returned by `alloc.shrink` with `layout0`.
        unsafe { alloc.deallocate(non_null_final, layout0) };
    }

    #[gtest]
    fn libc_alloc_allocate_and_deallocate() {
        let alloc = LibcAlloc;
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(16));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` is currently allocated via `alloc.allocate(layout)` with `layout`.
        unsafe { alloc.deallocate(non_null, layout) };
    }

    #[gtest]
    fn libc_alloc_deallocate_by_ptr() {
        let alloc = LibcAlloc;
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(16));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` was allocated by `LibcAlloc` with a non-zero layout (`16` bytes),
        // no references or aliases exist, and ownership is transferred to `deallocate_by_ptr`.
        unsafe { LibcAlloc::deallocate_by_ptr(non_null) };
    }

    #[gtest]
    fn libc_alloc_allocate_zeroed() {
        let alloc = LibcAlloc;
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate_zeroed(layout).expect("allocate_zeroed succeeds");
        assert_that!(slice.len(), eq(16));
        // SAFETY: `slice` is a non-null, 1-byte-aligned allocation of 16 zero-initialized `u8`
        // bytes within a single live allocation that is not mutated while `bytes` is read.
        let bytes = unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u8, 16) };
        assert!(bytes.iter().all(|&b| b == 0));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` is currently allocated via `alloc.allocate_zeroed(layout)` with
        // `layout`.
        unsafe { alloc.deallocate(non_null, layout) };
    }

    #[gtest]
    fn libc_alloc_grow_and_shrink() {
        let alloc = LibcAlloc;
        let l1 = Layout::array::<u32>(2).unwrap();
        let l2 = Layout::array::<u32>(8).unwrap();
        let slice = alloc.allocate(l1).expect("allocation succeeds");
        let non_null1 = NonNull::new(slice.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null1` is currently allocated via `alloc` with `l1`, and `l2.size() >=
        // l1.size()`.
        let grown = unsafe { alloc.grow(non_null1, l1, l2) }.expect("grow succeeds");
        assert_that!(grown.len(), eq(32));
        let non_null2 = NonNull::new(grown.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null2` is currently allocated via `alloc` with `l2`, and `l1.size() <=
        // l2.size()`.
        let shrunk = unsafe { alloc.shrink(non_null2, l2, l1) }.expect("shrink succeeds");
        assert_that!(shrunk.len(), eq(8));
        let non_null3 = NonNull::new(shrunk.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null3` is currently allocated via `alloc` with `l1`.
        unsafe { alloc.deallocate(non_null3, l1) };
    }

    #[gtest]
    fn cbox_usage() {
        let b = CBox::new_in(12345i32, LibcAlloc);
        assert_that!(*b, eq(12345));
    }
}

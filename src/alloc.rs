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
use core::ptr::NonNull;

/// A `Box` that uses the C allocator (`LibcAlloc`).
pub type CBox<T> = Box<T, LibcAlloc>;

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
    // How this works:
    // - `layout.align()` is guaranteed by `Layout` invariants to be a non-zero power of two (>= 1).
    // - Casting `layout.align() as *mut u8` yields an integer memory address equal to the alignment.
    //   Because this address is non-zero, `NonNull::new` is guaranteed to succeed and never panic.
    // - Because the address is numerically equal to `layout.align()`, it is naturally an integer
    //   multiple of `layout.align()`, ensuring the pointer is properly aligned.
    // - `NonNull::slice_from_raw_parts` attaches a slice length of 0 to form the `NonNull<[u8]>`.
    let ptr = NonNull::new(layout.align() as *mut u8).unwrap();
    NonNull::slice_from_raw_parts(ptr, 0)
}

/// Zero-sized allocator backed by standard C library `malloc`, `calloc`, `realloc`, and `free`.
///
/// managed by C standard library functions across the FFI boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibcAlloc;

// SAFETY: `LibcAlloc` delegates to standard C allocation functions whose memory is safe
// to free with `libc::free`. Copying, cloning, or moving this allocator is not invalidating memory
// blocks returned from it, since the state lives in the allocator.
unsafe impl Allocator for LibcAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }

        if layout.align() > MALLOC_ALIGN {
            return Err(AllocError);
        }
        // SAFETY: `malloc` has no preconditions.
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

        // SAFETY: `calloc` has no preconditions.
        let ptr = unsafe { libc::calloc(layout.size(), 1) as *mut u8 };

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
        _old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if new_layout.align() > MALLOC_ALIGN {
            return Err(AllocError);
        }

        // SAFETY: The caller ensures that `ptr` was allocated by this allocator
        // and has not been deallocated yet.
        let new_ptr = unsafe {
            libc::realloc(ptr.cast::<core::ffi::c_void>().as_ptr(), new_layout.size()) as *mut u8
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn libc_alloc_zero_size() {
        let alloc = LibcAlloc;
        let layout = Layout::from_size_align(0, 8).unwrap();
        let res = alloc.allocate(layout);
        assert_that!(res, ok(anything()));
        let slice = res.unwrap();
        assert_that!(slice.len(), eq(0));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` was returned by `allocate` with `layout`.
        unsafe { alloc.deallocate(non_null, layout) };
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

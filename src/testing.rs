// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Shared test utilities and mock allocators for tests.

#![allow(dead_code)]

use crate::alloc::LibcAlloc;
use crate::c_buf::{CBufLen, CBufPtr};
use allocator_api2::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::ffi::c_int;

/// Helper to create a malloc-backed buffer with the given values.
/// Returns `(ptr, len)` suitable for `CBufPtr`/`CVecRefMut`.
pub(crate) fn malloc_array<T, const N: usize>(values: [T; N]) -> (CBufPtr<T>, c_int) {
    malloc_array_typed::<T, c_int, N>(values).expect("malloc_array failed")
}

/// Helper to allocate an uninitialized non-null `CBufPtr<i32>` (16 bytes via `libc::malloc`)
/// for testing non-null zero-length buffer handling.
pub(crate) fn malloc_uninit_i32_buf() -> CBufPtr<i32> {
    // SAFETY: `libc::malloc` has no memory-safety preconditions when called with a non-zero
    // allocation size (`16 > 0`), and returns either null or a pointer aligned to at least
    // `align_of::<libc::max_align_t>()` (`>= align_of::<i32>()`).
    let raw = unsafe { libc::malloc(16) } as *mut i32;
    assert!(!raw.is_null(), "libc::malloc failed in test helper");
    // SAFETY: `raw` is non-null (verified by `assert!` above), properly aligned for `i32`
    // (guaranteed by `libc::malloc`), and points to a live 16-byte allocation from `LibcAlloc`
    // (`libc::malloc`) exclusively owned by the returned `CBufPtr<i32>`.
    unsafe { CBufPtr::from_raw(raw) }
}

/// Helper to free a malloc-backed buffer without running element destructors.
///
/// # Safety
///
/// The caller must ensure that:
/// - Either `ptr` is null, or `ptr` was allocated by `libc::malloc` / `LibcAlloc` and has not yet
///   been freed.
/// - No active references or aliases to the allocation exist, and `ptr` is not used after this
///   call.
pub(crate) unsafe fn free_array<T>(ptr: CBufPtr<T>) {
    if !ptr.is_null() {
        // SAFETY: By this function's `# Safety` contract, `ptr` is non-null, was allocated by
        // `libc::malloc`, and is exclusively owned with no remaining uses.
        unsafe { libc::free(ptr.as_ptr() as *mut libc::c_void) };
    }
}

/// Helper to create a malloc-backed buffer with a typed length.
pub(crate) fn malloc_array_typed<T, L: CBufLen, const N: usize>(
    values: [T; N],
) -> Result<(CBufPtr<T>, L), AllocError> {
    const {
        assert!(core::mem::size_of::<T>() > 0);
        assert!(core::mem::align_of::<T>() <= core::mem::align_of::<libc::max_align_t>());
    }
    let size = core::mem::size_of_val(&values);
    let len = L::try_from(N).map_err(|_| AllocError)?;
    if size == 0 {
        return Ok((CBufPtr::null(), len));
    }
    // SAFETY: `libc::malloc` has no memory-safety preconditions when called with `size > 0`.
    let p = unsafe { libc::malloc(size) } as *mut T;
    if p.is_null() {
        return Err(AllocError);
    }
    for (i, v) in values.into_iter().enumerate() {
        // SAFETY: `p` is non-null and points to a live allocation of `N` elements of `T` (whose
        // size comes from the existing stack array `values`, so `size <= isize::MAX`), and `i < N`.
        let dst = unsafe { p.add(i) };
        // SAFETY: `dst` is non-null, aligned for `T` (`align_of::<T>() <= max_align_t` asserted
        // above), and points to an uninitialized slot within the exclusively owned allocation.
        unsafe { core::ptr::write(dst, v) };
    }
    // SAFETY: `p` is non-null, aligned for `T`, and points to `N` initialized `T` elements
    // allocated by `LibcAlloc` (`libc::malloc`).
    Ok((unsafe { CBufPtr::from_raw(p) }, len))
}

/// Custom tracking allocator for testing generic Allocator support.
#[derive(Default, Debug)]
pub(crate) struct TrackingAlloc {
    pub(crate) alloc_count: AtomicUsize,
    pub(crate) dealloc_count: AtomicUsize,
    pub(crate) grow_count: AtomicUsize,
}

impl PartialEq for TrackingAlloc {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self, other)
    }
}
impl Eq for TrackingAlloc {}

// SAFETY: `&TrackingAlloc` delegates all allocation, deallocation, growth, and shrinking operations
// directly to `LibcAlloc` while incrementing atomic counters. Because `LibcAlloc` satisfies the
// `Allocator` safety contract and copying/moving `&TrackingAlloc` does not invalidate any allocated
// memory blocks, `&TrackingAlloc` also satisfies the `Allocator` contract.
unsafe impl Allocator for &TrackingAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_count.fetch_add(1, Ordering::SeqCst);
        LibcAlloc.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc_count.fetch_add(1, Ordering::SeqCst);
        // SAFETY: The caller of `Allocator::deallocate` guarantees that `ptr` denotes a block of
        // memory currently allocated via this allocator (which delegates to `LibcAlloc`) and that
        // `layout` fits that block of memory, satisfying `LibcAlloc::deallocate`.
        unsafe { LibcAlloc.deallocate(ptr, layout) };
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.grow_count.fetch_add(1, Ordering::SeqCst);
        // SAFETY: The caller of `Allocator::grow` guarantees that `ptr` is currently allocated via
        // this allocator (which delegates to `LibcAlloc`), `old_layout` fits that block, and
        // `new_layout.size() >= old_layout.size()`, satisfying `LibcAlloc::grow`.
        unsafe { LibcAlloc.grow(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: The caller of `Allocator::shrink` guarantees that `ptr` is currently allocated
        // via this allocator (which delegates to `LibcAlloc`), `old_layout` fits that block, and
        // `new_layout.size() <= old_layout.size()`, satisfying `LibcAlloc::shrink`.
        unsafe { LibcAlloc.shrink(ptr, old_layout, new_layout) }
    }
}

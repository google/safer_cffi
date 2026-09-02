// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Shared test utilities and mock allocators for tests.

#![allow(dead_code)]
#![allow(clippy::undocumented_unsafe_blocks)]

use crate::alloc::LibcAlloc;
use crate::c_slice::{CSliceLen, CSlicePtr};
use allocator_api2::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::ffi::c_int;

/// Helper to create a malloc-backed buffer with the given values.
/// Returns `(ptr, len)` suitable for `CSlicePtr`/`CVecRefMut`.
///
/// # Safety
///
/// Caller must ensure that values can be written to malloc-allocated memory.
pub(crate) unsafe fn malloc_array<T, const N: usize>(values: [T; N]) -> (CSlicePtr<T>, c_int) {
    let size = core::mem::size_of_val(&values);
    let p = unsafe { libc::malloc(size) } as *mut T;
    for (i, v) in values.into_iter().enumerate() {
        unsafe { core::ptr::write(p.add(i), v) };
    }
    (unsafe { CSlicePtr::from_raw(p) }, N as c_int)
}

/// Helper to free a malloc-backed buffer.
///
/// # Safety
///
/// `ptr` must have been allocated via `libc::malloc` or be null.
pub(crate) unsafe fn free_array<T>(ptr: CSlicePtr<T>) {
    if !ptr.is_null() {
        unsafe { libc::free(ptr.as_ptr() as *mut libc::c_void) };
    }
}

/// Helper to create a malloc-backed buffer with a typed length.
pub(crate) fn malloc_array_typed<T, L: CSliceLen, const N: usize>(
    values: [T; N],
) -> Result<(CSlicePtr<T>, L), AllocError> {
    let size = core::mem::size_of_val(&values);
    let p = unsafe { libc::malloc(size) } as *mut T;
    if p.is_null() {
        return Err(AllocError);
    }
    for (i, v) in values.into_iter().enumerate() {
        unsafe { core::ptr::write(p.add(i), v) };
    }
    let len = L::try_from(N).ok().expect("valid len");
    Ok((unsafe { CSlicePtr::from_raw(p) }, len))
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

// SAFETY: Tracking wrapper delegating directly to `LibcAlloc`.
unsafe impl Allocator for &TrackingAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_count.fetch_add(1, Ordering::SeqCst);
        LibcAlloc.allocate(layout)
    }

    // SAFETY: Delegates to `LibcAlloc::deallocate`.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc_count.fetch_add(1, Ordering::SeqCst);
        // SAFETY: Caller guarantees `ptr` was allocated by this allocator with `layout`.
        unsafe { LibcAlloc.deallocate(ptr, layout) };
    }

    // SAFETY: Delegates to `LibcAlloc::grow`.
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.grow_count.fetch_add(1, Ordering::SeqCst);
        // SAFETY: Caller guarantees `ptr` was allocated with `old_layout`.
        unsafe { LibcAlloc.grow(ptr, old_layout, new_layout) }
    }

    // SAFETY: Delegates to `LibcAlloc::shrink`.
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: Caller guarantees `ptr` was allocated with `old_layout`.
        unsafe { LibcAlloc.shrink(ptr, old_layout, new_layout) }
    }
}

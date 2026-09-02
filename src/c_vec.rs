// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Safe handles for growable raw-pointer-backed dynamic vectors in C structs.
//!
//! [`CVecRefMut`] represents a borrowed mutable view over a `(*mut T, L)` pair
//! (where `L` is an integer length type such as `c_int` or `usize`), providing
//! vector operations like [`push_back`](CVecRefMut::push_back), [`try_push_back`](CVecRefMut::try_push_back),
//! [`clear`](CVecRefMut::clear), [`replace`](CVecRefMut::replace), and [`swap`](CVecRefMut::swap).

use crate::alloc::LibcAlloc;
use crate::c_slice::{max_slice_len, CSliceLen, CSlicePtr};
use allocator_api2::alloc::{Allocator, Layout};
use core::ptr::{self, NonNull};

// ---------------------------------------------------------------------------
//  CVecRefMut — mutable vector handle
// ---------------------------------------------------------------------------

/// A borrowed mutable handle over a `(*mut T, L)` pair representing a dynamic C vector.
///
/// Created via [`CSlicePtr::with_len_vec_mut`] or [`CSlicePtr::with_len_vec_mut_in`].
/// Provides mutable slice access and vector mutation operations ([`push_back`](Self::push_back),
/// [`try_push_back`](Self::try_push_back), [`clear`](Self::clear), [`replace`](Self::replace), [`swap`](Self::swap)).
///
/// Slice access is provided through [`Deref`](core::ops::Deref) and
/// [`DerefMut`](core::ops::DerefMut), which correctly tie the returned
/// slice's lifetime to the borrow of this handle.
///
/// # Safety Invariant
///
/// If `len > 0`, it is the length of array `ptr`, and must be <= `isize::MAX`.
/// If `len <= 0`, the array is empty.
pub struct CVecRefMut<'a, T, L: CSliceLen, A: Allocator + PartialEq = LibcAlloc> {
    pub(crate) ptr: &'a mut CSlicePtr<T, A>,
    pub(crate) len: &'a mut L,
    pub(crate) alloc: A,
}

impl<'a, T, L: CSliceLen, A: Allocator + PartialEq> CVecRefMut<'a, T, L, A> {
    /// Return the slice view with the lifetime tied to the borrow.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let len = (*self.len).try_into().unwrap_or(0);
        if self.ptr.is_null() || len == 0 {
            return &[];
        }
        // SAFETY:
        // - Since `ptr` is not null, the invariants for `CSlicePtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CSlicePtr` owns the underlying array, so the pointer is valid for
        //   reads for the lifetime of this object (&self, created from a `CSlicePtr`).
        // - The invariant for `CSliceRefMut` guarantees that `len` is the valid length of the array
        //   (as per CSliceLen's safety contract) pointed to by `ptr`.
        // - `L: CSliceLen` guarantees that `try_into()` is deterministic and pure.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), len) }
    }

    /// Return the mutable slice view with the lifetime tied to the borrow.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let len = (*self.len).try_into().unwrap_or(0);
        if self.ptr.is_null() || len == 0 {
            return &mut [];
        }
        // SAFETY:
        // - Since `ptr` is not null, the invariants for `CSlicePtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CSlicePtr` owns the underlying array, so the pointer is valid for
        //   reads for the lifetime of this object (&self, created from a `CSlicePtr`).
        // - The invariant for `CVecRefMut` guarantees that `len` is the valid length of the array
        //   pointed to by `ptr`.
        // - `L: CSliceLen` guarantees that `try_into()` is deterministic and pure.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }

    /// Return a reference to the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Return the number of elements in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        (*self.len).try_into().unwrap_or(0)
    }

    /// Return `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append an element, reallocating via the configured [`Allocator`] to grow by one slot.
    ///
    /// Returns `Err(value)` if the array cannot grow (out of memory or `len`
    /// overflow), giving the caller the element back.
    pub fn try_push_back(&mut self, value: T) -> Result<(), T> {
        let old_len = (*self.len).try_into().unwrap_or(0);
        let Some(new_len) = old_len.checked_add(1).filter(|&n| n <= max_slice_len::<T>()) else {
            core::hint::cold_path();
            return Err(value);
        };
        let Ok(new_len_val) = L::try_from(new_len) else {
            core::hint::cold_path();
            return Err(value);
        };

        // Cannot overflow: `new_len <= max_slice_len::<T>()` guarantees
        // `new_len * size_of::<T>() <= isize::MAX`.
        let Ok(old_layout) = Layout::array::<T>(old_len.max(1)) else {
            core::hint::cold_path();
            return Err(value);
        };
        let Ok(new_layout) = Layout::array::<T>(new_len) else {
            core::hint::cold_path();
            return Err(value);
        };

        let result = match NonNull::new(self.ptr.as_ptr() as *mut u8) {
            // SAFETY: If `*self.ptr` is non-null, it was allocated by `self.alloc` with `old_layout`.
            Some(old_ptr) => unsafe { self.alloc.grow(old_ptr, old_layout, new_layout) },
            None => self.alloc.allocate(new_layout),
        };
        let Ok(new_slice) = result else {
            core::hint::cold_path();
            return Err(value);
        };
        let new_ptr = new_slice.as_ptr() as *mut T;

        // SAFETY: `new_ptr` has room for `new_len` elements; the first
        // `old_len` are already initialised. We write one more at the end.
        unsafe { ptr::write(new_ptr.add(old_len), value) };

        // SAFETY: `new_ptr` was allocated via the configured allocator, points to an array of
        // `T` and is aligned.
        let p = unsafe { CSlicePtr::from_raw(new_ptr) };
        *self.ptr = p;
        *self.len = new_len_val;
        Ok(())
    }

    /// Append an element, reallocating via the configured [`Allocator`] to grow by one slot.
    ///
    /// # Panics
    /// Panics if the array cannot grow (out of memory or `len` overflow).
    pub fn push_back(&mut self, value: T) {
        if self.try_push_back(value).is_err() {
            core::hint::cold_path();
            panic!("CVecRefMut: allocation failed");
        }
    }

    /// Swap the underlying pointer and len with another handle that uses the same allocator.
    ///
    /// # Panics
    /// Panics if `self` and `other` do not share the same allocator instance (as determined by [`PartialEq`]).
    pub fn swap(&mut self, other: &mut CVecRefMut<'_, T, L, A>) {
        assert!(self.alloc == other.alloc, "CVecRefMut::swap: handles must use the same allocator");
        core::mem::swap(self.ptr, other.ptr);
        core::mem::swap(self.len, other.len);
    }

    /// Drop all elements and deallocate the buffer using the configured [`Allocator`].
    pub fn clear(&mut self) {
        let len = (*self.len).try_into().unwrap_or(0);
        // We replace the pointer and length first to leave the handle in a valid, empty
        // state immediately. This is necessary for panic safety: if dropping elements
        // panics, the handle won't point to invalid memory.
        let old_ptr = core::mem::replace(self.ptr, CSlicePtr::null());
        *self.len = L::default();

        if !old_ptr.is_null()
            && let Some(non_null) = NonNull::new(old_ptr.as_ptr() as *mut u8)
        {
            // We use a local Drop guard to guarantee that deallocation is called
            // even if `ptr::drop_in_place` panics while dropping the elements.
            // This prevents leaking the underlying allocation.
            struct AllocDropGuard<'g, A: Allocator> {
                alloc: &'g A,
                ptr: NonNull<u8>,
                layout: Layout,
            }
            impl<'g, A: Allocator> Drop for AllocDropGuard<'g, A> {
                fn drop(&mut self) {
                    // SAFETY: `self.ptr` was allocated via `self.alloc` with `self.layout`.
                    unsafe { self.alloc.deallocate(self.ptr, self.layout) };
                }
            }
            let layout = Layout::array::<T>(len.max(1)).expect("CVecRefMut: valid layout");
            let _guard = AllocDropGuard { alloc: &self.alloc, ptr: non_null, layout };

            if len > 0 {
                let slice = ptr::slice_from_raw_parts_mut(old_ptr.as_ptr(), len);
                // SAFETY: `old_ptr` points to `n` valid elements.
                // Run `Drop` impls on all elements (see `cslice_ref_mut_clear_drops_elements`).
                unsafe { ptr::drop_in_place(slice) };
            } else {
                core::hint::cold_path();
            }
        }
    }
}

impl<T, L: CSliceLen, A: Allocator + PartialEq> core::ops::Deref for CVecRefMut<'_, T, L, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, L: CSliceLen, A: Allocator + PartialEq> core::ops::DerefMut for CVecRefMut<'_, T, L, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T: core::fmt::Debug, L: CSliceLen, A: Allocator + PartialEq> core::fmt::Debug
    for CVecRefMut<'_, T, L, A>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_slice(), f)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use super::*;
    use crate::testing::*;
    use core::sync::atomic::{AtomicU8, Ordering};
    use googletest::prelude::*;
    use std::ffi::c_int;

    #[gtest]
    fn c_vec_ref_mut_null_ptr() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = 0;
        let handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
    }

    #[gtest]
    fn c_vec_ref_mut_nonnull_ptr_zero_len() {
        // Simulate a C struct where a buffer was allocated but len is 0.
        // clear() must still free the buffer.
        let mut ptr = unsafe { CSlicePtr::<i32>::from_raw(libc::malloc(16) as *mut i32) };
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_nonnull_ptr_zero_len_push_back() {
        // Simulate a C struct where a buffer was allocated but len is 0.
        // push_back() must free the previous buffer and grow properly.
        let mut ptr = unsafe { CSlicePtr::<i32>::from_raw(libc::malloc(16) as *mut i32) };
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle.push_back(123);
        assert_that!(&*handle, container_eq([123]));
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_deref() {
        let (mut ptr, mut len) = unsafe { malloc_array([5, 6, 7]) };
        let handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        assert_that!(&*handle, container_eq([5, 6, 7]));
        assert_that!(handle.len(), eq(3));

        // Clean up.
        let mut handle = handle;
        handle.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_deref_mut() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle[0] = 99;
        assert_that!(&*handle, container_eq([99, 2, 3]));
        handle.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_push_back_to_empty() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle.push_back(42);
        assert_that!(&*handle, container_eq([42]));
        handle.push_back(43);
        assert_that!(&*handle, container_eq([42, 43]));
        handle.clear();
        // Verify len was updated correctly (check after dropping the borrow).
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_push_back_to_existing() {
        let (mut ptr, mut len) = unsafe { malloc_array([10]) };
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle.push_back(20);
        handle.push_back(30);
        assert_that!(&*handle, container_eq([10, 20, 30]));
        handle.clear();
        // Verify len was updated correctly (check after dropping the borrow).
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_custom_allocator_push_back_and_clear() {
        let alloc = TrackingAlloc::default();
        let mut ptr = CSlicePtr::null();
        let mut count: c_int = 0;
        {
            // SAFETY: Null pointer with length 0 is safe.
            let mut handle = unsafe { ptr.with_len_vec_mut_in(&mut count, &alloc) };
            assert_that!(handle.allocator().alloc_count.load(Ordering::SeqCst), eq(0));
            handle.push_back(100);
            assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));
            handle.push_back(200);
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(1));
            handle.push_back(300);
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(2));
            assert_that!(&*handle, container_eq([100, 200, 300]));
            handle.clear();
            assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(1));
        }
        assert!(ptr.is_null());
        assert_that!(count, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_custom_allocator_push_back_to_nonnull_zero_len_grows() {
        let alloc = TrackingAlloc::default();
        // Allocate a dummy buffer first.
        let slice = (&alloc).allocate(Layout::new::<i32>()).unwrap();
        // SAFETY: ptr is valid and was just allocated, we don't modify it.
        let mut ptr = unsafe { CSlicePtr::from_raw(slice.as_ptr() as *mut i32) };
        let mut count: c_int = 0;
        {
            // SAFETY: ptr is non-null, count is 0, allocated via alloc.
            let mut handle = unsafe { ptr.with_len_vec_mut_in(&mut count, &alloc) };
            assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(0));
            assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(0));
            handle.push_back(42);
            // Must have grown the existing buffer via grow().
            assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(1));
            assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(0));
            assert_that!(&*handle, container_eq([42]));
            handle.clear();
            assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(1));
        }
        assert!(ptr.is_null());
        assert_that!(count, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_try_push_back_overflow() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = c_int::MAX;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        let result = handle.try_push_back(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }

    #[gtest]
    fn c_vec_ref_mut_clear_nonempty() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle.clear();
        assert_that!(handle.len(), eq(0));
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_swap() {
        let (mut ptr1, mut len1) = unsafe { malloc_array([1, 2, 3]) };
        let (mut ptr2, mut len2) = unsafe { malloc_array([4, 5]) };

        {
            let mut handle1 = unsafe { ptr1.with_len_vec_mut(&mut len1) };
            let mut handle2 = unsafe { ptr2.with_len_vec_mut(&mut len2) };

            handle1.swap(&mut handle2);

            assert_that!(&*handle1, container_eq([4, 5]));
            assert_that!(&*handle2, container_eq([1, 2, 3]));
        }

        // Verify that the underlying values were swapped.
        assert_that!(len1, eq(2));
        assert_that!(len2, eq(3));
        // SAFETY: `ptr1` has `len1` (2) elements after swap.
        assert_that!(unsafe { ptr1.with_len(len1) }, container_eq([4, 5]));
        // SAFETY: `ptr2` has `len2` (3) elements after swap.
        assert_that!(unsafe { ptr2.with_len(len2) }, container_eq([1, 2, 3]));

        // Clean up both handles.
        let mut handle1 = unsafe { ptr1.with_len_vec_mut(&mut len1) };
        handle1.clear();
        let mut handle2 = unsafe { ptr2.with_len_vec_mut(&mut len2) };
        handle2.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_swap_different_allocators_panics() {
        let alloc1 = TrackingAlloc::default();
        let alloc2 = TrackingAlloc::default();

        let mut ptr1 = CSlicePtr::<i32, _>::null();
        let mut len1: c_int = 0;
        let mut ptr2 = CSlicePtr::<i32, _>::null();
        let mut len2: c_int = 0;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: Testing that swap panics when allocators differ.
            let mut handle1 = unsafe { ptr1.with_len_vec_mut_in(&mut len1, &alloc1) };
            // SAFETY: Testing that swap panics when allocators differ.
            let mut handle2 = unsafe { ptr2.with_len_vec_mut_in(&mut len2, &alloc2) };
            handle1.swap(&mut handle2);
        }));
        assert!(result.is_err());
    }

    #[gtest]
    fn c_vec_ref_mut_clear_already_empty() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        // Clearing an already-empty slice should not panic.
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_clear_drops_elements() {
        static DROPPED: AtomicU8 = AtomicU8::new(0);
        struct Foo(u8);
        impl Drop for Foo {
            fn drop(&mut self) {
                DROPPED.fetch_add(self.0, Ordering::Relaxed);
            }
        }

        let (mut ptr, mut len) = unsafe { malloc_array([Foo(1), Foo(2)]) };
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };

        assert_that!(DROPPED.load(Ordering::Relaxed), eq(0));
        handle.clear();
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(3));
        handle.clear(); // Should be a no-op.
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(3));
    }

    #[gtest]
    fn c_vec_ref_mut_clear_panic_safety() {
        struct PanickingDrop(#[allow(dead_code)] u8);
        impl Drop for PanickingDrop {
            fn drop(&mut self) {
                panic!("intentional drop panic");
            }
        }

        let (mut ptr, mut len) = unsafe { malloc_array([PanickingDrop(1)]) };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
            handle.clear();
        }));

        // Underlying pointer and len should already be reset to null/0 despite the panic.
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_usize() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: usize = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle.push_back(100);
        handle.push_back(200);
        assert_that!(&*handle, container_eq([100, 200]));
        assert_that!(handle.len(), eq(2));
        handle.clear();
        assert_that!(len, eq(0usize));
        assert!(ptr.is_null());
    }

    #[gtest]
    fn c_vec_ref_mut_u32() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: u32 = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle.push_back(42);
        assert_that!(&*handle, container_eq([42]));
        assert_that!(handle.len(), eq(1));
        handle.clear();
        assert_that!(len, eq(0u32));
    }

    #[gtest]
    fn c_vec_ref_mut_u64() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: u64 = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        handle.push_back(77);
        assert_that!(&*handle, container_eq([77]));
        assert_that!(handle.len(), eq(1));
        handle.clear();
        assert_that!(len, eq(0u64));
    }

    #[gtest]
    fn c_vec_ref_mut_u8_overflow() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: u8 = u8::MAX;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut len) };
        let result = handle.try_push_back(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }
}

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
use crate::c_buf::{max_slice_len, CBufLen, CBufPtr};
use allocator_api2::alloc::{Allocator, Layout};
use core::ptr::{self, NonNull};

/// The minimum non-zero capacity to allocate when a capacity-tracking vector
/// first grows from empty.
///
/// This mirrors the amortisation strategy of the standard library's `Vec`,
/// avoiding a burst of tiny reallocations for small element types while not
/// over-allocating for large ones.
const fn min_non_zero_cap<T>() -> usize {
    if core::mem::size_of::<T>() == 1 {
        8
    } else if core::mem::size_of::<T>() <= 1024 {
        4
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
//  CVecRefMut — mutable vector handle
// ---------------------------------------------------------------------------

/// A borrowed mutable handle over a `(*mut T, L)` pair representing a dynamic C vector.
///
/// Created via [`CBufPtr::as_vec_mut`] or [`CBufPtr::as_vec_mut_in`].
/// Provides mutable slice access and vector mutation operations ([`push_back`](Self::push_back),
/// [`try_push_back`](Self::try_push_back), [`clear`](Self::clear), [`replace`](Self::replace), [`swap`](Self::swap)).
///
/// Slice access is provided through [`Deref`](core::ops::Deref) and
/// [`DerefMut`](core::ops::DerefMut), which correctly tie the returned
/// slice's lifetime to the borrow of this handle.
///
/// # Capacity
///
/// A handle may optionally track a separate *capacity* — the number of elements
/// the allocation can hold. Such a handle is created via [`CBufPtr::as_vec_mut_with_cap`] or
/// [`CBufPtr::as_vec_mut_with_cap_in`]. When capacity is tracked,
/// [`push_back`](Self::push_back) appends into spare capacity without reallocating
/// and grows geometrically once full, matching `Vec`'s amortised behaviour. When
/// capacity is not tracked (the default), the allocation always holds exactly `len`
/// elements and every push reallocates.
///
/// # Safety Invariant
///
/// - Let `cap` be the tracked capacity if present, or `len` otherwise. If `cap > 0`,
///   `ptr` is non-null and points to an allocation of exactly `cap` elements of `T`
///   (allocated by `A`), and `cap <= isize::MAX / size_of::<T>()`.
/// - `len` is the number of initialised, leading elements and satisfies `len <= cap`.
///   If `len == 0`, no element is initialised.
pub struct CVecRefMut<'a, T, L: CBufLen, A: Allocator = LibcAlloc, C: CBufLen = L> {
    pub(crate) ptr: &'a mut CBufPtr<T, A>,
    pub(crate) len: &'a mut L,
    /// Optional capacity field. When `Some`, the allocation holds `*capacity`
    /// elements; when `None`, the allocation holds exactly `*len` elements.
    pub(crate) capacity: Option<&'a mut C>,
    pub(crate) alloc: A,
}

impl<'a, T, L: CBufLen, A: Allocator, C: CBufLen> CVecRefMut<'a, T, L, A, C> {
    /// Return the slice view with the lifetime tied to the borrow.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let len = self.len();
        if self.ptr.is_null() || len == 0 {
            return &[];
        }
        // SAFETY:
        // - Since `ptr` is not null, the invariants for `CBufPtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CBufPtr` owns the underlying array, so the pointer is valid for
        //   reads for the lifetime of this object (&self, created from a `CBufPtr`).
        // - The invariant for `CVecRefMut` guarantees that `len` is the valid length of the array
        //   (as per CBufLen's safety contract) pointed to by `ptr`.
        // - `L: CBufLen` guarantees that `try_into()` is deterministic and pure.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), len) }
    }

    /// Return the mutable slice view with the lifetime tied to the borrow.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let len = self.len();
        if self.ptr.is_null() || len == 0 {
            return &mut [];
        }
        // SAFETY:
        // - Since `ptr` is not null, the invariants for `CBufPtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CBufPtr` owns the underlying array, so the pointer is valid for
        //   reads for the lifetime of this object (&self, created from a `CBufPtr`).
        // - The invariant for `CVecRefMut` guarantees that `len` is the valid length of the array
        //   pointed to by `ptr`.
        // - `L: CBufLen` guarantees that `try_into()` is deterministic and pure.
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
        (*self.len).try_into().expect("CVecRefMut: len is negative")
    }

    /// Return `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of elements the allocation can hold without reallocating,
    /// or `None` if this handle does not track capacity separately from length.
    ///
    /// When `None`, the allocation always holds exactly [`len`](Self::len) elements.
    #[inline]
    pub fn capacity(&self) -> Option<usize> {
        self.capacity.as_ref().map(|c| (**c).try_into().expect("CVecRefMut: capacity is negative"))
    }

    /// Append an element.
    ///
    /// If this handle [tracks capacity](Self::capacity) and spare capacity is
    /// available, the element is written in place without reallocating. Otherwise
    /// the buffer is reallocated via the configured [`Allocator`]: geometrically
    /// (roughly doubling) when tracking capacity, or by exactly one slot when not.
    ///
    /// Returns `Err(value)` if the array cannot grow (out of memory, or `len`/`cap`
    /// overflow), giving the caller the element back.
    pub fn try_push_back(&mut self, value: T) -> Result<(), T> {
        let old_len = self.len();
        let Some(new_len) = old_len.checked_add(1).filter(|&n| n <= max_slice_len::<T>()) else {
            core::hint::cold_path();
            return Err(value);
        };
        let Ok(new_len_val) = L::try_from(new_len) else {
            core::hint::cold_path();
            return Err(value);
        };

        // Currently allocated capacity. When capacity is not tracked, the
        // allocation holds exactly `old_len` elements.
        let old_cap = self.capacity();

        if old_cap.is_none_or(|cap| cap == old_len) {
            // Slow path: (re)allocate. Grow geometrically when tracking capacity to
            // amortise future pushes; otherwise grow by exactly one slot.
            let new_cap = match old_cap {
                Some(0) => min_non_zero_cap::<T>(),
                // This is guaranteed to increase capacity at least by 1 because we already
                // asserted that old_len < max_slice_len::<T>() and that old_len == old_cap.
                Some(cap) => cap.saturating_mul(2).min(max_slice_len::<T>()),
                None => new_len,
            };
            let Ok(new_cap_val) = C::try_from(new_cap) else {
                core::hint::cold_path();
                return Err(value);
            };

            let Ok(new_layout) = Layout::array::<T>(new_cap) else {
                core::hint::cold_path();
                return Err(value);
            };
            let Ok(old_layout) = Layout::array::<T>(old_cap.unwrap_or(old_len).max(1)) else {
                core::hint::cold_path();
                return Err(value);
            };

            let alloc_result = match NonNull::new(self.ptr.as_ptr() as *mut u8) {
                // SAFETY: If `old_ptr` is non-null, it was allocated by `self.alloc` with
                // `old_layout` (`old_cap` elements, or `old_len` when capacity is untracked).
                Some(old_ptr) => unsafe { self.alloc.grow(old_ptr, old_layout, new_layout) },
                None => self.alloc.allocate(new_layout),
            };
            let Ok(new_ptr) = alloc_result else {
                core::hint::cold_path();
                return Err(value);
            };
            // SAFETY: `new_ptr` is non-null and points to an allocation of `new_cap`
            // elements of `T`.
            *self.ptr = unsafe { CBufPtr::from_raw(new_ptr.as_ptr() as *mut T) };
            if let Some(cap) = self.capacity.as_deref_mut() {
                *cap = new_cap_val;
            }
        }

        // SAFETY: By the safety invariant of `CVecRefMut` the allocation has room for `old_cap`
        // elements. The slot at index `old_len` is uninitialised,
        // so `ptr::write` correctly places `value` without dropping anything.
        unsafe { ptr::write(self.ptr.as_ptr().add(old_len), value) };
        *self.len = new_len_val;
        Ok(())
    }

    /// Append an element, growing the allocation if necessary.
    ///
    /// See [`try_push_back`](Self::try_push_back) for the growth strategy.
    ///
    /// # Panics
    /// Panics if the array cannot grow (out of memory, or `len`/`cap` overflow).
    pub fn push_back(&mut self, value: T) {
        if self.try_push_back(value).is_err() {
            core::hint::cold_path();
            panic!("CVecRefMut: allocation failed");
        }
    }

    /// Drop all elements and deallocate the buffer using the configured [`Allocator`].
    pub fn clear(&mut self) {
        let len = self.len();
        // The allocation spans `capacity` elements when tracked, otherwise exactly `len`.
        let cap = self.capacity().unwrap_or(len);
        // We replace the pointer and lengths first to leave the handle in a valid, empty
        // state immediately. This is necessary for panic safety: if dropping elements
        // panics, the handle won't point to invalid memory.
        let old_ptr = core::mem::replace(self.ptr, CBufPtr::null());
        *self.len = L::default();
        if let Some(cap_ref) = self.capacity.as_deref_mut() {
            *cap_ref = C::default();
        }

        if let Some(non_null) = NonNull::new(old_ptr.as_ptr() as *mut u8) {
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
            // The allocation covers `cap` elements, so it must be freed with a
            // `cap`-sized layout (which equals `len` when capacity is not tracked).
            let layout = Layout::array::<T>(cap.max(1)).expect("CVecRefMut: valid layout");
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

impl<'a, T, L: CBufLen, A: Allocator + PartialEq, C: CBufLen> CVecRefMut<'a, T, L, A, C> {
    /// Swap the underlying pointer, len, and capacity with another handle that uses
    /// the same allocator.
    ///
    /// # Panics
    /// Panics if `self` and `other` do not share the same allocator instance, or if
    /// exactly one of the two handles tracks capacity (both must track capacity, or
    /// neither).
    pub fn swap(&mut self, other: &mut CVecRefMut<'_, T, L, A, C>) {
        assert!(self.alloc == other.alloc, "CVecRefMut::swap: handles must use the same allocator");
        core::mem::swap(self.ptr, other.ptr);
        core::mem::swap(self.len, other.len);
        match (self.capacity.as_deref_mut(), other.capacity.as_deref_mut()) {
            (Some(a), Some(b)) => core::mem::swap(a, b),
            (None, None) => {}
            _ => panic!("CVecRefMut::swap: both handles must track capacity, or neither"),
        }
    }
}

impl<T, L: CBufLen, A: Allocator, C: CBufLen> core::ops::Deref for CVecRefMut<'_, T, L, A, C> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, L: CBufLen, A: Allocator, C: CBufLen> core::ops::DerefMut for CVecRefMut<'_, T, L, A, C> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T: core::fmt::Debug, L: CBufLen, A: Allocator, C: CBufLen> core::fmt::Debug
    for CVecRefMut<'_, T, L, A, C>
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
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let handle = unsafe { ptr.as_vec_mut(&mut len) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
    }

    #[gtest]
    fn c_vec_ref_mut_nonnull_ptr_zero_len() {
        // Simulate a C struct where a buffer was allocated but len is 0.
        // clear() must still free the buffer.
        let mut ptr = unsafe { CBufPtr::<i32>::from_raw(libc::malloc(16) as *mut i32) };
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
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
        let mut ptr = unsafe { CBufPtr::<i32>::from_raw(libc::malloc(16) as *mut i32) };
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        handle.push_back(123);
        assert_that!(&*handle, container_eq([123]));
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_deref() {
        let (mut ptr, mut len) = unsafe { malloc_array([5, 6, 7]) };
        let handle = unsafe { ptr.as_vec_mut(&mut len) };
        assert_that!(&*handle, container_eq([5, 6, 7]));
        assert_that!(handle.len(), eq(3));

        // Clean up.
        let mut handle = handle;
        handle.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_deref_mut() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        handle[0] = 99;
        assert_that!(&*handle, container_eq([99, 2, 3]));
        handle.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_push_back_to_empty() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
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
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
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
        let mut ptr = CBufPtr::null();
        let mut count: c_int = 0;
        {
            // SAFETY: Null pointer with length 0 is safe.
            let mut handle = unsafe { ptr.as_vec_mut_in(&mut count, &alloc) };
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
        let mut ptr = unsafe { CBufPtr::from_raw(slice.as_ptr() as *mut i32) };
        let mut count: c_int = 0;
        {
            // SAFETY: ptr is non-null, count is 0, allocated via alloc.
            let mut handle = unsafe { ptr.as_vec_mut_in(&mut count, &alloc) };
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
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = c_int::MAX;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        let result = handle.try_push_back(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }

    #[gtest]
    fn c_vec_ref_mut_clear_nonempty() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
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
            let mut handle1 = unsafe { ptr1.as_vec_mut(&mut len1) };
            let mut handle2 = unsafe { ptr2.as_vec_mut(&mut len2) };

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
        let mut handle1 = unsafe { ptr1.as_vec_mut(&mut len1) };
        handle1.clear();
        let mut handle2 = unsafe { ptr2.as_vec_mut(&mut len2) };
        handle2.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_swap_different_allocators_panics() {
        let alloc1 = TrackingAlloc::default();
        let alloc2 = TrackingAlloc::default();

        let mut ptr1 = CBufPtr::<i32, _>::null();
        let mut len1: c_int = 0;
        let mut ptr2 = CBufPtr::<i32, _>::null();
        let mut len2: c_int = 0;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: Testing that swap panics when allocators differ.
            let mut handle1 = unsafe { ptr1.as_vec_mut_in(&mut len1, &alloc1) };
            // SAFETY: Testing that swap panics when allocators differ.
            let mut handle2 = unsafe { ptr2.as_vec_mut_in(&mut len2, &alloc2) };
            handle1.swap(&mut handle2);
        }));
        assert!(result.is_err());
    }

    #[gtest]
    fn c_vec_ref_mut_clear_already_empty() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
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
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };

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
            let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
            handle.clear();
        }));

        // Underlying pointer and len should already be reset to null/0 despite the panic.
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_usize() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: usize = 0;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
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
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: u32 = 0;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        handle.push_back(42);
        assert_that!(&*handle, container_eq([42]));
        assert_that!(handle.len(), eq(1));
        handle.clear();
        assert_that!(len, eq(0u32));
    }

    #[gtest]
    fn c_vec_ref_mut_u64() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: u64 = 0;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        handle.push_back(77);
        assert_that!(&*handle, container_eq([77]));
        assert_that!(handle.len(), eq(1));
        handle.clear();
        assert_that!(len, eq(0u64));
    }

    #[gtest]
    fn c_vec_ref_mut_u8_overflow() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: u8 = u8::MAX;
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        let result = handle.try_push_back(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }

    // -----------------------------------------------------------------------
    //  Capacity-tracking (as_vec_mut_with_cap) tests
    // -----------------------------------------------------------------------

    #[gtest]
    fn c_vec_ref_mut_with_cap_reuses_spare_capacity() {
        let alloc = TrackingAlloc::default();
        let mut ptr = CBufPtr::null();
        let mut len: c_int = 0;
        let mut cap: c_int = 0;
        {
            // SAFETY: null pointer with len 0 and cap 0 is safe.
            let mut handle = unsafe { ptr.as_vec_mut_with_cap_in(&mut len, &mut cap, &alloc) };

            // First push allocates a buffer with `min_non_zero_cap::<i32>() == 4` slots.
            handle.push_back(1);
            assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(0));
            assert_that!(handle.capacity(), some(eq(4)));

            // The next three pushes reuse spare capacity: no allocations at all.
            handle.push_back(2);
            handle.push_back(3);
            handle.push_back(4);
            assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(0));
            assert_that!(handle.len(), eq(4));
            assert_that!(handle.capacity(), some(eq(4)));

            // The fifth push is full, so it grows geometrically to 8.
            handle.push_back(5);
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(1));
            assert_that!(handle.capacity(), some(eq(8)));
            assert_that!(&*handle, container_eq([1, 2, 3, 4, 5]));

            handle.clear();
            assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(1));
        }
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
        assert_that!(cap, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_with_cap_distinct_length_types() {
        // The capacity field may use a different integer type than the length field.
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let mut cap: usize = 0;
        {
            let mut handle = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
            handle.push_back(10);
            handle.push_back(20);
            assert_that!(&*handle, container_eq([10, 20]));
            handle.clear();
        }
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
        assert_that!(cap, eq(0usize));
    }

    #[gtest]
    fn c_vec_ref_mut_with_cap_existing_spare_buffer() {
        // Simulate a C struct with a buffer that already has spare capacity.
        let alloc = TrackingAlloc::default();
        let slice = (&alloc).allocate(Layout::array::<i32>(4).unwrap()).unwrap();
        let raw = slice.as_ptr() as *mut i32;
        // Initialise the first two elements.
        unsafe {
            ptr::write(raw, 1);
            ptr::write(raw.add(1), 2);
        }
        // SAFETY: `raw` was allocated by `alloc` for 4 i32s.
        let mut ptr = unsafe { CBufPtr::from_raw(raw) };
        let mut len: c_int = 2;
        let mut cap: c_int = 4;
        {
            // SAFETY: ptr holds 4 slots, 2 initialised, allocated via `alloc`.
            let mut handle = unsafe { ptr.as_vec_mut_with_cap_in(&mut len, &mut cap, &alloc) };
            assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));

            // Pushing into spare capacity must not reallocate.
            handle.push_back(3);
            handle.push_back(4);
            assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));
            assert_that!(alloc.grow_count.load(Ordering::SeqCst), eq(0));
            assert_that!(&*handle, container_eq([1, 2, 3, 4]));

            handle.clear();
            assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(1));
        }
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
        assert_that!(cap, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_without_cap_reports_no_capacity() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let handle = unsafe { ptr.as_vec_mut(&mut len) };
        assert_that!(handle.capacity(), none());
    }

    #[gtest]
    #[should_panic(expected = "CBufPtr: len 3 exceeds cap 2")]
    fn c_vec_ref_mut_with_cap_len_exceeds_cap_panics() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 3;
        let mut cap: c_int = 2;
        let _ = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
    }

    #[gtest]
    #[should_panic(expected = "CBufPtr: cap is negative")]
    fn c_vec_ref_mut_with_cap_negative_cap_panics() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let mut cap: c_int = -1;
        let _ = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
    }

    #[gtest]
    fn c_vec_ref_mut_with_cap_clear_drops_elements() {
        static DROPPED: AtomicU8 = AtomicU8::new(0);
        struct Foo(u8);
        impl Drop for Foo {
            fn drop(&mut self) {
                DROPPED.fetch_add(self.0, Ordering::Relaxed);
            }
        }

        let mut ptr = CBufPtr::<Foo>::null();
        let mut len: c_int = 0;
        let mut cap: c_int = 0;
        let mut handle = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
        handle.push_back(Foo(1));
        handle.push_back(Foo(2));
        // Spare capacity exists, but only the two initialised elements must be dropped.
        assert!(handle.capacity().unwrap() >= 2);
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(0));
        handle.clear();
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(3));
    }

    #[gtest]
    fn c_vec_ref_mut_with_cap_swap() {
        let mut ptr1 = CBufPtr::<i32>::null();
        let mut len1: c_int = 0;
        let mut cap1: c_int = 0;
        let mut ptr2 = CBufPtr::<i32>::null();
        let mut len2: c_int = 0;
        let mut cap2: c_int = 0;
        {
            let mut handle1 = unsafe { ptr1.as_vec_mut_with_cap(&mut len1, &mut cap1) };
            handle1.push_back(1);
            handle1.push_back(2);
            let mut handle2 = unsafe { ptr2.as_vec_mut_with_cap(&mut len2, &mut cap2) };
            handle2.push_back(9);

            handle1.swap(&mut handle2);
            assert_that!(&*handle1, container_eq([9]));
            assert_that!(&*handle2, container_eq([1, 2]));

            handle1.clear();
            handle2.clear();
        }
        assert_that!(len1, eq(0));
        assert_that!(len2, eq(0));
    }
}

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
//! vector operations like [`push_back`](CVecRefMut::push_back),
//! [`try_push_back`](CVecRefMut::try_push_back), [`clear`](CVecRefMut::clear), and
//! [`swap`](CVecRefMut::swap).

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
/// [`try_push_back`](Self::try_push_back), [`clear`](Self::clear), [`swap`](Self::swap)).
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
/// - Let `cap` be the tracked capacity if present, or `len` otherwise. If `cap > 0`, `ptr` is
///   non-null and points to an allocation of exactly `cap` elements of `T` (allocated by `A`), and
///   `cap <= isize::MAX / size_of::<T>()`.
/// - `len` is the number of initialised, leading elements and satisfies `len <= cap`. If `len ==
///   0`, no element is initialised.
/// - `alloc` is the allocator used for all allocations.
/// - `capacity` is `Some` if and only if this handle tracks capacity.
pub struct CVecRefMut<'a, T, L: CBufLen, A: Allocator = LibcAlloc, C: CBufLen = L> {
    // Safety invariant: Let `cap` be the non-negative `usize` value of `*capacity` when `capacity`
    // is `Some`, or the non-negative `usize` value of `*len` when `capacity` is `None`. Either
    // `ptr` is null and `cap == 0`, or `ptr` is non-null, aligned for `T`, and points to a live
    // allocation of `cap.max(1)` elements of `T` created by `alloc` with
    // `cap <= isize::MAX as usize / size_of::<T>()` and the first `*len` elements initialized.
    pub(crate) ptr: &'a mut CBufPtr<T, A>,
    // Safety invariant: `*len` converts via `L::try_into` to `Ok(n)` where `0 <= n <= cap` is the
    // exact number of initialized leading `T` elements at `ptr`.
    pub(crate) len: &'a mut L,
    // Safety invariant: When `Some(c)`, `*c` converts via `C::try_into` to `Ok(cap)` where
    // `cap >= len` and `cap <= isize::MAX as usize / size_of::<T>()` is the element capacity of
    // the allocation at `ptr` (or `0` when `ptr` is null).
    /// Optional capacity field. When `Some`, the allocation holds `*capacity`
    /// elements; when `None`, the allocation holds exactly `*len` elements.
    pub(crate) capacity: Option<&'a mut C>,
    // Safety invariant: `alloc` is the allocator instance that created (and must resize/free) any
    // non-null allocation stored in `*ptr`.
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
        // - By the safety-usable invariant of `self.len()` and the field safety invariants of
        //   `CVecRefMut`, since `!self.ptr.is_null()` and `len > 0`, `self.ptr` is non-null,
        //   properly aligned for `T`, and points to `len` consecutive initialized `T` elements
        //   within a single live allocation of size `len * size_of::<T>() <= isize::MAX as usize`.
        // - `L: CBufLen` guarantees that `try_into()` is deterministic and pure.
        // - The shared borrow `&self` prevents any mutation of the underlying elements for the
        //   lifetime of the returned `&[T]` slice.
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
        // - By the safety-usable invariant of `self.len()` and the field safety invariants of
        //   `CVecRefMut`, since `!self.ptr.is_null()` and `len > 0`, `self.ptr` is non-null,
        //   properly aligned for `T`, and points to `len` consecutive initialized `T` elements
        //   within a single live allocation of size `len * size_of::<T>() <= isize::MAX as usize`.
        // - `L: CBufLen` guarantees that `try_into()` is deterministic and pure.
        // - The exclusive borrow `&mut self` guarantees no other reference aliases the elements for
        //   the lifetime of the returned `&mut [T]` slice.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }

    /// Return a reference to the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Return the number of elements in the vector.
    ///
    /// # Safety-usable invariant
    ///
    /// Returns the exact number `n` of initialized leading `T` elements in `self.ptr`, with
    /// `n <= self.capacity().unwrap_or(n) <= isize::MAX as usize / size_of::<T>()`, and if `n > 0`,
    /// `self.ptr` is non-null.
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
    ///
    /// # Safety-usable invariant
    ///
    /// When `Some(cap)`, `self.len() <= cap <= isize::MAX as usize / size_of::<T>()`, and if
    /// `cap > 0`, `self.ptr` is non-null and points to an allocation of `cap` elements of `T`
    /// created by `self.alloc`.
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
            let mut new_cap = match old_cap {
                Some(0) => min_non_zero_cap::<T>(),
                // This is guaranteed to increase capacity at least by 1 because we already
                // asserted that old_len < max_slice_len::<T>() and that old_len == old_cap.
                Some(cap) => cap.saturating_mul(2).min(max_slice_len::<T>()),
                None => new_len,
            };
            let new_cap_val = match C::try_from(new_cap) {
                Ok(val) => val,
                Err(_) => {
                    core::hint::cold_path();
                    if new_len > C::MAX {
                        // Cannot represent the new length in `capacity`, so we cannot grow.
                        return Err(value);
                    } else {
                        // We cannot represent old_cap * 2, but new_len is representable.
                        // Reduce cap to the maximum representable value.
                        new_cap = C::MAX;
                        C::try_from(C::MAX).unwrap()
                    }
                }
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
                // SAFETY: By `CVecRefMut`'s field invariants, when `old_ptr` is non-null, it is
                // currently allocated via `self.alloc` with `old_layout`, and `new_cap >=
                // old_cap.unwrap_or(old_len).max(1)` implies `new_layout.size() >=
                // old_layout.size()`, satisfying `Allocator::grow`.
                Some(old_ptr) => unsafe { self.alloc.grow(old_ptr, old_layout, new_layout) },
                None => self.alloc.allocate(new_layout),
            };
            let Ok(new_ptr) = alloc_result else {
                core::hint::cold_path();
                return Err(value);
            };
            // SAFETY: `new_ptr` was returned by `self.alloc` for `new_layout`
            // (`Layout::array::<T>(new_cap)`), so `new_ptr.as_ptr() as *mut T` is
            // non-null, aligned for `T`, and allocated by `A`.
            *self.ptr = unsafe { CBufPtr::from_raw(new_ptr.as_ptr() as *mut T) };
            if let Some(cap) = self.capacity.as_deref_mut() {
                *cap = new_cap_val;
            }
        }

        // SAFETY: By `CVecRefMut`'s field invariants and the reallocation branch above, `self.ptr`
        // is non-null and points to a live allocation of at least `new_len = old_len + 1` elements
        // of `T` (`new_len * size_of::<T>() <= isize::MAX as usize`), so offsetting by `old_len`
        // remains in bounds of that allocation without wrapping.
        let dst = unsafe { self.ptr.as_ptr().add(old_len) };
        // SAFETY: `dst` is non-null, aligned for `T`, and points to the uninitialized slot at index
        // `old_len` within the exclusively owned allocation.
        unsafe { ptr::write(dst, value) };
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
                // Safety invariant: `alloc` is the allocator instance that created `ptr`.
                alloc: &'g A,
                // Safety invariant: `ptr` is a live block of memory allocated by `alloc` with
                // `layout` that is exclusively owned by this guard and has not yet been freed.
                ptr: NonNull<u8>,
                // Safety invariant: `layout` is the exact `Layout` used to allocate `ptr`.
                layout: Layout,
            }
            impl<'g, A: Allocator> Drop for AllocDropGuard<'g, A> {
                fn drop(&mut self) {
                    // SAFETY: By `AllocDropGuard`'s field safety invariants, `self.ptr` is
                    // currently allocated via `self.alloc` with `self.layout`, and no active
                    // references to the block remain when `AllocDropGuard::drop` runs.
                    unsafe { self.alloc.deallocate(self.ptr, self.layout) };
                }
            }
            // The allocation covers `cap` elements, so it must be freed with a
            // `cap`-sized layout (which equals `len` when capacity is not tracked).
            let layout = Layout::array::<T>(cap.max(1)).expect("CVecRefMut: valid layout");
            let _guard = AllocDropGuard { alloc: &self.alloc, ptr: non_null, layout };

            if len > 0 {
                let slice = ptr::slice_from_raw_parts_mut(old_ptr.as_ptr(), len);
                // SAFETY: By `CVecRefMut`'s field invariants prior to resetting `self.ptr`,
                // `old_ptr` is non-null, aligned for `T`, and points to `len` consecutive
                // initialized `T` elements in a single live allocation exclusively owned by this
                // call. `_guard` only deallocates the raw storage after `drop_in_place` completes
                // (or unwinds), so the elements are never dropped twice.
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
        // panic safety: swap capacity first so that we don't panic after swapping pointers.
        match (self.capacity.as_deref_mut(), other.capacity.as_deref_mut()) {
            (Some(a), Some(b)) => core::mem::swap(a, b),
            (None, None) => {}
            _ => panic!("CVecRefMut::swap: both handles must track capacity, or neither"),
        }
        core::mem::swap(self.ptr, other.ptr);
        core::mem::swap(self.len, other.len);
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
    use super::*;
    use crate::testing::*;
    use core::sync::atomic::{AtomicU8, Ordering};
    use googletest::prelude::*;
    use std::ffi::c_int;

    #[gtest]
    fn c_vec_ref_mut_null_ptr() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        // SAFETY: `ptr` is null and `len` is `0`.
        let handle = unsafe { ptr.as_vec_mut(&mut len) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
    }

    #[gtest]
    fn c_vec_ref_mut_nonnull_ptr_zero_len() {
        // Simulate a C struct where a buffer was allocated but len is 0.
        // clear() must still free the buffer.
        let mut ptr = malloc_uninit_i32_buf();
        let mut len: c_int = 0;
        // SAFETY: `ptr` was allocated by `LibcAlloc` with 16 bytes (`>= Layout::array::<i32>(1)`)
        // and `len` is `0` initialized elements.
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
        let mut ptr = malloc_uninit_i32_buf();
        let mut len: c_int = 0;
        // SAFETY: `ptr` was allocated by `LibcAlloc` with 16 bytes (`>= Layout::array::<i32>(1)`)
        // and `len` is `0` initialized elements.
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        handle.push_back(123);
        assert_that!(&*handle, container_eq([123]));
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_deref() {
        let (mut ptr, mut len) = malloc_array([5, 6, 7]);
        // SAFETY: `ptr` points to `len` (`3`) initialized `i32` elements allocated by `LibcAlloc`.
        let handle = unsafe { ptr.as_vec_mut(&mut len) };
        assert_that!(&*handle, container_eq([5, 6, 7]));
        assert_that!(handle.len(), eq(3));

        // Clean up.
        let mut handle = handle;
        handle.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_deref_mut() {
        let (mut ptr, mut len) = malloc_array([1, 2, 3]);
        // SAFETY: `ptr` points to `len` (`3`) initialized `i32` elements allocated by `LibcAlloc`.
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        handle[0] = 99;
        assert_that!(&*handle, container_eq([99, 2, 3]));
        handle.clear();
    }

    #[gtest]
    fn c_vec_ref_mut_push_back_to_empty() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        // SAFETY: `ptr` is null and `len` is `0`.
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
        let (mut ptr, mut len) = malloc_array([10]);
        // SAFETY: `ptr` points to `len` (`1`) initialized `i32` element allocated by `LibcAlloc`.
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
            // SAFETY: `ptr` is null and `count` is `0`.
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
        // SAFETY: `slice` was just allocated by `&alloc` with `Layout::new::<i32>()` (non-null and
        // aligned for `i32`).
        let mut ptr = unsafe { CBufPtr::from_raw(slice.as_ptr() as *mut i32) };
        let mut count: c_int = 0;
        {
            // SAFETY: `ptr` is non-null, allocated via `&alloc`, and `count` is `0` initialized
            // elements.
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
        let mut dummy = 0i32;
        // SAFETY: `&mut dummy` is non-null and aligned for `i32`; `try_push_back` returns `Err` on
        // `c_int::MAX` overflow before touching the pointer or allocator.
        let mut ptr = unsafe { CBufPtr::from_raw(&mut dummy) };
        let mut len: c_int = c_int::MAX;
        // SAFETY: Test-only synthetic handle with `len == c_int::MAX`; `try_push_back` checks
        // `L::try_from(old_len + 1)` and returns `Err` before dereferencing or reallocating `ptr`.
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        let result = handle.try_push_back(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }

    #[gtest]
    fn c_vec_ref_mut_clear_nonempty() {
        let (mut ptr, mut len) = malloc_array([1, 2, 3]);
        // SAFETY: `ptr` points to `len` (`3`) initialized `i32` elements allocated by `LibcAlloc`.
        let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
        handle.clear();
        assert_that!(handle.len(), eq(0));
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_swap() {
        let (mut ptr1, mut len1) = malloc_array([1, 2, 3]);
        let (mut ptr2, mut len2) = malloc_array([4, 5]);

        {
            // SAFETY: `ptr1` points to `len1` (`3`) initialized `i32` elements allocated by
            // `LibcAlloc`.
            let mut handle1 = unsafe { ptr1.as_vec_mut(&mut len1) };
            // SAFETY: `ptr2` points to `len2` (`2`) initialized `i32` elements allocated by
            // `LibcAlloc`.
            let mut handle2 = unsafe { ptr2.as_vec_mut(&mut len2) };

            handle1.swap(&mut handle2);

            assert_that!(&*handle1, container_eq([4, 5]));
            assert_that!(&*handle2, container_eq([1, 2, 3]));
        }

        // Verify that the underlying values were swapped.
        assert_that!(len1, eq(2));
        assert_that!(len2, eq(3));
        // SAFETY: After `swap`, `ptr1` points to `len1` (`2`) initialized `i32` elements.
        assert_that!(unsafe { ptr1.with_len(len1) }, container_eq([4, 5]));
        // SAFETY: After `swap`, `ptr2` points to `len2` (`3`) initialized `i32` elements.
        assert_that!(unsafe { ptr2.with_len(len2) }, container_eq([1, 2, 3]));

        // Clean up both handles.
        // SAFETY: `ptr1` points to `len1` (`2`) initialized `i32` elements allocated by
        // `LibcAlloc`.
        let mut handle1 = unsafe { ptr1.as_vec_mut(&mut len1) };
        handle1.clear();
        // SAFETY: `ptr2` points to `len2` (`3`) initialized `i32` elements allocated by
        // `LibcAlloc`.
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
            // SAFETY: `ptr1` is null and `len1` is `0`.
            let mut handle1 = unsafe { ptr1.as_vec_mut_in(&mut len1, &alloc1) };
            // SAFETY: `ptr2` is null and `len2` is `0`.
            let mut handle2 = unsafe { ptr2.as_vec_mut_in(&mut len2, &alloc2) };
            handle1.swap(&mut handle2);
        }));
        assert!(result.is_err());
    }

    #[gtest]
    fn c_vec_ref_mut_clear_already_empty() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        // SAFETY: `ptr` is null and `len` is `0`.
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

        let (mut ptr, mut len) = malloc_array([Foo(1), Foo(2)]);
        // SAFETY: `ptr` points to `len` (`2`) initialized `Foo` elements allocated by `LibcAlloc`.
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

        let (mut ptr, mut len) = malloc_array([PanickingDrop(1)]);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: `ptr` points to `len` (`1`) initialized `PanickingDrop` element allocated by
            // `LibcAlloc`.
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
        // SAFETY: `ptr` is null and `len` is `0`.
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
        // SAFETY: `ptr` is null and `len` is 0.
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
        // SAFETY: `ptr` is null and `len` is 0.
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
        let mut len: u8 = 0;
        {
            // SAFETY: `ptr` is null and `len` is 0.
            let mut handle = unsafe { ptr.as_vec_mut(&mut len) };
            for i in 0..=254 {
                assert!(handle.try_push_back(i as i32).is_ok());
            }
            assert_that!(handle.len(), eq(255));

            let result = handle.try_push_back(999);
            assert!(result.is_err());
            assert_that!(result.unwrap_err(), eq(999));
            handle.clear();
        }
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
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
            // SAFETY: `ptr` is null and `len` and `cap` are 0.
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
            // SAFETY: `ptr` is null and `len` and `cap` are 0.
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
        // SAFETY: `raw` points to a freshly allocated, aligned buffer for 4 `i32`s.
        unsafe {
            ptr::write(raw, 1);
            ptr::write(raw.add(1), 2);
        }
        // SAFETY: `raw` is properly aligned for `i32` and allocated by `alloc` for 4 `i32`s.
        let mut ptr = unsafe { CBufPtr::from_raw(raw) };
        let mut len: c_int = 2;
        let mut cap: c_int = 4;
        {
            // SAFETY: `ptr` holds 4 slots (2 initialized), allocated via `alloc`.
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
        // SAFETY: `ptr` is null and `len` is 0.
        let handle = unsafe { ptr.as_vec_mut(&mut len) };
        assert_that!(handle.capacity(), none());
    }

    #[gtest]
    #[should_panic(expected = "CBufPtr: len 3 exceeds cap 2")]
    fn c_vec_ref_mut_with_cap_len_exceeds_cap_panics() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 3;
        let mut cap: c_int = 2;
        // SAFETY: `as_vec_mut_with_cap` validates `len <= cap` before performing any unsafe
        // operations and panics here as tested.
        let _ = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
    }

    #[gtest]
    #[should_panic(expected = "CBufPtr: cap is negative")]
    fn c_vec_ref_mut_with_cap_negative_cap_panics() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let mut cap: c_int = -1;
        // SAFETY: `as_vec_mut_with_cap` validates `cap >= 0` before performing any unsafe
        // operations and panics here as tested.
        let _ = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
    }

    #[gtest]
    #[should_panic(expected = "CBufPtr: null pointer with non-zero capacity 5")]
    fn c_vec_ref_mut_with_cap_null_ptr_non_zero_cap_panics() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 0;
        let mut cap: c_int = 5;
        // SAFETY: `as_vec_mut_with_cap` validates that a null pointer has `cap == 0` before
        // performing any unsafe operations and panics here as tested.
        let _ = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
    }

    #[gtest]
    #[should_panic(expected = "CBufPtr: null pointer with non-zero length 5")]
    fn c_vec_ref_mut_null_ptr_non_zero_len_panics() {
        let mut ptr = CBufPtr::<i32>::null();
        let mut len: c_int = 5;
        // SAFETY: `as_vec_mut` validates that a null pointer has `len == 0` before performing
        // any unsafe operations and panics here as tested.
        let _ = unsafe { ptr.as_vec_mut(&mut len) };
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
        // SAFETY: `ptr` is null and `len` and `cap` are 0.
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
            // SAFETY: `ptr1` is null and `len1` and `cap1` are 0.
            let mut handle1 = unsafe { ptr1.as_vec_mut_with_cap(&mut len1, &mut cap1) };
            handle1.push_back(1);
            handle1.push_back(2);
            // SAFETY: `ptr2` is null and `len2` and `cap2` are 0.
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

    #[gtest]
    fn c_vec_ref_mut_with_cap_u8_growth_past_128() {
        let mut ptr = CBufPtr::<u8>::null();
        let mut len: u8 = 0;
        let mut cap: u8 = 0;
        {
            // SAFETY: `ptr` is null and `len` and `cap` are 0.
            let mut handle = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
            for i in 0..=254 {
                assert!(handle.try_push_back(i as u8).is_ok());
            }
            assert_that!(handle.len(), eq(255));
            assert_that!(handle.capacity(), some(eq(255)));

            // 256th push cannot fit in u8 length or capacity.
            let res = handle.try_push_back(42);
            assert!(res.is_err());
            assert_that!(res.unwrap_err(), eq(42));
            handle.clear();
        }
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
        assert_that!(cap, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_with_cap_i8_growth_past_64() {
        let mut ptr = CBufPtr::<i8>::null();
        let mut len: i8 = 0;
        let mut cap: i8 = 0;
        {
            // SAFETY: `ptr` is null and `len` and `cap` are 0.
            let mut handle = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
            for i in 0..=126 {
                assert!(handle.try_push_back(i as i8).is_ok());
            }
            assert_that!(handle.len(), eq(127));
            assert_that!(handle.capacity(), some(eq(127)));

            // 128th push cannot fit in i8.
            let res = handle.try_push_back(99);
            assert!(res.is_err());
            assert_that!(res.unwrap_err(), eq(99));
            handle.clear();
        }
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
        assert_that!(cap, eq(0));
    }

    #[gtest]
    fn c_vec_ref_mut_with_cap_cap_type_smaller_than_len_type() {
        let mut ptr = CBufPtr::<u8>::null();
        let mut len: u16 = 0;
        let mut cap: u8 = 0;
        {
            // SAFETY: `ptr` is null and `len` and `cap` are 0.
            let mut handle = unsafe { ptr.as_vec_mut_with_cap(&mut len, &mut cap) };
            for i in 0..=254 {
                assert!(handle.try_push_back((i % 256) as u8).is_ok());
            }
            assert_that!(handle.len(), eq(255));
            assert_that!(handle.capacity(), some(eq(255)));

            // 256th push fits in len (u16) but exceeds cap (u8).
            let res = handle.try_push_back(100);
            assert!(res.is_err());
            assert_that!(res.unwrap_err(), eq(100));
            handle.clear();
        }
        assert!(ptr.is_null());
        assert_that!(len, eq(0u16));
        assert_that!(cap, eq(0u8));
    }

    #[gtest]
    #[should_panic(expected = "both handles must track capacity, or neither")]
    fn c_vec_ref_mut_swap_mismatched_capacity_tracking_panics() {
        let mut ptr1 = CBufPtr::<i32>::null();
        let mut len1: c_int = 0;
        let mut cap1: c_int = 0;
        let mut ptr2 = CBufPtr::<i32>::null();
        let mut len2: c_int = 0;
        // SAFETY: `ptr1` is null and `len1` and `cap1` are 0.
        let mut handle1 = unsafe { ptr1.as_vec_mut_with_cap(&mut len1, &mut cap1) };
        // SAFETY: `ptr2` is null and `len2` is 0.
        let mut handle2 = unsafe { ptr2.as_vec_mut(&mut len2) };
        handle1.swap(&mut handle2);
    }
}

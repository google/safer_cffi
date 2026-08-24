// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Safe handles for raw-pointer-backed arrays in C structs.
//!
//! These types centralise `unsafe` access to `(*mut T, L)` field pairs (where `L`
//! is an integer length type such as `c_int` or `usize`).
//! They ensure that all allocations and deallocations go through the **libc allocator**
//! (`malloc`/`free`), which would work even if the C and Rust allocators differ.
//!
//! # Usage
//!
//! Given a `#[repr(C)]` struct with raw pointer fields:
//!
//! ```
//! use std::os::raw::c_int;
//! use safer_cffi::{CSlicePtr, CSliceRefMut};
//!
//! #[repr(C)]
//! struct MyStruct {
//!     items: CSlicePtr<f32>,    // repr(transparent) wrapper around *mut f32
//!     item_len: c_int,
//! }
//!
//! impl MyStruct {
//!     // Shared accessor — returns &[T] (from &self).
//!     fn items(&self) -> &[f32] {
//!         // SAFETY: the length of `items` is `item_len`.
//!         unsafe { self.items.with_len(self.item_len) }
//!     }
//!
//!     // Mutable accessor — returns CSliceRefMut (from &mut self).
//!     fn items_mut(&mut self) -> CSliceRefMut<'_, f32, c_int> {
//!         // SAFETY: the length of `items` is `item_len`.
//!         unsafe { self.items.with_len_mut(&mut self.item_len) }
//!     }
//! }
//!
//! let mut my_struct = MyStruct {
//!     items: CSlicePtr::null(),
//!     item_len: 0,
//! };
//!
//! // Read:
//! let len = my_struct.items().len();
//! for item in my_struct.items() { /* ... */ }
//!
//! // Mutate:
//! my_struct.items_mut().add(40.0);
//! my_struct.items_mut()[0] += 2.0;
//!
//! // Clone impl:
//! let cloned_ptr: CSlicePtr<f32> = CSlicePtr::clone_and_leak(my_struct.items());
//!
//! // Drop impl:
//! my_struct.items_mut().clear();
//! ```

use crate::errors::AllocError;
use core::ptr;

/// The maximum slice length for type `T` that stays within the
/// [`isize::MAX`]-byte limit required by [`core::slice::from_raw_parts`].
///
/// On 64-bit platforms this vastly exceeds `c_int::MAX`, so any runtime
/// comparison against it is optimized away by the compiler.
const fn max_slice_len<T>() -> usize {
    if core::mem::size_of::<T>() == 0 {
        panic!("T has zero size")
    } else {
        isize::MAX as usize / core::mem::size_of::<T>()
    }
}

/// Static type assertions for `CSlicePtr`.
const fn slice_type_assertions<T>() {
    // Ensure that `T` is no larger than the alignment provided by `malloc`.
    // This is a constraint because we use `malloc` to allocate the buffer, and if `T` is
    // over-aligned, we cannot guarantee correct alignment.
    //
    // If this ever becomes an issue, consider using `aligned_alloc` instead of `malloc`.
    // We currently favor the universal availability of `malloc` over supporting
    // complex types.
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
    assert!(
        core::mem::align_of::<T>() <= MALLOC_ALIGN,
        "T is over-aligned for a standard malloc call"
    );

    // Ensure that `T` is not a zero-sized type (ZST). We only want to support C-native types,
    // and ZSTs are not a thing in C.
    assert!(core::mem::size_of::<T>() > 0, "T has zero size, which is not supported");
}

// ---------------------------------------------------------------------------
//  CSliceLen — integer types suitable for C slice lengths
// ---------------------------------------------------------------------------

/// An integer type that can represent the length of a C slice.
///
/// This trait is implemented for primitive integer types commonly used in C FFIs
/// (e.g. `c_int`, `usize`, `u32`, `i32`, etc.).
///
/// # Safety
///
/// Safe methods on [`CSliceRefMut`] (such as [`as_slice`](CSliceRefMut::as_slice),
/// [`as_slice_mut`](CSliceRefMut::as_slice_mut), [`add`](CSliceRefMut::add), and
/// [`clear`](CSliceRefMut::clear)) rely on the conversions defined by this trait to
/// preserve memory safety and prevent out-of-bounds access.
///
/// Implementations of this trait must guarantee:
/// 1. **Purity and Determinism**: `<Self as TryInto<usize>>::try_into` and
///    `<Self as TryFrom<usize>>::try_from` must be pure functions without side
///    effects, returning the exact same result for identical inputs every time.
/// 2. **Round-trip Equivalence**: For any `n: usize` that successfully converts to
///    `L = Self::try_from(n)`, `L.try_into()` must return `Ok(n)`.
/// 3. **Non-negative handling**: For signed types, negative values must fail conversion
///    via `TryInto<usize>` (returning `Err`), ensuring they are safely treated as length 0.
/// 4. **No Interior Mutability**: `Self` must not use interior mutability (`Cell`, `UnsafeCell`,
///    `Atomic*`, etc.) to change its conversion output over time.
pub unsafe trait CSliceLen:
    Copy + TryInto<usize> + TryFrom<usize> + Default + 'static
{
}

// SAFETY: Primitive unsigned integer types satisfy purity, determinism,
// and round-trip conversion to/from `usize` within their representable ranges.
unsafe impl CSliceLen for usize {}
unsafe impl CSliceLen for u8 {}
unsafe impl CSliceLen for u16 {}
unsafe impl CSliceLen for u32 {}
unsafe impl CSliceLen for u64 {}

// SAFETY: Primitive signed integer types satisfy purity, determinism,
// and correctly fail conversion via `TryInto<usize>` on negative values.
unsafe impl CSliceLen for isize {}
unsafe impl CSliceLen for i8 {}
unsafe impl CSliceLen for i16 {}
unsafe impl CSliceLen for i32 {}
unsafe impl CSliceLen for i64 {}

// ---------------------------------------------------------------------------
//  CSlicePtr — repr(transparent) wrapper around *mut T
// ---------------------------------------------------------------------------

/// A `#[repr(transparent)]` wrapper around `*mut T` for use in `#[repr(C)]`
/// structs.
///
/// `CSlicePtr` provides [`with_len`](Self::with_len) to get a `&[T]` slice,
/// [`with_len_mut`](Self::with_len_mut) to create a [`CSliceRefMut`] handle,
/// and [`clone_and_leak`](Self::clone_and_leak) to clone a Rust slice into a
/// C-allocated buffer.
///
/// Because it is `repr(transparent)`, it has the exact same layout as `*mut T`
/// and can be used directly in `#[repr(C)]` struct definitions without
/// affecting ABI compatibility.
///
/// # Safety Invariants
///
/// - The pointer is either null, or points to an owned array of `T`s of externally specified length
///   that is not accessed through any other pointer.
/// - The pointer is always aligned for `T`.
/// - If the pointer is non-null, the array has been allocated with `malloc`.
///
/// Additional invariants enforced by compile-time assertions in [`from_raw`]:
/// - `T` is not a zero-sized type, i.e. `size_of::<T>() > 0`.
/// - `T` is not over-aligned for a standard `malloc` call.
#[repr(transparent)]
pub struct CSlicePtr<T> {
    ptr: *mut T,
}

impl<T> CSlicePtr<T> {
    /// Create a null `CSlicePtr`.
    pub const fn null() -> Self {
        const {
            slice_type_assertions::<T>();
        }
        // SAFETY: null pointers trivially satisfy all other safety invariants of CSlicePtr.
        Self { ptr: ptr::null_mut() }
    }

    /// Construct a `CSlicePtr` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` satisfies all the safety invariants of
    /// [`CSlicePtr`].
    pub const unsafe fn from_raw(raw: *mut T) -> Self {
        const {
            slice_type_assertions::<T>();
        }
        Self { ptr: raw }
    }

    /// Return the inner raw pointer.
    pub const fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Return `true` if the inner pointer is null.
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    /// Create a shared (read-only) slice view with the given element length.
    ///
    /// Returns a plain `&[T]` whose lifetime is tied to `&self`, preventing
    /// mutation while the returned slice exists. If len is negative, clamp
    /// it to zero.
    ///
    /// # Safety
    ///
    /// `len` must be at most as long as the array pointed to by `self.ptr`.
    ///
    /// # Panics
    ///
    /// Panics if `len` exceeds the maximum safe slice length.
    pub unsafe fn with_len<L: CSliceLen>(&self, len: L) -> &[T] {
        let len: usize = len.try_into().unwrap_or(0);
        if self.ptr.is_null() || len == 0 {
            return &[];
        }
        assert!(len <= max_slice_len::<T>(), "CSlicePtr: len exceeds maximum safe slice length");
        // SAFETY: The caller guarantees the pointer/len invariant.
        // `&self` ties the lifetime of the returned slice to the borrow.
        unsafe { core::slice::from_raw_parts(self.ptr, len) }
    }

    /// Create a mutable slice handle with the given element length.
    ///
    /// This is the primary way to construct a [`CSliceRefMut`]. The lifetime
    /// `'_` is tied to the exclusive borrow of `&mut self`, preventing aliasing
    /// while the returned handle exists. If len is negative, clamp
    /// it to zero.
    ///
    /// # Safety
    ///
    /// `len` must be exactly as long as the array pointed to by `self.ptr`.
    ///
    /// # Panics
    ///
    /// Panics if `len` exceeds the maximum safe slice length.
    pub unsafe fn with_len_mut<'a, L: CSliceLen>(
        &'a mut self,
        len: &'a mut L,
    ) -> CSliceRefMut<'a, T, L> {
        let slice_len: usize = (*len).try_into().unwrap_or(0);
        assert!(
            slice_len <= max_slice_len::<T>(),
            "CSlicePtr: len exceeds maximum safe slice length"
        );
        // SAFETY: The caller guarantees the pointer/len invariant.
        // `&mut self` ties the lifetime of the returned `CSliceRefMut` to the
        // exclusive borrow, preventing aliasing.
        CSliceRefMut { ptr: self, len }
    }

    /// Clone the contents of a Rust slice into a new C-allocated buffer.
    ///
    /// This function allocates a new buffer using `malloc`, clones each element
    /// from `src` into it, and returns a [`CSlicePtr`] to the buffer.
    /// If successful, the caller assumes ownership of the returned pointer and is
    /// responsible for freeing it via `free` and dropping its elements.
    /// If not null, the returned pointer points to `src.len()` cloned elements.
    ///
    /// Returns `Ok(CSlicePtr::null())` for empty slices and `Err(AllocError)` on
    /// allocation failure.
    pub fn try_clone_and_leak(src: &[T]) -> Result<CSlicePtr<T>, AllocError>
    where
        T: Clone,
    {
        if src.is_empty() {
            return Ok(CSlicePtr::null());
        }
        let size = core::mem::size_of_val(src);
        // SAFETY: `size` is non-zero because the slice is non-empty, `T` is not a ZST,
        // and there was no overflow in the size calculation.
        let dst = unsafe { libc::malloc(size) } as *mut T;
        if dst.is_null() {
            return Err(AllocError);
        }
        // On most platforms we can assume alignment (see `slice_type_assertions`), but not all.
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows",
            target_os = "android",
            target_os = "ios"
        )))]
        if !dst.is_aligned() {
            return Err(AllocError);
        }
        // Clone each element directly into the C-allocated buffer.
        for (i, item) in src.iter().enumerate() {
            // SAFETY: `dst.add(i)` is within the allocated region and not yet
            // initialised, so `ptr::write` is the correct way to place a value.
            // Alignment is guaranteed by `CSlicePtr`'s safety invariant.
            unsafe { ptr::write(dst.add(i), item.clone()) };
        }
        // SAFETY: `dst` was just allocated via `malloc` and fully initialised.
        Ok(unsafe { CSlicePtr::from_raw(dst) })
    }

    /// Clone the contents of a Rust slice into a leaked [`CSlicePtr`] suitable
    /// for storage in a C struct. Returns null for empty slices.
    ///
    /// # Panics
    /// Panics if `malloc` returns null (out of memory).
    pub fn clone_and_leak(src: &[T]) -> CSlicePtr<T>
    where
        T: Clone,
    {
        Self::try_clone_and_leak(src).expect("CSlicePtr: malloc failed")
    }
}

// SAFETY: `CSlicePtr` is an owning pointer, so it is `Send` if `T` is `Send`.
unsafe impl<T: Send> Send for CSlicePtr<T> {}

impl<T> core::fmt::Debug for CSlicePtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("CSlicePtr").field(&self.ptr).finish()
    }
}

// ---------------------------------------------------------------------------
//  CSliceRefMut — mutable handle
// ---------------------------------------------------------------------------

/// A borrowed mutable handle over a `(*mut T, L)` pair in a C struct.
///
/// Created via [`CSlicePtr::with_len_mut`]. Provides mutable slice access and
/// mutation operations ([`add`](Self::add), [`clear`](Self::clear)).
///
/// Slice access is provided through [`Deref`](core::ops::Deref) and
/// [`DerefMut`](core::ops::DerefMut), which correctly tie the returned
/// slice's lifetime to the borrow of this handle.
///
/// # Safety Invariant
///
/// If `len > 0`, it is the length of slice `ptr`, and must be <= `isize::MAX`.
/// If `len <= 0`, the slice is empty.
pub struct CSliceRefMut<'a, T, L: CSliceLen> {
    ptr: &'a mut CSlicePtr<T>,
    len: &'a mut L,
}

impl<'a, T, L: CSliceLen> CSliceRefMut<'a, T, L> {
    /// Return the slice view with the lifetime tied to the borrow.
    pub fn as_slice(&self) -> &[T] {
        let len = (*self.len).try_into().unwrap_or(0);
        if self.ptr.is_null() || len == 0 {
            return &[];
        }
        // SAFETY:
        // - After checking for null, the invariants for `CSlicePtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CSlicePtr` owns the underlying array, so the pointer is valid for
        //   reads as long as we borrow it via `&self`.
        // - The invariant for `CSliceRefMut` guarantees that `len` is the valid length of the array
        //   pointed to by `ptr`.
        // - `L: CSliceLen` guarantees that `try_into()` is deterministic, pure, and truthfully
        //   converts the length without side-effects.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), len) }
    }

    /// Return the mutable slice view with the lifetime tied to the borrow.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let len = (*self.len).try_into().unwrap_or(0);
        if self.ptr.is_null() || len == 0 {
            return &mut [];
        }
        // SAFETY:
        // - After checking for null, the invariants for `CSlicePtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CSlicePtr` owns the underlying array, so the pointer is valid for
        //   reads and writes as long as we borrow it via `&mut self`.
        // - The invariant for `CSliceRefMut` guarantees that `len` is the valid length of the array
        //   pointed to by `ptr`.
        // - `L: CSliceLen` guarantees that `try_into()` is deterministic, pure, and truthfully
        //   converts the length without side-effects.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }

    /// Append an element, reallocating via `realloc` to grow by one slot.
    ///
    /// Returns `Err(value)` if the slice cannot grow (out of memory or `len`
    /// overflow), giving the caller the element back.
    pub fn try_add(&mut self, value: T) -> Result<(), T> {
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
        let new_size = new_len.wrapping_mul(core::mem::size_of::<T>());

        // SAFETY: `*self.ptr` is either null (realloc acts like malloc) or was
        // previously allocated by the C allocator with `old_len` elements.
        let new_ptr =
            unsafe { libc::realloc(self.ptr.as_ptr() as *mut libc::c_void, new_size) } as *mut T;
        if new_ptr.is_null() {
            core::hint::cold_path();
            return Err(value);
        }

        // SAFETY: `new_ptr` has room for `new_len` elements; the first
        // `old_len` are already initialised. We write one more at the end.
        unsafe { ptr::write(new_ptr.add(old_len), value) };

        // SAFETY: `new_ptr` was allocated via `realloc` and is valid.
        let p = unsafe { CSlicePtr::from_raw(new_ptr) };
        *self.ptr = p;
        *self.len = new_len_val;
        Ok(())
    }

    /// Append an element, reallocating via `realloc` to grow by one slot.
    ///
    /// # Panics
    /// Panics if the array cannot grow (out of memory or `len` overflow).
    pub fn add(&mut self, value: T) {
        if self.try_add(value).is_err() {
            core::hint::cold_path();
            panic!("CSliceRefMut: realloc failed");
        }
    }

    /// Replace the underlying pointer and len with new values, returning the old ones.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `new_ptr` points to a valid array of `new_len` elements
    /// (or is null with `new_len == 0`), satisfying the [safety invariants](CSlicePtr) of
    /// `CSlicePtr`.
    pub unsafe fn replace(&mut self, new_ptr: CSlicePtr<T>, new_len: L) -> (CSlicePtr<T>, L) {
        let old_ptr = core::mem::replace(self.ptr, new_ptr);
        let old_len = core::mem::replace(self.len, new_len);
        (old_ptr, old_len)
    }

    /// Swap the underlying pointer and len with another handle.
    ///
    /// This is safe because both handles already satisfy the `CSlicePtr`
    /// invariants, and swapping two valid `(ptr, len)` pairs preserves them.
    pub fn swap(&mut self, other: &mut CSliceRefMut<'_, T, L>) {
        core::mem::swap(self.ptr, other.ptr);
        core::mem::swap(self.len, other.len);
    }

    /// Drop all elements and reset to null / 0.
    pub fn clear(&mut self) {
        let n = (*self.len).try_into().unwrap_or(0);
        // We replace the pointer and length first to leave the handle in a valid, empty
        // state immediately. This is necessary for panic safety: if dropping elements
        // panics, the handle won't point to invalid memory.
        let old_ptr = core::mem::replace(self.ptr, CSlicePtr::null());
        *self.len = L::default();

        if !old_ptr.is_null() {
            // We use a local Drop guard to guarantee that `libc::free` is called
            // even if `ptr::drop_in_place` panics while dropping the elements.
            // This prevents leaking the underlying allocation.
            struct DropGuard(*mut libc::c_void);
            impl Drop for DropGuard {
                fn drop(&mut self) {
                    // SAFETY: `self.0` was allocated via the C allocator.
                    unsafe { libc::free(self.0) };
                }
            }
            let _guard = DropGuard(old_ptr.as_ptr() as *mut libc::c_void);

            if n > 0 {
                let slice = ptr::slice_from_raw_parts_mut(old_ptr.as_ptr(), n);
                // SAFETY: `old_ptr` points to `n` valid elements.
                // Run `Drop` impls on all elements (see `cslice_ref_mut_clear_drops_elements`).
                unsafe { ptr::drop_in_place(slice) };
            } else {
                core::hint::cold_path();
            }
        }
    }
}

impl<T, L: CSliceLen> core::ops::Deref for CSliceRefMut<'_, T, L> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, L: CSliceLen> core::ops::DerefMut for CSliceRefMut<'_, T, L> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU8, Ordering};
    use googletest::prelude::*;
    use std::ffi::c_int;

    // Helper to create a malloc-backed buffer with the given values.
    // Returns (ptr, len) suitable for CSlicePtr/CSliceRefMut.
    unsafe fn malloc_array<T, const N: usize>(values: [T; N]) -> (CSlicePtr<T>, c_int) {
        let size = core::mem::size_of_val(&values);
        let p = unsafe { libc::malloc(size) } as *mut T;
        for (i, v) in values.into_iter().enumerate() {
            // SAFETY: `p.add(i)` is within the allocated region.
            unsafe { ptr::write(p.add(i), v) };
        }
        // SAFETY: `p` is either null (and N=0) or points to N initialised elements.
        (unsafe { CSlicePtr::from_raw(p) }, N as c_int)
    }

    // Helper to free a malloc-backed buffer.
    unsafe fn free_array<T>(ptr: CSlicePtr<T>) {
        if !ptr.is_null() {
            unsafe { libc::free(ptr.as_ptr() as *mut libc::c_void) };
        }
    }

    // -----------------------------------------------------------------------
    //  CSlicePtr tests
    // -----------------------------------------------------------------------

    #[gtest]
    fn with_len_null_ptr() {
        let ptr = CSlicePtr::<i32>::null();
        let s = unsafe { ptr.with_len(0) };
        assert_that!(s.len(), eq(0));
        assert!(s.is_empty());
    }

    #[gtest]
    fn with_len_nonnull_ptr_zero_len() {
        // Simulate a C struct where a buffer was allocated but len is 0.
        let raw_ptr = unsafe { libc::malloc(16) } as *mut i32;
        let ptr = unsafe { CSlicePtr::from_raw(raw_ptr) };
        let s = unsafe { ptr.with_len(0) };
        assert_that!(s.len(), eq(0));
        assert!(s.is_empty());
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn with_len_negative_len() {
        let ptr = CSlicePtr::<i32>::null();
        let s = unsafe { ptr.with_len(-5) };
        assert_that!(s.len(), eq(0));
    }

    #[gtest]
    fn with_len_deref() {
        let (ptr, len) = unsafe { malloc_array([10, 20, 30]) };
        let s = unsafe { ptr.with_len(len) };
        assert_that!(s, container_eq([10, 20, 30]));
        assert_that!(s.len(), eq(3));
        assert!(!s.is_empty());
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn with_len_into_iterator() {
        let (ptr, len) = unsafe { malloc_array([1, 2, 3]) };
        let s = unsafe { ptr.with_len(len) };
        let collected: Vec<&i32> = s.iter().collect();
        assert_that!(collected, container_eq([&1, &2, &3]));
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn try_clone_and_leak_empty() {
        let result = CSlicePtr::<i32>::try_clone_and_leak(&[]);
        assert_that!(result, ok(anything()));
        assert!(result.unwrap().is_null());
    }

    #[gtest]
    fn try_clone_and_leak_nonempty() {
        let src = [100, 200, 300];
        let cloned = CSlicePtr::try_clone_and_leak(&src).unwrap();
        assert!(!cloned.is_null());
        // Verify cloned data is independent.
        let cloned_slice = unsafe { core::slice::from_raw_parts(cloned.as_ptr(), 3) };
        assert_that!(cloned_slice, container_eq([100, 200, 300]));
        unsafe {
            free_array(cloned);
        }
    }

    // -----------------------------------------------------------------------
    //  CSliceRefMut tests
    // -----------------------------------------------------------------------

    #[gtest]
    fn cslice_ref_mut_null_ptr() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = 0;
        let handle = unsafe { ptr.with_len_mut(&mut len) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
    }

    #[gtest]
    fn cslice_ref_mut_nonnull_ptr_zero_len() {
        // Simulate a C struct where a buffer was allocated but len is 0.
        // clear() must still free the buffer.
        let mut ptr = unsafe { CSlicePtr::from_raw(libc::malloc(16) as *mut i32) };
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_deref() {
        let (mut ptr, mut len) = unsafe { malloc_array([5, 6, 7]) };
        let handle = unsafe { ptr.with_len_mut(&mut len) };
        assert_that!(&*handle, container_eq([5, 6, 7]));
        assert_that!(handle.len(), eq(3));

        // Clean up.
        let mut handle = handle;
        handle.clear();
    }

    #[gtest]
    fn cslice_ref_mut_deref_mut() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        handle[0] = 99;
        assert_that!(&*handle, container_eq([99, 2, 3]));
        handle.clear();
    }

    #[gtest]
    fn cslice_ref_mut_add_to_empty() {
        let mut ptr = CSlicePtr::null();
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        handle.add(42);
        assert_that!(&*handle, container_eq([42]));
        handle.add(43);
        assert_that!(&*handle, container_eq([42, 43]));
        handle.clear();
        // Verify len was updated correctly (check after dropping the borrow).
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_add_to_existing() {
        let (mut ptr, mut len) = unsafe { malloc_array([10]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        handle.add(20);
        handle.add(30);
        assert_that!(&*handle, container_eq([10, 20, 30]));
        handle.clear();
        // Verify len was updated correctly (check after dropping the borrow).
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_try_add_overflow() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = c_int::MAX;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        let result = handle.try_add(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }

    #[gtest]
    fn cslice_ref_mut_clear_nonempty() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        handle.clear();
        assert_that!(handle.len(), eq(0));
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_replace() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let expected_old_ptr = ptr.as_ptr();
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };

        let (new_ptr, new_len) = unsafe { malloc_array([4, 5]) };
        let expected_new_ptr = new_ptr.as_ptr();

        // SAFETY: new_ptr/new_len come from malloc_array and are valid.
        let (old_ptr, old_len) = unsafe { handle.replace(new_ptr, new_len) };

        assert_that!(old_len, eq(3));
        assert_that!(old_ptr.as_ptr(), eq(expected_old_ptr));

        assert_that!(handle.len(), eq(2));
        assert_that!(handle.as_ptr(), eq(expected_new_ptr as *const _));

        unsafe { free_array(old_ptr) };
        handle.clear();
        assert_that!(handle.len(), eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_clear_already_empty() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        // Clearing an already-empty slice should not panic.
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_into_iterator() {
        let (mut ptr, mut len) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        {
            let collected: Vec<&mut i32> = handle.iter_mut().collect();
            assert_that!(collected.len(), eq(3));
            assert_that!(*collected[0], eq(1));
            assert_that!(*collected[1], eq(2));
            assert_that!(*collected[2], eq(3));
        }
        // Clean up
        handle.clear();
    }

    #[gtest]
    fn cslice_ref_mut_into_iterator_empty() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        let collected: Vec<&mut i32> = handle.iter_mut().collect();
        assert_that!(collected.len(), eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_clear_drops_elements() {
        static DROPPED: AtomicU8 = AtomicU8::new(0);
        struct Foo(u8);
        impl Drop for Foo {
            fn drop(&mut self) {
                DROPPED.fetch_add(self.0, Ordering::Relaxed);
            }
        }

        let (mut ptr, mut len) = unsafe { malloc_array([Foo(1), Foo(2)]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };

        assert_that!(DROPPED.load(Ordering::Relaxed), eq(0));
        handle.clear();
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(3));
        handle.clear(); // Should be a no-op.
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(3));
    }

    #[gtest]
    fn cslice_ref_mut_clear_panic_safety() {
        struct PanickingDrop(u8);
        impl Drop for PanickingDrop {
            fn drop(&mut self) {
                panic!("intentional drop panic");
            }
        }

        let (mut ptr, mut len) = unsafe { malloc_array([PanickingDrop(1)]) };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut handle = unsafe { ptr.with_len_mut(&mut len) };
            handle.clear();
        }));

        // Underlying pointer and len should already be reset to null/0 despite the panic.
        assert!(ptr.is_null());
        assert_that!(len, eq(0));
    }

    // -----------------------------------------------------------------------
    //  Generic Len tests (usize, u32, u8, i8, isize, u64)
    // -----------------------------------------------------------------------

    fn malloc_array_typed<T, L: CSliceLen, const N: usize>(
        values: [T; N],
    ) -> Result<(CSlicePtr<T>, L), AllocError> {
        let size = core::mem::size_of_val(&values);
        // SAFETY:
        let p = unsafe { libc::malloc(size) } as *mut T;
        if p.is_null() {
            return Err(AllocError);
        }
        for (i, v) in values.into_iter().enumerate() {
            // SAFETY: `p.add(i)` is within the allocated region.
            unsafe { ptr::write(p.add(i), v) };
        }
        // SAFETY: `p` is either null (and N=0) or points to N initialised elements.
        let len = L::try_from(N).ok().expect("valid len");
        Ok((unsafe { CSlicePtr::from_raw(p) }, len))
    }

    #[gtest]
    fn cslice_with_len_usize() {
        let result = malloc_array_typed::<i32, usize, 3>([10, 20, 30]);
        assert!(result.is_ok());
        let (ptr, len) = result.unwrap();
        let s = unsafe { ptr.with_len(len) };
        assert_that!(s, container_eq([10, 20, 30]));
        assert_that!(s.len(), eq(3));
        unsafe { free_array(ptr) };

        let null_ptr = CSlicePtr::<i32>::null();
        let empty = unsafe { null_ptr.with_len(0usize) };
        assert!(empty.is_empty());
    }

    #[gtest]
    fn cslice_ref_mut_usize() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: usize = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        handle.add(100);
        handle.add(200);
        assert_that!(&*handle, container_eq([100, 200]));
        assert_that!(handle.len(), eq(2));
        handle.clear();
        drop(handle);
        assert_that!(len, eq(0usize));
        assert!(ptr.is_null());
    }

    #[gtest]
    fn cslice_ref_mut_u32() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: u32 = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        handle.add(42);
        assert_that!(&*handle, container_eq([42]));
        assert_that!(handle.len(), eq(1));
        handle.clear();
        drop(handle);
        assert_that!(len, eq(0u32));
    }

    #[gtest]
    fn cslice_ref_mut_u64() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: u64 = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        handle.add(77);
        assert_that!(&*handle, container_eq([77]));
        assert_that!(handle.len(), eq(1));
        handle.clear();
        drop(handle);
        assert_that!(len, eq(0u64));
    }

    #[gtest]
    fn cslice_ref_mut_u8_overflow() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut len: u8 = u8::MAX;
        let mut handle = unsafe { ptr.with_len_mut(&mut len) };
        let result = handle.try_add(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }

    #[gtest]
    fn cslice_with_len_signed_negative() {
        let ptr = CSlicePtr::<i32>::null();
        let s_i8 = unsafe { ptr.with_len(-1i8) };
        assert!(s_i8.is_empty());

        let s_i16 = unsafe { ptr.with_len(-10i16) };
        assert!(s_i16.is_empty());

        let s_isize = unsafe { ptr.with_len(-100isize) };
        assert!(s_isize.is_empty());

        let s_i64 = unsafe { ptr.with_len(-1000i64) };
        assert!(s_i64.is_empty());
    }
}

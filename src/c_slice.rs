//! Safe handles for raw-pointer-backed arrays in C structs.
//!
//! These types centralise `unsafe` access to `(*mut T, c_int)` field pairs.
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
//!     item_count: c_int,
//! }
//!
//! impl MyStruct {
//!     // Shared accessor — returns &[T] (from &self).
//!     fn items(&self) -> &[f32] {
//!         // SAFETY: the length of `items` is `item_count`.
//!         unsafe { self.items.with_len(self.item_count) }
//!     }
//!
//!     // Mutable accessor — returns CSliceRefMut (from &mut self).
//!     fn items_mut(&mut self) -> CSliceRefMut<'_, f32> {
//!         // SAFETY: the length of `items` is `item_count`.
//!         unsafe { self.items.with_len_mut(&mut self.item_count) }
//!     }
//! }
//!
//! let mut my_struct = MyStruct {
//!     items: CSlicePtr::null(),
//!     item_count: 0,
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
use core::ffi::c_int;
use core::ptr;

/// The maximum element count for type `T` that stays within the
/// [`isize::MAX`]-byte limit required by [`core::slice::from_raw_parts`].
///
/// On 64-bit platforms this vastly exceeds `c_int::MAX`, so any runtime
/// comparison against it is optimized away by the compiler.
const fn max_slice_count<T>() -> usize {
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
    /// Create a `CSlicePtr` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer satisfies the [Safety Invariants](#safety-invariant)
    /// of `CSlicePtr`.
    pub const unsafe fn from_raw(ptr: *mut T) -> Self {
        const {
            slice_type_assertions::<T>();
        }
        Self { ptr }
    }

    /// Create a null `CSlicePtr`.
    pub const fn null() -> Self {
        const {
            slice_type_assertions::<T>();
        }
        // SAFETY: null pointers trivially satisfy all safety invariants of `CSlicePtr`.
        Self { ptr: ptr::null_mut() }
    }

    /// Return the inner raw pointer.
    pub const fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Return `true` if the inner pointer is null.
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    /// Create a shared (read-only) slice view with the given element count.
    ///
    /// Returns a plain `&[T]` whose lifetime is tied to `&self`, preventing
    /// mutation while the returned slice exists.
    ///
    /// # Safety
    ///
    /// `count` must be at most as long as the array pointed to by `self.ptr`.
    ///
    /// # Panics
    ///
    /// Panics if `count` exceeds the maximum safe slice length.
    pub unsafe fn with_len(&self, count: c_int) -> &[T] {
        if self.ptr.is_null() || count <= 0 {
            return &[];
        }
        assert!(
            (count as usize) <= max_slice_count::<T>(),
            "CSlicePtr: count exceeds maximum safe slice length"
        );
        // SAFETY: The caller guarantees the pointer/count invariant.
        // `&self` ties the lifetime of the returned slice to the borrow.
        unsafe { core::slice::from_raw_parts(self.ptr, count as usize) }
    }

    /// Create a mutable slice handle with the given element count.
    ///
    /// This is the primary way to construct a [`CSliceRefMut`]. The lifetime
    /// `'_` is tied to the exclusive borrow of `&mut self`, preventing aliasing
    /// while the returned handle exists.
    ///
    /// # Safety
    ///
    /// `count` must be exactly as long as the array pointed to by `self.ptr`.
    ///
    /// # Panics
    ///
    /// Panics if `count` exceeds the maximum safe slice length.
    pub unsafe fn with_len_mut<'a>(&'a mut self, count: &'a mut c_int) -> CSliceRefMut<'a, T> {
        assert!(
            *count <= 0 || (*count as usize) <= max_slice_count::<T>(),
            "CSlicePtr: count exceeds maximum safe slice length"
        );
        // SAFETY: The caller guarantees the pointer/count invariant.
        // `&mut self` ties the lifetime of the returned `CSliceRefMut` to the
        // exclusive borrow, preventing aliasing.
        CSliceRefMut { ptr: self, count }
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

/// A borrowed mutable handle over a `(*mut T, c_int)` pair in a C struct.
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
/// If `count > 0`, it is the length of slice `ptr`, and must be <= `isize::MAX`.
/// If `count <= 0`, the slice is empty.
pub struct CSliceRefMut<'a, T> {
    ptr: &'a mut CSlicePtr<T>,
    count: &'a mut c_int,
}

impl<'a, T> CSliceRefMut<'a, T> {
    /// Return the slice with the lifetime of the underlying data.
    pub fn as_slice(&self) -> &'a [T] {
        if self.ptr.is_null() || *self.count <= 0 {
            return &[];
        }
        // SAFETY:
        // - After checking for null, the invariants for `CSlicePtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CSlicePtr` owns the underlying array, so the pointer is valid for
        //   reads as long as we borrow it via `&self`.
        // - The invariant for `CSliceRefMut` guarantees that `count` is the valid length of the array
        //   pointed to by `ptr`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), *self.count as usize) }
    }

    /// Return the mutable slice with the lifetime of the underlying data.
    pub fn as_slice_mut(&mut self) -> &'a mut [T] {
        if self.ptr.is_null() || *self.count <= 0 {
            return &mut [];
        }
        // SAFETY:
        // - After checking for null, the invariants for `CSlicePtr` guarantee that `ptr` points to
        //   an owned array of `T`s, and that the pointer is aligned for `T`.
        // - `CSlicePtr` owns the underlying array, so the pointer is valid for
        //   reads and writes as long as we borrow it via `&mut self`.
        // - The invariant for `CSliceRefMut` guarantees that `count` is the valid length of the array
        //   pointed to by `ptr`.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), *self.count as usize) }
    }

    /// Append an element, reallocating via `realloc` to grow by one slot.
    ///
    /// Returns `Err(value)` if the slice cannot grow (out of memory or `c_int`
    /// overflow), giving the caller the element back.
    pub fn try_add(&mut self, value: T) -> Result<(), T> {
        let old_len = (*self.count).max(0);
        let Some(new_len) =
            old_len.checked_add(1).map(|n| n as usize).filter(|&n| n <= max_slice_count::<T>())
        else {
            core::hint::cold_path();
            return Err(value);
        };
        let old_len = old_len as usize;

        // Cannot overflow: `new_len <= max_slice_count::<T>()` guarantees
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
        *self.count = new_len as c_int;
        Ok(())
    }

    /// Append an element, reallocating via `realloc` to grow by one slot.
    ///
    /// # Panics
    /// Panics if the array cannot grow (out of memory or `c_int` overflow).
    pub fn add(&mut self, value: T) {
        if self.try_add(value).is_err() {
            core::hint::cold_path();
            panic!("CSliceRefMut: realloc failed");
        }
    }

    /// Replace the underlying pointer and count with new values, returning the old ones.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `new_ptr` points to a valid array of `new_count` elements
    /// (or is null with `new_count == 0`), satisfying the [safety invariants](CSlicePtr) of
    /// `CSlicePtr`.
    pub unsafe fn replace(
        &mut self,
        new_ptr: CSlicePtr<T>,
        new_count: c_int,
    ) -> (CSlicePtr<T>, c_int) {
        let old_ptr = core::mem::replace(self.ptr, new_ptr);
        let old_count = core::mem::replace(self.count, new_count);
        (old_ptr, old_count)
    }

    /// Swap the underlying pointer and count with another handle.
    ///
    /// This is safe because both handles already satisfy the `CSlicePtr`
    /// invariants, and swapping two valid `(ptr, count)` pairs preserves them.
    pub fn swap(&mut self, other: &mut CSliceRefMut<'_, T>) {
        core::mem::swap(self.ptr, other.ptr);
        core::mem::swap(self.count, other.count);
    }

    /// Drop all elements and reset to null / 0.
    pub fn clear(&mut self) {
        let n = (*self.count).max(0) as usize;
        if !self.ptr.is_null() {
            if n > 0 {
                let slice = ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), n);
                // SAFETY: `*self.ptr` points to `n` valid elements.
                // Run `Drop` impls on all elements (see `cslice_ref_mut_clear_drops_elements`)
                unsafe { ptr::drop_in_place(slice) };
            } else {
                core::hint::cold_path();
            }
            // SAFETY: `*self.ptr` was allocated via the C allocator.
            // We have exclusive access and will clear `ptr` afterwards, so the data will never be
            // accessed again.
            unsafe { libc::free(self.ptr.as_ptr() as *mut libc::c_void) };
        }
        *self.ptr = CSlicePtr::null();
        *self.count = 0;
    }
}

impl<T> core::ops::Deref for CSliceRefMut<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> core::ops::DerefMut for CSliceRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU8, Ordering};
    use googletest::prelude::*;

    // Helper to create a malloc-backed buffer with the given values.
    // Returns (ptr, count) suitable for CSlicePtr/CSliceRefMut.
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
    fn with_len_nonnull_ptr_zero_count() {
        // Simulate a C struct where a buffer was allocated but count is 0.
        let raw_ptr = unsafe { libc::malloc(16) } as *mut i32;
        let ptr = unsafe { CSlicePtr::from_raw(raw_ptr) };
        let s = unsafe { ptr.with_len(0) };
        assert_that!(s.len(), eq(0));
        assert!(s.is_empty());
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn with_len_negative_count() {
        let ptr = CSlicePtr::<i32>::null();
        let s = unsafe { ptr.with_len(-5) };
        assert_that!(s.len(), eq(0));
    }

    #[gtest]
    fn with_len_deref() {
        let (ptr, count) = unsafe { malloc_array([10, 20, 30]) };
        let s = unsafe { ptr.with_len(count) };
        assert_that!(s, container_eq([10, 20, 30]));
        assert_that!(s.len(), eq(3));
        assert!(!s.is_empty());
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn with_len_into_iterator() {
        let (ptr, count) = unsafe { malloc_array([1, 2, 3]) };
        let s = unsafe { ptr.with_len(count) };
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
        let mut count: c_int = 0;
        let handle = unsafe { ptr.with_len_mut(&mut count) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
    }

    #[gtest]
    fn cslice_ref_mut_nonnull_ptr_zero_count() {
        // Simulate a C struct where a buffer was allocated but count is 0.
        // clear() must still free the buffer.
        let mut ptr = unsafe { CSlicePtr::from_raw(libc::malloc(16) as *mut i32) };
        let mut count: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
        assert_that!(handle.len(), eq(0));
        assert!(handle.is_empty());
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(count, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_deref() {
        let (mut ptr, mut count) = unsafe { malloc_array([5, 6, 7]) };
        let handle = unsafe { ptr.with_len_mut(&mut count) };
        assert_that!(&*handle, container_eq([5, 6, 7]));
        assert_that!(handle.len(), eq(3));

        // Clean up.
        let mut handle = handle;
        handle.clear();
    }

    #[gtest]
    fn cslice_ref_mut_deref_mut() {
        let (mut ptr, mut count) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
        handle[0] = 99;
        assert_that!(&*handle, container_eq([99, 2, 3]));
        handle.clear();
    }

    #[gtest]
    fn cslice_ref_mut_add_to_empty() {
        let mut ptr = CSlicePtr::null();
        let mut count: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
        handle.add(42);
        assert_that!(&*handle, container_eq([42]));
        handle.add(43);
        assert_that!(&*handle, container_eq([42, 43]));
        handle.clear();
        // Verify count was updated correctly (check after dropping the borrow).
        assert_that!(count, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_add_to_existing() {
        let (mut ptr, mut count) = unsafe { malloc_array([10]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
        handle.add(20);
        handle.add(30);
        assert_that!(&*handle, container_eq([10, 20, 30]));
        handle.clear();
        // Verify count was updated correctly (check after dropping the borrow).
        assert_that!(count, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_try_add_overflow() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut count: c_int = c_int::MAX;
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
        let result = handle.try_add(999);
        assert!(result.is_err());
        assert_that!(result.unwrap_err(), eq(999));
    }

    #[gtest]
    fn cslice_ref_mut_clear_nonempty() {
        let (mut ptr, mut count) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
        handle.clear();
        assert_that!(handle.len(), eq(0));
        assert!(ptr.is_null());
        assert_that!(count, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_replace() {
        let (mut ptr, mut count) = unsafe { malloc_array([1, 2, 3]) };
        let expected_old_ptr = ptr.as_ptr();
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };

        let (new_ptr, new_count) = unsafe { malloc_array([4, 5]) };
        let expected_new_ptr = new_ptr.as_ptr();

        // SAFETY: new_ptr/new_count come from malloc_array and are valid.
        let (old_ptr, old_count) = unsafe { handle.replace(new_ptr, new_count) };

        assert_that!(old_count, eq(3));
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
        let mut count: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
        // Clearing an already-empty slice should not panic.
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(count, eq(0));
    }

    #[gtest]
    fn cslice_ref_mut_into_iterator() {
        let (mut ptr, mut count) = unsafe { malloc_array([1, 2, 3]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
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
        let mut count: c_int = 0;
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };
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

        let (mut ptr, mut count) = unsafe { malloc_array([Foo(1), Foo(2)]) };
        let mut handle = unsafe { ptr.with_len_mut(&mut count) };

        assert_that!(DROPPED.load(Ordering::Relaxed), eq(0));
        handle.clear();
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(3));
        handle.clear(); // Should be a no-op.
        assert_that!(DROPPED.load(Ordering::Relaxed), eq(3));
    }
}

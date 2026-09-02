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
//! They support generic memory allocators via [`core::alloc::Allocator`], defaulting to
//! the [`LibcAlloc`] allocator (`malloc`/`realloc`/`free`).
//!
//! # Usage
//!
//! Given a `#[repr(C)]` struct with raw pointer fields:
//!
//! ```
//! use std::os::raw::c_int;
//! use safer_cffi::{CSlicePtr, CVecRefMut};
//!
//! #[repr(C)]
//! struct MyStruct {
//!     items: CSlicePtr<f32>,    // repr(transparent) wrapper around *mut f32
//!     item_len: c_int,
//! }
//!
//! impl MyStruct {
//!     // Shared slice accessor — returns &[T] (from &self).
//!     fn items(&self) -> &[f32] {
//!         // SAFETY: the length of `items` is `item_len`.
//!         unsafe { self.items.with_len(self.item_len) }
//!     }
//!
//!     // Mutable slice accessor — returns &mut [T] (from &mut self).
//!     fn items_mut(&mut self) -> &mut [f32] {
//!         // SAFETY: the length of `items` is `item_len`.
//!         unsafe { self.items.with_len_mut(self.item_len) }
//!     }
//!
//!     // Mutable vector accessor — returns CVecRefMut (from &mut self and &mut item_len).
//!     fn items_vec_mut(&mut self) -> CVecRefMut<'_, f32, c_int> {
//!         // SAFETY: the length of `items` is `item_len`.
//!         unsafe { self.items.with_len_vec_mut(&mut self.item_len) }
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
//! // Mutate slice in place:
//! my_struct.items_mut()[0] = 2.0;
//!
//! // Resizing / dynamic vector mutation:
//! my_struct.items_vec_mut().push_back(40.0);
//!
//! // Clone impl:
//! let cloned_ptr: CSlicePtr<f32> = CSlicePtr::clone_and_leak(my_struct.items());
//!
//! // Drop impl:
//! my_struct.items_vec_mut().clear();
//! ```

use crate::alloc::LibcAlloc;
use crate::c_vec::CVecRefMut;
use crate::errors::AllocError;
use allocator_api2::alloc::{Allocator, Layout};
use core::marker::PhantomData;
use core::ptr::{self, NonNull};

/// The maximum slice length for type `T` that stays within the
/// [`isize::MAX`]-byte limit required by [`core::slice::from_raw_parts`].
///
/// On 64-bit platforms this vastly exceeds `c_int::MAX`, so any runtime
/// comparison against it is optimized away by the compiler.
pub(crate) const fn max_slice_len<T>() -> usize {
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
/// # Safety invariant:
///
/// Safe methods on [`CVecRefMut`] (such as
/// [`as_slice`](CVecRefMut::as_slice), [`as_slice_mut`](CVecRefMut::as_slice_mut),
/// [`push_back`](CVecRefMut::push_back), and [`clear`](CVecRefMut::clear)) rely on the conversions defined
/// by this trait to preserve memory safety and prevent out-of-bounds access.
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
/// [`with_len_mut`](Self::with_len_mut) to get a `&mut [T]` slice,
/// [`with_len_vec_mut`](Self::with_len_vec_mut) to create a [`CVecRefMut`] handle,
/// and [`clone_and_leak`](Self::clone_and_leak) to clone a Rust slice into a
/// C-allocated buffer.
///
/// The allocator `A` defaults to [`LibcAlloc`]. It has the exact same layout and ABI
/// as `*mut T` and can be used directly in `#[repr(C)]` struct definitions without
/// affecting ABI compatibility.
///
/// # Safety Invariants
///
/// - The pointer is either null, or points to an owned array of `T`s of externally specified length
///   that is not accessed through any other pointer. The array of `T`s ought to also be initialized.
/// - The pointer is always aligned for `T`.
/// - If the pointer is non-null, it has been allocated with the allocator `A` with layout
///   matching `Layout::array::<T>(len)`.
///
/// Additional invariants enforced by compile-time assertions in [`from_raw`]:
/// - `T` is not a zero-sized type, i.e. `size_of::<T>() > 0`.
/// - `T` is not over-aligned for a standard `malloc` call.
#[repr(transparent)]
pub struct CSlicePtr<T, A = LibcAlloc> {
    ptr: *mut T,
    _allocator: PhantomData<A>,
}

impl<T, A: Allocator> CSlicePtr<T, A> {
    /// Create a null `CSlicePtr`.
    pub const fn null() -> Self {
        const {
            slice_type_assertions::<T>();
        }
        // SAFETY: null pointers trivially satisfy all other safety invariants of CSlicePtr.
        Self { ptr: ptr::null_mut(), _allocator: PhantomData }
    }

    /// Construct a `CSlicePtr` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` satisfies all the safety invariants of
    /// [`CSlicePtr`], including that if non-null, it points to memory allocated
    /// by the allocator `A`.
    pub const unsafe fn from_raw(raw: *mut T) -> Self {
        const {
            slice_type_assertions::<T>();
        }
        Self { ptr: raw, _allocator: PhantomData }
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
        // SAFETY: The caller guarantees that `self.ptr` points to at least `len` initialised
        // elements of type `T`. `&self` ties the lifetime of the returned slice to the borrow.
        unsafe { core::slice::from_raw_parts(self.ptr, len) }
    }

    /// Create a mutable slice view over the array with the given element length.
    ///
    /// The lifetime of the result is tied to the exclusive borrow `&'a mut self`, preventing
    /// aliasing.
    /// If `len` is negative, it is clamped to zero.
    ///
    /// # Safety
    ///
    /// - If `self.ptr` is non-null and `len > 0`, `self.ptr` points to at least `len` initialized,
    ///   properly aligned elements of type `T`.
    ///
    /// # Panics
    ///
    /// Panics if `len` exceeds the maximum safe slice length.
    pub unsafe fn with_len_mut<L: CSliceLen>(&mut self, len: L) -> &mut [T] {
        let slice_len: usize = len.try_into().unwrap_or(0);
        assert!(
            slice_len <= max_slice_len::<T>(),
            "CSlicePtr: len exceeds maximum safe slice length"
        );
        if self.ptr.is_null() || slice_len == 0 {
            return &mut [];
        }
        // SAFETY: The caller guarantees `self.ptr` is valid for reads and writes for `slice_len`
        // elements of type `T`, properly aligned, and unaliased for `'a`.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, slice_len) }
    }
}

impl<T, A: Allocator + PartialEq> CSlicePtr<T, A> {
    /// Create a mutable vector handle with the given element length and custom [`Allocator`].
    ///
    /// This is the primary way to construct a [`CVecRefMut`]. The lifetime
    /// `'_` is tied to the exclusive borrow of `&mut self`, preventing aliasing
    /// while the returned handle exists.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `len` reflects the exact number of initialized elements pointed to by `self.ptr` (or `<= 0` if empty).
    /// - The instance of `A` passed to this function MUST BE the same instance that was used for
    ///   allocation of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `len` exceeds the maximum safe slice length.
    pub unsafe fn with_len_vec_mut_in<'a, L: CSliceLen>(
        &'a mut self,
        len: &'a mut L,
        alloc: A,
    ) -> CVecRefMut<'a, T, L, A> {
        let slice_len: usize = (*len).try_into().unwrap_or(0);
        assert!(
            slice_len <= max_slice_len::<T>(),
            "CSlicePtr: len exceeds maximum safe slice length"
        );
        // SAFETY: The caller guarantees the pointer/len invariant, allocator compatibility,
        // validity to deallocate/reallocate, and absence of aliases.
        // `&mut self` ties the lifetime of the returned `CVecRefMut` to the exclusive borrow,
        // preventing aliasing through `self`.
        CVecRefMut { ptr: self, len, alloc }
    }

    /// Clone the contents of a Rust slice into a new C-allocated buffer using a custom [`Allocator`].
    ///
    /// This function allocates a new buffer using `alloc`, clones each element
    /// from `src` into it, and returns a [`CSlicePtr`] to the buffer.
    /// If successful, the caller assumes ownership of the returned pointer and is
    /// responsible for freeing it via `alloc` and dropping its elements.
    /// If not null, the returned pointer points to `src.len()` cloned elements.
    ///
    /// Returns `Ok(CSlicePtr::null())` for empty slices and `Err(AllocError)` on
    /// allocation failure.
    pub fn try_clone_and_leak_in(src: &[T], alloc: A) -> Result<CSlicePtr<T, A>, AllocError>
    where
        T: Clone,
    {
        if src.is_empty() {
            return Ok(CSlicePtr::null());
        }
        let layout = Layout::for_value(src);
        let slice = alloc.allocate(layout)?;
        let dst = slice.as_ptr() as *mut T;

        /// Drop guard that cleans up allocated memory and drops initialized elements
        /// if element cloning panics.
        ///
        /// # Safety invariant:
        /// - `ptr` was allocated via `alloc::allocate` and is properly aligned for `T`.
        /// - The first `initialized` elements at `ptr` are valid instances of `T`.
        struct CloneDropGuard<'g, T, A: Allocator> {
            alloc: &'g A,
            ptr: NonNull<u8>,
            layout: Layout,
            initialized: usize,
            _marker: PhantomData<T>,
        }

        impl<'g, T, A: Allocator> Drop for CloneDropGuard<'g, T, A> {
            fn drop(&mut self) {
                if self.initialized > 0 {
                    let slice = ptr::slice_from_raw_parts_mut(
                        self.ptr.as_ptr() as *mut T,
                        self.initialized,
                    );
                    // SAFETY: By the safety invariants of `CloneDropGuard`, `self.ptr` is aligned for
                    // `T` and the first `self.initialized` elements are valid, fully initialized
                    // instances of `T` that can be safely dropped in place.
                    unsafe { ptr::drop_in_place(slice) };
                }
                // SAFETY: By the safety invariants of `CloneDropGuard`, `self.ptr` was allocated via
                // `self.alloc` with `self.layout`.
                unsafe { self.alloc.deallocate(self.ptr, self.layout) };
            }
        }

        let non_null = NonNull::new(dst as *mut u8).unwrap();
        // Safety note: `non_null` was allocated with `alloc` using `layout` (aligned for `T`),
        // and 0 elements are initialized, trivially satisfying the safety invariants.
        let mut guard = CloneDropGuard {
            alloc: &alloc,
            ptr: non_null,
            layout,
            initialized: 0,
            _marker: PhantomData::<T>,
        };

        // Clone each element directly into the allocated buffer.
        for (i, item) in src.iter().enumerate() {
            // SAFETY: `dst.add(i)` is within the allocated region and not yet
            // initialised, so `ptr::write` is the correct way to place a value.
            // Alignment is guaranteed by `CSlicePtr`'s safety invariant.
            unsafe { ptr::write(dst.add(i), item.clone()) };
            // There is now one more initialized element, so increment.
            guard.initialized += 1;
        }

        // Success: disarm the guard so the buffer is leaked to the caller as intended.
        core::mem::forget(guard);

        // SAFETY: `dst` was just allocated via `alloc` and all `src.len()` elements were fully
        // initialised.
        Ok(unsafe { CSlicePtr::from_raw(dst) })
    }

    /// Clone the contents of a Rust slice into a leaked [`CSlicePtr`] using a custom [`Allocator`].
    /// Returns null for empty slices.
    ///
    /// # Panics
    /// Panics if allocation fails.
    pub fn clone_and_leak_in(src: &[T], alloc: A) -> CSlicePtr<T, A>
    where
        T: Clone,
    {
        Self::try_clone_and_leak_in(src, alloc).expect("CSlicePtr: allocation failed")
    }
}

impl<T> CSlicePtr<T, LibcAlloc> {
    /// Create a mutable vector handle with the given element length and custom [`Allocator`].
    ///
    /// This is the primary way to construct a [`CVecRefMut`]. The lifetime
    /// `'_` is tied to the exclusive borrow of `&mut self`, preventing aliasing
    /// while the returned handle exists.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `len` reflects the exact number of initialized elements pointed to by `self.ptr` (or `<= 0` if empty).
    ///
    /// # Panics
    ///
    /// Panics if `len` exceeds the maximum safe slice length.
    pub unsafe fn with_len_vec_mut<'a, L: CSliceLen>(
        &'a mut self,
        len: &'a mut L,
    ) -> CVecRefMut<'a, T, L, LibcAlloc> {
        // SAFETY: The caller guarantees `len` is the exact length and no active aliases exist.
        // Compatibility with `LibcAlloc` is an invariant of `CSlicePtr<T, LibcAlloc>` and the
        // fact that all instances of LibcAlloc are equivalent.
        unsafe { self.with_len_vec_mut_in(len, LibcAlloc) }
    }

    /// Clone the contents of a Rust slice into a new C-allocated buffer using [`LibcAlloc`].
    ///
    /// Returns `Ok(CSlicePtr::null())` for empty slices and `Err(AllocError)` on
    /// allocation failure.
    pub fn try_clone_and_leak(src: &[T]) -> Result<CSlicePtr<T, LibcAlloc>, AllocError>
    where
        T: Clone,
    {
        Self::try_clone_and_leak_in(src, LibcAlloc)
    }

    /// Clone the contents of a Rust slice into a leaked [`CSlicePtr`] suitable
    /// for storage in a C struct. Returns null for empty slices.
    ///
    /// # Panics
    /// Panics if `malloc` returns null (out of memory).
    pub fn clone_and_leak(src: &[T]) -> CSlicePtr<T, LibcAlloc>
    where
        T: Clone,
    {
        Self::try_clone_and_leak(src).expect("CSlicePtr: allocation failed")
    }
}

// SAFETY: `CSlicePtr` is an owning pointer, so it is `Send` if `T` is `Send`.
unsafe impl<T: Send, A> Send for CSlicePtr<T, A> {}

impl<T, A> core::fmt::Debug for CSlicePtr<T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("CSlicePtr").field(&self.ptr).finish()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]
    use super::*;
    use crate::testing::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use googletest::prelude::*;
    use std::ffi::c_int;

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

    #[gtest]
    fn try_clone_and_leak_in_custom_allocator() {
        let alloc = TrackingAlloc::default();
        let src = [10, 20, 30];
        let cloned = CSlicePtr::try_clone_and_leak_in(&src, &alloc).unwrap();
        assert!(!cloned.is_null());
        assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));

        // SAFETY: `cloned` has 3 elements.
        let cloned_slice = unsafe { core::slice::from_raw_parts(cloned.as_ptr(), 3) };
        assert_that!(cloned_slice, container_eq([10, 20, 30]));

        let mut count: c_int = 3;
        let mut ptr = cloned;
        // SAFETY: `ptr` was allocated via `alloc` with `count` elements.
        let mut handle = unsafe { ptr.with_len_vec_mut_in(&mut count, &alloc) };
        handle.clear();
        assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(1));
    }

    #[gtest]
    fn try_clone_and_leak_panic_safety() {
        static DROPPED: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct PanickingClone(usize);
        impl Clone for PanickingClone {
            fn clone(&self) -> Self {
                if self.0 == 2 {
                    panic!("intentional clone panic");
                }
                Self(self.0)
            }
        }
        impl Drop for PanickingClone {
            fn drop(&mut self) {
                DROPPED.fetch_add(1, Ordering::SeqCst);
            }
        }

        let alloc = TrackingAlloc::default();
        let src = [PanickingClone(0), PanickingClone(1), PanickingClone(2)];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = CSlicePtr::try_clone_and_leak_in(&src, &alloc);
        }));
        assert!(result.is_err());
        // Cloned elements 0 and 1 must be dropped, and the buffer must be deallocated.
        assert_that!(alloc.alloc_count.load(Ordering::SeqCst), eq(1));
        assert_that!(alloc.dealloc_count.load(Ordering::SeqCst), eq(1));
        assert_that!(DROPPED.load(Ordering::SeqCst), eq(2));
    }

    // -----------------------------------------------------------------------
    //  with_len_mut (&mut [T]) tests
    // -----------------------------------------------------------------------

    #[gtest]
    fn cslice_with_len_mut_null_ptr() {
        let mut ptr = CSlicePtr::<i32>::null();
        let slice = unsafe { ptr.with_len_mut(0) };
        assert_that!(slice.len(), eq(0));
        assert!(slice.is_empty());
    }

    #[gtest]
    fn cslice_with_len_mut_nonnull_ptr_zero_len() {
        let mut ptr = unsafe { CSlicePtr::from_raw(libc::malloc(16) as *mut i32) };
        let slice = unsafe { ptr.with_len_mut(0) };
        assert_that!(slice.len(), eq(0));
        assert!(slice.is_empty());
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn cslice_with_len_mut_read() {
        let (mut ptr, len) = unsafe { malloc_array([5, 6, 7]) };
        let slice = unsafe { ptr.with_len_mut(len) };
        assert_that!(&*slice, container_eq([5, 6, 7]));
        assert_that!(slice.len(), eq(3));
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn cslice_with_len_mut_modify() {
        let (mut ptr, len) = unsafe { malloc_array([1, 2, 3]) };
        let slice = unsafe { ptr.with_len_mut(len) };
        slice[0] = 99;
        assert_that!(&*slice, container_eq([99, 2, 3]));
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn cslice_with_len_mut_into_iterator() {
        let (mut ptr, len) = unsafe { malloc_array([1, 2, 3]) };
        let slice = unsafe { ptr.with_len_mut(len) };
        let collected: Vec<&mut i32> = slice.iter_mut().collect();
        assert_that!(collected.len(), eq(3));
        assert_that!(*collected[0], eq(1));
        assert_that!(*collected[1], eq(2));
        assert_that!(*collected[2], eq(3));
        unsafe { free_array(ptr) };
    }

    #[gtest]
    fn cslice_with_len_mut_into_iterator_empty() {
        let mut ptr = CSlicePtr::<i32>::null();
        let slice = unsafe { ptr.with_len_mut(0) };
        let collected: Vec<&mut i32> = slice.iter_mut().collect();
        assert_that!(collected.len(), eq(0));
    }

    // -----------------------------------------------------------------------
    //  Generic Len tests (usize, u32, u8, i8, isize, u64)
    // -----------------------------------------------------------------------

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

    #[gtest]
    fn c_slice_ptr_layout() {
        // CSlicePtr must have the exact same size and alignment as a raw pointer regardless of A.
        assert_eq!(core::mem::size_of::<CSlicePtr<i32>>(), core::mem::size_of::<*mut i32>());
        assert_eq!(core::mem::align_of::<CSlicePtr<i32>>(), core::mem::align_of::<*mut i32>());
        assert_eq!(
            core::mem::size_of::<CSlicePtr<i32, TrackingAlloc>>(),
            core::mem::size_of::<*mut i32>()
        );
        assert_eq!(
            core::mem::align_of::<CSlicePtr<i32, TrackingAlloc>>(),
            core::mem::align_of::<*mut i32>()
        );
    }

    #[gtest]
    fn c_slice_ptr_with_len_vec_mut_libc() {
        let mut ptr = CSlicePtr::<i32>::null();
        let mut count: c_int = 0;
        let mut handle = unsafe { ptr.with_len_vec_mut(&mut count) };
        handle.push_back(123);
        assert_that!(&*handle, container_eq([123]));
        handle.clear();
        assert!(ptr.is_null());
        assert_that!(count, eq(0));
    }
}

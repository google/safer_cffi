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
use core::ffi::c_void;
use core::ptr::NonNull;
use libc::size_t;

/// A `Box` that uses a C-compatible allocator (defaulting to [`LibcAlloc`]).
pub type CBox<T, A = LibcAlloc> = Box<T, A>;

/// A `Vec` that uses a C-compatible allocator (defaulting to [`LibcAlloc`]).
pub type CVec<T, A = LibcAlloc> = Vec<T, A>;

/// The maximum alignment guaranteed by standard `malloc`: that of C's `max_align_t`, taken from
/// `libc` on the targets where it defines it.
#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    all(target_os = "android", not(target_arch = "arm")),
    all(target_os = "windows", target_env = "gnu"),
))]
const MALLOC_ALIGN: usize = core::mem::align_of::<libc::max_align_t>();

/// The maximum alignment guaranteed by standard `malloc`.
///
/// `libc` does not define `max_align_t` for this target, so approximate it with a `repr(C)`
/// union of the fundamental types. Erring high would be unsound: a layout `malloc` cannot
/// satisfy would get through. Erring low only refuses layouts it could have served. The
/// `libc_alloc_results_are_aligned` test checks the result at runtime.
#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    all(target_os = "android", not(target_arch = "arm")),
    all(target_os = "windows", target_env = "gnu"),
)))]
const MALLOC_ALIGN: usize = {
    #[repr(C)]
    union MaxAlign {
        _f64: f64,
        // `u128` pulls this up to 16 bytes where the C ABI gives `long double` or `__int128`
        // that alignment. Rust aligns it to 16 on 32-bit x86 too, where `malloc` guarantees
        // 8, and on aarch64 Apple targets `long double` is `double`, so leave it out on both.
        #[cfg(all(
            target_pointer_width = "64",
            not(all(target_vendor = "apple", target_arch = "aarch64"))
        ))]
        _u128: u128,
        _u64: u64,
        _ptr: *const (),
    }
    core::mem::align_of::<MaxAlign>()
};

/// Returns a non-null, dangling slice pointer aligned to `layout.align()` with length 0.
///
/// This is used to satisfy the [`Allocator`] contract for zero-sized allocations
/// (`layout.size() == 0`) without calling the system allocator (`malloc(0)`).
#[inline]
fn dangling_slice(layout: Layout) -> NonNull<[u8]> {
    NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0)
}

/// Number of bytes to request from `malloc` for `layout`, or `None` if `malloc` cannot satisfy
/// its alignment.
///
/// C23 (7.24.3, via N2293) only promises that `malloc(n)` is aligned for types of size at most
/// `n`, so `malloc(1)` may return a 1-aligned pointer. WG14 treats this as a clarification of
/// earlier wording, and Miri's `malloc` implements it. Requesting at least `layout.align()`
/// bytes brings the guarantee up to the alignment asked for. `std::alloc::System` avoids the
/// same hole by using plain `malloc` only when `align <= size`.
///
/// # Safety-usable invariant
///
/// If this returns `Some(n)`, then `n >= layout.size()` and a non-null result of `malloc(n)`,
/// `calloc(n, 1)`, or `realloc(_, n)` is aligned to `layout.align()`. The second part follows
/// from C23 7.24.3 given that `layout.align() <= MALLOC_ALIGN` and that on every supported
/// target each power of two up to `alignof(max_align_t)` is the alignment of some
/// fundamental type no larger than itself (`char`, `short`, `int`, `long long` or a pointer,
/// `long double` or `__int128`). `libc_alloc_results_are_aligned` checks this at runtime.
#[inline]
fn malloc_size(layout: Layout) -> Option<usize> {
    (layout.align() <= MALLOC_ALIGN).then(|| layout.size().max(layout.align()))
}

/// Allocator backed by custom C allocation functions, such as the memory hooks that many C
/// libraries let their users install, for memory whose ownership crosses the FFI boundary.
///
/// A `CLikeAlloc<F>` calls the functions of an [`AllocFns`] value of type `F` that it stores:
///
/// - [`LibcFns`] for the C standard library's `malloc`, `calloc`, `realloc`, and `free`,
/// - [`CustomFns`] for plain `malloc`- or `calloc`-like C function pointers, or
/// - a type of your own that adapts allocation hooks with other signatures.
///
/// Create one with [`LibcAlloc`](const@LibcAlloc),
/// [`new_with_malloc`](CLikeAlloc::new_with_malloc), or
/// [`new_with_calloc`](CLikeAlloc::new_with_calloc) from function pointers, or with
/// [`new`](Self::new) from an `F` of your own.
///
/// [`grow`](Allocator::grow) and [`shrink`](Allocator::shrink) use `F`'s `realloc`. Memory
/// allocated through a `CLikeAlloc` can be deallocated through any copy of it, including on another
/// thread if `F` is [`Send`] or [`Sync`] (and so `CLikeAlloc<F>` is).
///
/// Beyond the [`Allocator`] contract, unsafe code may rely on the following. If `F` implements
/// [`PartialEq`], two `CLikeAlloc<F>`s compare equal if their `F`s do, and equal allocators are
/// equivalent (memory allocated through one may be deallocated through the other).
///
/// Every non-zero-sized block comes straight from its `F`'s `malloc`, `calloc`, or `realloc`, so
/// C code may release it with the matching `free`. Conversely, a block that C allocated with these
/// functions may be passed to `deallocate`, `grow`, and `shrink` with any layout whose size is
/// within the block. Zero-sized blocks are dangling and must not be passed to `free`.
///
/// Freeing memory requires the stored `F`, so apart from [`LibcAlloc`], `CLikeAlloc` does not
/// implement [`DropByPtrAllocator`] and cannot be used with [`OwnedCBufPtr`](crate::OwnedCBufPtr).
/// Instead, pass it to methods such as [`CBufPtr::as_vec_mut_in`](crate::CBufPtr::as_vec_mut_in).
///
/// # Example
///
/// ```
/// use core::ffi::c_int;
/// use safer_cffi::{CBufPtr, CLikeAlloc, CustomFns};
///
/// // SAFETY: These are the C standard library's `calloc` and `free`.
/// let alloc = unsafe { CLikeAlloc::new_with_calloc(libc::calloc, libc::free) };
///
/// let mut items: CBufPtr<c_int, CLikeAlloc<CustomFns>> = CBufPtr::null();
/// let mut item_len: c_int = 0;
/// // SAFETY: `items` is null and `item_len` is 0.
/// let mut items_vec = unsafe { items.as_vec_mut_in(&mut item_len, alloc) };
/// items_vec.push_back(1);
/// items_vec.push_back(2);
/// assert_eq!(items_vec.as_slice(), [1, 2]);
/// items_vec.clear();
/// ```
#[derive(Copy, Debug, Default)]
pub struct CLikeAlloc<F: AllocFns> {
    fns: F,
}

impl<F: AllocFns> CLikeAlloc<F> {
    /// Creates an allocator that allocates and frees memory with the methods of `fns`.
    #[inline]
    #[must_use]
    pub const fn new(fns: F) -> Self {
        Self { fns }
    }
}

/// C-style allocation functions for [`CLikeAlloc`].
///
/// [`CustomFns`] implements this trait for plain `malloc`- or `calloc`-like C function pointers.
/// For allocation hooks with other signatures, such as ones that take a context pointer, implement
/// it for a type that holds the hooks and their context, and adapt the hooks in its methods. Only
/// `malloc` and `free` are required: `calloc` and `realloc` default to implementations on top of
/// them, which you may override if the C library has hooks for them, so that, for example,
/// `CLikeAlloc` can grow blocks in place.
///
/// # Safety
///
/// Implementations must ensure that for every value of the type, the methods do not unwind and
/// behave like C's `malloc(size)`, `calloc(size, 1)`, `realloc(ptr, new_size)`, and `free(ptr)`
/// (C23 7.24.3), with the following differences:
///
/// - [`free`](Self::free) and [`realloc`](Self::realloc) may assume their `# Safety` preconditions:
///   in particular, `ptr` is non-null, and `new_size > 0`.
/// - [`realloc`](Self::realloc) also takes `old_size` (which may be less than the block's allocated
///   size) and on success need only preserve the first `min(old_size, new_size)` bytes.
/// - The methods need to be thread-safe only if the type is [`Send`] or [`Sync`].
///
/// Copies (and, if the type implements [`PartialEq`], equal values) must accept each other's
/// allocations.
///
/// # Example
///
/// miniz lets C callers set allocation hooks in an `mz_stream`, along with an `opaque` pointer
/// that it passes to them. The `unsafe impl` below vouches for every `MzHooks`, so its fields are
/// private and its constructor is `unsafe`:
///
/// ```
/// use core::ffi::c_void;
/// use safer_cffi::{AllocFns, CBox, CLikeAlloc};
///
/// /// `void *(*mz_alloc_func)(void *opaque, size_t items, size_t size)`
/// type MzAllocFunc = unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void;
/// /// `void (*mz_free_func)(void *opaque, void *address)`
/// type MzFreeFunc = unsafe extern "C" fn(*mut c_void, *mut c_void);
///
/// /// The allocation hooks of an `mz_stream`.
/// #[derive(Clone, Copy)]
/// struct MzHooks {
///     zalloc: MzAllocFunc,
///     zfree: MzFreeFunc,
///     opaque: *mut c_void,
/// }
///
/// impl MzHooks {
///     /// # Safety
///     ///
///     /// `zalloc(opaque, 1, n)` and `zfree(opaque, ptr)` must behave like `malloc(n)` and
///     /// `free(ptr)` while any copy of the result exists.
///     unsafe fn new(zalloc: MzAllocFunc, zfree: MzFreeFunc, opaque: *mut c_void) -> Self {
///         Self { zalloc, zfree, opaque }
///     }
/// }
///
/// // SAFETY: Every `MzHooks` comes from `MzHooks::new`, so its hooks behave like `malloc` and
/// // `free`. As `MzHooks` holds raw pointers, it is neither `Send` nor `Sync`.
/// unsafe impl AllocFns for MzHooks {
///     fn malloc(&self, size: usize) -> *mut c_void {
///         // SAFETY: `zalloc` can be called like `malloc` (see `MzHooks::new`).
///         unsafe { (self.zalloc)(self.opaque, 1, size) }
///     }
///
///     unsafe fn free(&self, ptr: *mut c_void) {
///         // SAFETY: `zfree` can be called like `free` (see `MzHooks::new`), and the caller
///         // ensures that `ptr` may be passed to it.
///         unsafe { (self.zfree)(self.opaque, ptr) }
///     }
/// }
///
/// # unsafe extern "C" fn zalloc(_: *mut c_void, items: usize, size: usize) -> *mut c_void {
/// #     unsafe { libc::malloc(items * size) }
/// # }
/// # unsafe extern "C" fn zfree(_: *mut c_void, address: *mut c_void) {
/// #     unsafe { libc::free(address) }
/// # }
/// // SAFETY: `zalloc` and `zfree` call `malloc` and `free`.
/// let hooks = unsafe { MzHooks::new(zalloc, zfree, core::ptr::null_mut()) };
/// let boxed = CBox::new_in(42, CLikeAlloc::new(hooks));
/// assert_eq!(*boxed, 42);
/// ```
pub unsafe trait AllocFns: Copy {
    /// Allocates `size` bytes, like C's `malloc(size)`.
    fn malloc(&self, size: usize) -> *mut c_void;

    /// Allocates `size` zero-initialized bytes, like C's `calloc(size, 1)`.
    ///
    /// The default implementation calls [`malloc`](Self::malloc) and zeroes the allocation.
    fn calloc(&self, size: usize) -> *mut c_void {
        default_calloc(self, size)
    }

    /// Resizes the block at `ptr` to `new_size` bytes, like C's `realloc(ptr, new_size)`.
    ///
    /// Returns either null, leaving the block unchanged, or a pointer to a block of at least
    /// `new_size` bytes that holds the first `min(old_size, new_size)` bytes of the old one, which
    /// it frees. The default implementation allocates a new block with [`malloc`](Self::malloc),
    /// copies `min(old_size, new_size)` bytes, and frees the old block with [`free`](Self::free).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` may be passed to [`free`](Self::free) and points to at
    /// least `old_size` bytes, and that `new_size > 0`.
    unsafe fn realloc(&self, ptr: *mut c_void, old_size: usize, new_size: usize) -> *mut c_void {
        let new_ptr = self.malloc(new_size);
        if !new_ptr.is_null() {
            // SAFETY: `ptr` is valid for reads of `old_size` bytes, and `new_ptr` is a fresh block,
            // valid for writes of `new_size` bytes, so the two do not overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    ptr.cast::<u8>(),
                    new_ptr.cast::<u8>(),
                    old_size.min(new_size),
                )
            };
            // SAFETY: The caller ensures that `ptr` may be passed to `free`. It is not used
            // afterwards.
            unsafe { self.free(ptr) };
        }
        new_ptr
    }

    /// Frees `ptr`, like C's `free(ptr)`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is a non-null pointer returned by `malloc`, `calloc`, or
    /// `realloc` of `self` (or of a copy or an equal value) that has not been freed.
    unsafe fn free(&self, ptr: *mut c_void);
}

/// The default implementation of [`AllocFns::calloc`]: allocates `size` bytes with
/// `fns.malloc` and zeroes them.
fn default_calloc(fns: &impl AllocFns, size: usize) -> *mut c_void {
    let ptr = fns.malloc(size);
    if !ptr.is_null() {
        // SAFETY: By the `AllocFns` contract, a non-null result of `malloc(size)` is a fresh
        // block, valid for writes of `size` bytes.
        unsafe { ptr.cast::<u8>().write_bytes(0, size) };
    }
    ptr
}

// SAFETY: The `Allocator` contract asks for three things.
//
// - Blocks stay valid until the allocator and its clones are dropped: by the `AllocFns` contract,
//   non-zero-sized blocks are released only by an explicit `free`/`realloc`; zero-sized blocks are
//   dangling pointers that are never dereferenced. `CLikeAlloc` has no `Drop`.
// - Clones and moves behave like the same allocator: clones are copies, and by the `AllocFns`
//   contract, copies of `fns` accept each other's allocations.
// - Any currently allocated block may be passed to any method: `deallocate`, `grow`, and `shrink`
//   use only `ptr` and the layout sizes, which every block satisfies.
//
// Returned blocks meet the requested layout: `malloc_size` (see its safety-usable invariant)
// picks a request size that is at least `layout.size()` and, as `fns` behaves like C's
// functions, makes the result aligned to `layout.align()`. `calloc` also zeroes the block, as
// `allocate_zeroed` promises.
unsafe impl<F: AllocFns> Allocator for CLikeAlloc<F> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }
        let size = malloc_size(layout).ok_or(AllocError)?;
        let ptr = self.fns.malloc(size);
        NonNull::new(ptr.cast::<u8>())
            .map(|p| NonNull::slice_from_raw_parts(p, layout.size()))
            .ok_or(AllocError)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }
        let size = malloc_size(layout).ok_or(AllocError)?;
        let ptr = self.fns.calloc(size);
        NonNull::new(ptr.cast::<u8>())
            .map(|p| NonNull::slice_from_raw_parts(p, layout.size()))
            .ok_or(AllocError)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: By the `deallocate` contract, `ptr` is currently allocated by this allocator
            // and `layout` fits it, so as `layout.size() != 0`, it may be passed to `free` (see the
            // `CLikeAlloc` docs). Afterwards `ptr` is invalid, as `deallocate` promises.
            unsafe { self.fns.free(ptr.as_ptr().cast::<c_void>()) };
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: `resize` requires `ptr` to be currently allocated by this allocator with
        // `old_layout` fitting it, which is the `grow` contract. The size ordering `grow` also
        // requires is not needed by `resize`.
        unsafe { self.resize(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: `resize` requires `ptr` to be currently allocated by this allocator with
        // `old_layout` fitting it, which is the `shrink` contract. The size ordering `shrink` also
        // requires is not needed by `resize`.
        unsafe { self.resize(ptr, old_layout, new_layout) }
    }
}

impl<F: AllocFns> CLikeAlloc<F> {
    /// Moves the block at `ptr` from `old_layout` to `new_layout`, in either direction. Shared
    /// body of [`Allocator::grow`] and [`Allocator::shrink`].
    ///
    /// On `Ok`, the first `min(old_layout.size(), new_layout.size())` bytes are preserved and
    /// `ptr` is no longer valid. On `Err`, `ptr` is still allocated and unchanged.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` denotes a block currently allocated via this allocator
    /// and that `old_layout` fits it, in the sense of the [`Allocator`] docs. `new_layout` may be
    /// larger or smaller than `old_layout` and may have a different alignment.
    #[inline]
    unsafe fn resize(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if new_layout.size() == 0 {
            // SAFETY: `ptr` is currently allocated with a fitting `old_layout`, as this function's
            // contract requires and `deallocate` needs.
            unsafe { self.deallocate(ptr, old_layout) };
            return Ok(dangling_slice(new_layout));
        }
        if old_layout.size() == 0 {
            // `ptr` is a dangling pointer that owns nothing, so there is nothing to move or free.
            return self.allocate(new_layout);
        }
        let size = malloc_size(new_layout).ok_or(AllocError)?;

        // SAFETY: By `malloc_size`'s safety-usable invariant, `size >= new_layout.size() > 0`. By
        // this function's contract `ptr` is currently allocated with a fitting `old_layout`, and
        // `old_layout.size() != 0` here, so it may be passed to `free` and points to at least
        // `old_layout.size()` bytes (see the `CLikeAlloc` docs). If the result is non-null it is
        // valid for reads and writes of `size` bytes with the first `min(old_layout.size(), size)`
        // bytes preserved, and by the same invariant aligned to `new_layout.align()`. If it is
        // null, `ptr` is still allocated and unchanged.
        let new_ptr =
            unsafe { self.fns.realloc(ptr.as_ptr().cast::<c_void>(), old_layout.size(), size) };
        NonNull::new(new_ptr.cast::<u8>())
            .map(|p| NonNull::slice_from_raw_parts(p, new_layout.size()))
            .ok_or(AllocError)
    }
}

/// Zero-sized allocator backed by standard C library `malloc`, `calloc`, `realloc`, and `free`,
/// for memory whose ownership crosses the FFI boundary.
///
/// `LibcAlloc` is a [`CLikeAlloc`] with [`LibcFns`], so [`CLikeAlloc`]'s guarantees apply to it.
/// Use the [`LibcAlloc`](const@LibcAlloc) constant to obtain an instance. All `LibcAlloc` values
/// are equivalent allocators: memory allocated through one may be deallocated, grown, or shrunk
/// through any other.
///
/// Every non-zero-sized block comes straight from `libc::malloc`, `libc::calloc`, or
/// `libc::realloc`, so C code may `free` or `realloc` it. Conversely, a block from C's `malloc` may
/// be passed to `deallocate`, `grow`, and `shrink` with any layout whose size is within the block.
pub type LibcAlloc = CLikeAlloc<LibcFns>;

/// Default [`LibcAlloc`](type@LibcAlloc) instance.
#[allow(non_upper_case_globals)]
pub const LibcAlloc: LibcAlloc = CLikeAlloc::new(LibcFns);

/// [`AllocFns`] for the C standard library's `malloc`, `calloc`, `realloc`, and `free`, which
/// [`LibcAlloc`] uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibcFns;

// SAFETY: These are the C standard library's allocation functions, which are thread-safe and do
// not unwind, and all `LibcFns` share them.
unsafe impl AllocFns for LibcFns {
    #[inline]
    fn malloc(&self, size: usize) -> *mut c_void {
        // SAFETY: `malloc` has no preconditions.
        unsafe { libc::malloc(size) }
    }

    #[inline]
    fn calloc(&self, size: usize) -> *mut c_void {
        // SAFETY: `calloc` has no preconditions.
        unsafe { libc::calloc(size, 1) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut c_void, _old_size: usize, new_size: usize) -> *mut c_void {
        // SAFETY: `realloc` requires a non-zero size and a pointer from `malloc`/`calloc`/`realloc`
        // that has not been freed, which the caller ensures (`LibcFns` uses these functions).
        unsafe { libc::realloc(ptr, new_size) }
    }

    #[inline]
    unsafe fn free(&self, ptr: *mut c_void) {
        // SAFETY: `free` requires a pointer from `malloc`/`calloc`/`realloc` that has not been
        // freed, which the caller ensures (`LibcFns` uses these functions).
        unsafe { libc::free(ptr) }
    }
}

/// A `malloc`-like C allocation function: `void *malloc(size_t size)`.
///
/// See [`CustomFns`]. For allocation functions with other signatures, such as ones that take a
/// context pointer, implement [`AllocFns`].
pub type MallocFn = unsafe extern "C" fn(size: size_t) -> *mut c_void;

/// A `calloc`-like C allocation function: `void *calloc(size_t nmemb, size_t size)`.
///
/// See [`CustomFns`]. For allocation functions with other signatures, implement [`AllocFns`].
pub type CallocFn = unsafe extern "C" fn(nmemb: size_t, size: size_t) -> *mut c_void;

/// A `free`-like C deallocation function: `void free(void *ptr)`.
///
/// See [`CustomFns`]. For deallocation functions with other signatures, implement [`AllocFns`].
pub type FreeFn = unsafe extern "C" fn(ptr: *mut c_void);

/// [`AllocFns`] for C allocation functions passed as function pointers: a `malloc`- or a
/// `calloc`-like function, and a `free`-like function. Create a `CLikeAlloc<CustomFns>` with
/// [`CLikeAlloc::new_with_malloc`] or [`CLikeAlloc::new_with_calloc`].
///
/// Two `CustomFns` compare equal if their function pointers do (via [`core::ptr::fn_addr_eq`]), so
/// separately created pointers to the same function may compare unequal.
#[derive(Clone, Copy, Debug)]
pub struct CustomFns {
    // Safety invariant: the functions behave like the C standard library functions of the same
    // name, including alignment and thread safety. `malloc`, `calloc`, or both are `Some`.
    malloc: Option<MallocFn>,
    calloc: Option<CallocFn>,
    free: FreeFn,
}

// SAFETY: By the field invariant, the functions behave like the C standard library's, including
// that they are thread-safe and do not unwind, and equal `CustomFns` hold the same functions.
unsafe impl AllocFns for CustomFns {
    #[inline]
    fn malloc(&self, size: usize) -> *mut c_void {
        match self.malloc {
            // SAFETY: By the field invariant, `malloc` behaves like C's `malloc`, which has no
            // preconditions.
            Some(malloc) => unsafe { malloc(size) },
            None => self.calloc(size),
        }
    }

    #[inline]
    fn calloc(&self, size: usize) -> *mut c_void {
        match self.calloc {
            // SAFETY: By the field invariant, `calloc` behaves like C's `calloc`, which has no
            // preconditions.
            Some(calloc) => unsafe { calloc(size, 1) },
            None => default_calloc(self, size),
        }
    }

    #[inline]
    unsafe fn free(&self, ptr: *mut c_void) {
        // SAFETY: By the field invariant, `self.free` behaves like C's `free`. The caller ensures
        // that `ptr` is an unfreed block from `self` or an equal value, which has the same
        // functions.
        unsafe { (self.free)(ptr) }
    }
}

impl CLikeAlloc<CustomFns> {
    /// Creates an allocator that allocates memory with `malloc` and frees it with `free`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `malloc` and `free` behave like the C standard library
    /// functions of the same name, including alignment and thread safety.
    #[inline]
    #[must_use]
    pub const unsafe fn new_with_malloc(malloc: MallocFn, free: FreeFn) -> Self {
        // The caller ensures the field invariant of `CustomFns`.
        Self::new(CustomFns { malloc: Some(malloc), calloc: None, free })
    }

    /// Creates an allocator that allocates memory with `calloc` and frees it with `free`.
    ///
    /// As there is no `malloc` function, all allocations are zero-initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `calloc` and `free` behave like the C standard library
    /// functions of the same name, including alignment and thread safety.
    #[inline]
    #[must_use]
    pub const unsafe fn new_with_calloc(calloc: CallocFn, free: FreeFn) -> Self {
        // The caller ensures the field invariant of `CustomFns`.
        Self::new(CustomFns { malloc: None, calloc: Some(calloc), free })
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
        // SAFETY: By the `deallocate_by_ptr` contract, `ptr` is currently allocated by `LibcAlloc`
        // with a non-zero size, so it may be passed to `free` (see the `LibcAlloc` docs). The
        // contract also forbids any later access through `ptr`.
        unsafe { LibcFns.free(ptr.as_ptr().cast::<c_void>()) };
    }
}

// Not derived, as the derived impl would call `F::clone`, which may do something other than copy
// `fns`, while a clone of an allocator must behave like the original.
impl<F: AllocFns> Clone for CLikeAlloc<F> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: AllocFns + PartialEq> PartialEq for CLikeAlloc<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.fns == other.fns
    }
}

impl<F: AllocFns + Eq> Eq for CLikeAlloc<F> {}

// Not derived, as comparing function pointers with `==` triggers the
// `unpredictable_function_pointer_comparisons` lint.
impl PartialEq for CustomFns {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        fn opt_eq<T>(a: Option<T>, b: Option<T>, eq: fn(T, T) -> bool) -> bool {
            match (a, b) {
                (Some(a), Some(b)) => eq(a, b),
                (None, None) => true,
                _ => false,
            }
        }
        opt_eq(self.malloc, other.malloc, core::ptr::fn_addr_eq)
            && opt_eq(self.calloc, other.calloc, core::ptr::fn_addr_eq)
            && core::ptr::fn_addr_eq(self.free, other.free)
    }
}

impl Eq for CustomFns {}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]
    use super::*;
    use crate::c_buf::CBufPtr;
    use core::cell::Cell;
    use core::ffi::c_int;
    use googletest::prelude::*;

    /// Checks that `allocate`, `allocate_zeroed`, `grow` and `shrink` of `alloc` serve every
    /// alignment `malloc` guarantees, return pointers aligned to the layout, and refuse anything
    /// stricter. Layouts smaller than their alignment are included: plain `malloc(size)` would not
    /// guarantee those.
    ///
    /// Under Miri this is exact, since its `malloc` aligns only as much as C requires. On a real
    /// libc a pass is evidence, not proof: allocators tend to over-align small blocks.
    fn check_results_are_aligned(alloc: &impl Allocator) {
        let is_aligned = |slice: NonNull<[u8]>, align: usize| {
            (slice.as_ptr() as *mut u8).addr().is_multiple_of(align)
        };

        let mut align = 1;
        while align <= MALLOC_ALIGN {
            for size in 1..=4 * MALLOC_ALIGN {
                let layout = Layout::from_size_align(size, align).unwrap();
                let larger = Layout::from_size_align(size + 1, align).unwrap();

                let slice = alloc.allocate(layout).unwrap();
                assert_that!(is_aligned(slice, align), eq(true), "allocate({size}, {align})");
                let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
                // SAFETY: `ptr` was allocated with `layout`, and `larger` is not smaller.
                let slice = unsafe { alloc.grow(ptr, layout, larger) }.unwrap();
                assert_that!(is_aligned(slice, align), eq(true), "grow({size}, {align})");
                let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
                // SAFETY: `ptr` was returned by `grow` with `larger`.
                unsafe { alloc.deallocate(ptr, larger) };

                let slice = alloc.allocate_zeroed(layout).unwrap();
                assert_that!(
                    is_aligned(slice, align),
                    eq(true),
                    "allocate_zeroed({size}, {align})"
                );
                let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
                // SAFETY: `ptr` was allocated with `layout`.
                unsafe { alloc.deallocate(ptr, layout) };
            }

            // Shrinking to below the alignment is the `realloc` form of the same case.
            let big = Layout::from_size_align(4 * MALLOC_ALIGN, align).unwrap();
            let one = Layout::from_size_align(1, align).unwrap();
            let ptr = NonNull::new(alloc.allocate(big).unwrap().as_ptr() as *mut u8).unwrap();
            // SAFETY: `ptr` was allocated with `big`, and `one` is not larger.
            let slice = unsafe { alloc.shrink(ptr, big, one) }.unwrap();
            assert_that!(is_aligned(slice, align), eq(true), "shrink(1, {align})");
            let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
            // SAFETY: `ptr` was returned by `shrink` with `one`.
            unsafe { alloc.deallocate(ptr, one) };

            align *= 2;
        }

        let too_strict = Layout::from_size_align(1, 2 * MALLOC_ALIGN).unwrap();
        assert_that!(alloc.allocate(too_strict), err(anything()));
        assert_that!(alloc.allocate_zeroed(too_strict), err(anything()));

        let small = Layout::from_size_align(1, 1).unwrap();
        let ptr = NonNull::new(alloc.allocate(small).unwrap().as_ptr() as *mut u8).unwrap();
        // SAFETY: `ptr` was allocated with `small`, and `too_strict` is not smaller.
        assert_that!(unsafe { alloc.grow(ptr, small, too_strict) }, err(anything()));
        // SAFETY: a failed `grow` leaves `ptr` allocated with `small`.
        unsafe { alloc.deallocate(ptr, small) };
    }

    #[gtest]
    fn libc_alloc_results_are_aligned() {
        check_results_are_aligned(&LibcAlloc);
    }

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
        let grown0 = unsafe { alloc.grow(non_null, layout0, layout0) }.unwrap();
        assert_that!(grown0.len(), eq(0));
        assert_that!((grown0.as_ptr() as *const u8 as usize) % layout0.align(), eq(0));
        let non_null0 = NonNull::new(grown0.as_ptr() as *mut u8).unwrap();

        // Grow 0 -> 16
        let grown16 = unsafe { alloc.grow(non_null0, layout0, layout16) }.unwrap();
        assert_that!(grown16.len(), eq(16));
        let non_null16 = NonNull::new(grown16.as_ptr() as *mut u8).unwrap();

        // Shrink 16 -> 0
        let shrunk0 = unsafe { alloc.shrink(non_null16, layout16, layout0) }.unwrap();
        assert_that!(shrunk0.len(), eq(0));
        assert_that!((shrunk0.as_ptr() as *const u8 as usize) % layout0.align(), eq(0));
        let non_null_final = NonNull::new(shrunk0.as_ptr() as *mut u8).unwrap();

        // SAFETY: `non_null_final` was returned by `shrink` with `layout0`.
        unsafe { alloc.deallocate(non_null_final, layout0) };
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
    fn libc_alloc_deallocate_by_ptr() {
        let alloc = LibcAlloc;
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(16));
        let non_null = NonNull::new(slice.as_ptr() as *mut u8).unwrap();
        // SAFETY: `non_null` was allocated by `LibcAlloc` with a non-zero layout,
        // no references or aliases exist, and ownership is transferred to `deallocate_by_ptr`.
        unsafe { LibcAlloc::deallocate_by_ptr(non_null) };
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

    #[gtest]
    fn libc_alloc_is_zst_and_pointer_sized_in_cbox() {
        fn assert_traits<T: Send + Sync + Copy + Clone + Eq + core::fmt::Debug>() {}
        fn assert_default<T: Default>() {}
        assert_traits::<LibcAlloc>();
        assert_traits::<CLikeAlloc<CustomFns>>();
        assert_default::<LibcAlloc>();

        expect_that!(core::mem::size_of::<LibcAlloc>(), eq(0));
        expect_that!(core::mem::align_of::<LibcAlloc>(), eq(1));
        expect_that!(
            core::mem::size_of::<Option<CBox<u64>>>(),
            eq(core::mem::size_of::<*mut u64>())
        );
    }

    // Call records for the C allocation functions below, so that `CLikeAlloc` tests can check
    // which functions are called and with which arguments, plus a switch to make allocations
    // fail. They are thread-local so that tests running in parallel don't interfere; each test
    // resets them in case a thread runs several tests.
    thread_local! {
        static MALLOC_CALLS: Cell<usize> = const { Cell::new(0) };
        static CALLOC_CALLS: Cell<usize> = const { Cell::new(0) };
        static FREE_CALLS: Cell<usize> = const { Cell::new(0) };
        static LAST_MALLOC_SIZE: Cell<usize> = const { Cell::new(0) };
        static LAST_CALLOC_ARGS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
        static REALLOC_CALLS: Cell<usize> = const { Cell::new(0) };
        static LAST_REALLOC_SIZE: Cell<usize> = const { Cell::new(0) };
        static LAST_CTX: Cell<usize> = const { Cell::new(0) };
        static FAIL_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    }

    fn reset_calls() {
        MALLOC_CALLS.set(0);
        CALLOC_CALLS.set(0);
        FREE_CALLS.set(0);
        LAST_MALLOC_SIZE.set(0);
        LAST_CALLOC_ARGS.set((0, 0));
        REALLOC_CALLS.set(0);
        LAST_REALLOC_SIZE.set(0);
        LAST_CTX.set(0);
        FAIL_ALLOCATIONS.set(false);
    }

    /// `libc::malloc` that records its calls and fills the memory with garbage, so that tests
    /// notice missing zero-initialization. Returns null while `FAIL_ALLOCATIONS` is set.
    unsafe extern "C" fn counting_malloc(size: usize) -> *mut c_void {
        MALLOC_CALLS.set(MALLOC_CALLS.get() + 1);
        LAST_MALLOC_SIZE.set(size);
        if FAIL_ALLOCATIONS.get() {
            return core::ptr::null_mut();
        }
        let ptr = unsafe { libc::malloc(size) };
        if !ptr.is_null() {
            unsafe { ptr.cast::<u8>().write_bytes(0xAB, size) };
        }
        ptr
    }

    /// `libc::calloc` that records its calls. Returns null while `FAIL_ALLOCATIONS` is set.
    unsafe extern "C" fn counting_calloc(nmemb: usize, size: usize) -> *mut c_void {
        CALLOC_CALLS.set(CALLOC_CALLS.get() + 1);
        LAST_CALLOC_ARGS.set((nmemb, size));
        if FAIL_ALLOCATIONS.get() {
            return core::ptr::null_mut();
        }
        unsafe { libc::calloc(nmemb, size) }
    }

    /// `libc::free` that counts its calls.
    unsafe extern "C" fn counting_free(ptr: *mut c_void) {
        FREE_CALLS.set(FREE_CALLS.get() + 1);
        unsafe { libc::free(ptr) };
    }

    /// `libc::realloc` that records its calls. Returns null, leaving `ptr` allocated, while
    /// `FAIL_ALLOCATIONS` is set.
    unsafe extern "C" fn counting_realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
        REALLOC_CALLS.set(REALLOC_CALLS.get() + 1);
        LAST_REALLOC_SIZE.set(size);
        if FAIL_ALLOCATIONS.get() {
            return core::ptr::null_mut();
        }
        unsafe { libc::realloc(ptr, size) }
    }

    /// `libc::free` under another name, for an allocator that differs only in its `free`.
    unsafe extern "C" fn other_free(ptr: *mut c_void) {
        unsafe { libc::free(ptr) };
    }

    fn counting_malloc_alloc() -> CLikeAlloc<CustomFns> {
        // SAFETY: `counting_malloc` and `counting_free` wrap `libc::malloc` and `libc::free`.
        unsafe { CLikeAlloc::new_with_malloc(counting_malloc, counting_free) }
    }

    fn counting_calloc_alloc() -> CLikeAlloc<CustomFns> {
        // SAFETY: `counting_calloc` and `counting_free` wrap `libc::calloc` and `libc::free`.
        unsafe { CLikeAlloc::new_with_calloc(counting_calloc, counting_free) }
    }

    /// `AllocFns` with a context, like many C allocation hooks, which `malloc` and `free` record
    /// in `LAST_CTX`. `calloc` is the default implementation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct CtxFns {
        ctx: usize,
    }

    // SAFETY: `counting_malloc` and `counting_free` wrap `libc::malloc` and `libc::free`.
    unsafe impl AllocFns for CtxFns {
        fn malloc(&self, size: usize) -> *mut c_void {
            LAST_CTX.set(self.ctx);
            unsafe { counting_malloc(size) }
        }

        unsafe fn free(&self, ptr: *mut c_void) {
            LAST_CTX.set(self.ctx);
            unsafe { counting_free(ptr) }
        }
    }

    fn ctx_alloc(ctx: usize) -> CLikeAlloc<CtxFns> {
        CLikeAlloc::new(CtxFns { ctx })
    }

    /// `AllocFns` with both a `malloc`- and a `calloc`-like function, like hiredis's hooks.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct MallocAndCallocFns;

    // SAFETY: The functions wrap `libc::malloc`, `libc::calloc`, and `libc::free`.
    unsafe impl AllocFns for MallocAndCallocFns {
        fn malloc(&self, size: usize) -> *mut c_void {
            unsafe { counting_malloc(size) }
        }

        fn calloc(&self, size: usize) -> *mut c_void {
            unsafe { counting_calloc(size, 1) }
        }

        unsafe fn free(&self, ptr: *mut c_void) {
            unsafe { counting_free(ptr) }
        }
    }

    fn malloc_and_calloc_alloc() -> CLikeAlloc<MallocAndCallocFns> {
        CLikeAlloc::new(MallocAndCallocFns)
    }

    /// `AllocFns` with a `realloc`-like function, which `grow` and `shrink` use.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ReallocFns;

    // SAFETY: The functions wrap `libc::malloc`, `libc::realloc`, and `libc::free`.
    unsafe impl AllocFns for ReallocFns {
        fn malloc(&self, size: usize) -> *mut c_void {
            unsafe { counting_malloc(size) }
        }

        unsafe fn realloc(
            &self,
            ptr: *mut c_void,
            _old_size: usize,
            new_size: usize,
        ) -> *mut c_void {
            unsafe { counting_realloc(ptr, new_size) }
        }

        unsafe fn free(&self, ptr: *mut c_void) {
            unsafe { counting_free(ptr) }
        }
    }

    fn realloc_alloc() -> CLikeAlloc<ReallocFns> {
        CLikeAlloc::new(ReallocFns)
    }

    #[gtest]
    fn c_like_alloc_with_libc_functions() {
        /// Zero-sized `AllocFns` for functions that are known at compile time.
        #[derive(Clone, Copy)]
        struct Libc;

        // SAFETY: These are the C standard library's allocation functions.
        unsafe impl AllocFns for Libc {
            fn malloc(&self, size: usize) -> *mut c_void {
                unsafe { libc::malloc(size) }
            }

            unsafe fn free(&self, ptr: *mut c_void) {
                unsafe { libc::free(ptr) }
            }
        }

        // The constructors are `const`, so allocators can be stored in statics.
        // SAFETY: These are the C standard library's allocation functions.
        static LIBC_MALLOC: CLikeAlloc<CustomFns> =
            unsafe { CLikeAlloc::new_with_malloc(libc::malloc, libc::free) };
        // SAFETY: These are the C standard library's allocation functions.
        static LIBC_CALLOC: CLikeAlloc<CustomFns> =
            unsafe { CLikeAlloc::new_with_calloc(libc::calloc, libc::free) };
        static LIBC: CLikeAlloc<Libc> = CLikeAlloc::new(Libc);

        let b = CBox::new_in([1u64, 2, 3], LIBC_MALLOC);
        assert_that!(*b, eq([1, 2, 3]));
        let b = CBox::new_in([1u64, 2, 3], LIBC_CALLOC);
        assert_that!(*b, eq([1, 2, 3]));
        let b = CBox::new_in([1u64, 2, 3], LIBC);
        assert_that!(*b, eq([1, 2, 3]));
        expect_that!(core::mem::size_of::<CLikeAlloc<Libc>>(), eq(0));
    }

    #[gtest]
    fn c_like_alloc_results_are_aligned() {
        // SAFETY: These are the C standard library's allocation functions.
        let (malloc_alloc, calloc_alloc) = unsafe {
            (
                CLikeAlloc::new_with_malloc(libc::malloc, libc::free),
                CLikeAlloc::new_with_calloc(libc::calloc, libc::free),
            )
        };
        check_results_are_aligned(&malloc_alloc);
        check_results_are_aligned(&calloc_alloc);
        check_results_are_aligned(&ctx_alloc(7));
        check_results_are_aligned(&malloc_and_calloc_alloc());
        check_results_are_aligned(&realloc_alloc());
    }

    #[gtest]
    fn c_like_alloc_requests_at_least_the_alignment() {
        // C23 only promises that `malloc(n)` is aligned for types of size at most `n`, so a layout
        // smaller than its alignment must request more bytes than its size (see `malloc_size`).
        reset_calls();
        let layout = Layout::from_size_align(1, 8).unwrap();

        let alloc = counting_malloc_alloc();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(1));
        assert_that!(LAST_MALLOC_SIZE.get(), eq(8));
        // SAFETY: `slice` was allocated with `layout`.
        unsafe { alloc.deallocate(slice.cast(), layout) };

        let alloc = counting_calloc_alloc();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(1));
        assert_that!(LAST_CALLOC_ARGS.get(), eq((8, 1)));
        // SAFETY: `slice` was allocated with `layout`.
        unsafe { alloc.deallocate(slice.cast(), layout) };
    }

    #[gtest]
    fn c_like_alloc_malloc_allocate_and_deallocate() {
        reset_calls();
        let alloc = counting_malloc_alloc();
        let layout = Layout::array::<u32>(4).unwrap();
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(slice.len(), eq(16));
        assert_that!(MALLOC_CALLS.get(), eq(1));
        assert_that!(LAST_MALLOC_SIZE.get(), eq(16));
        assert_that!(FREE_CALLS.get(), eq(0));

        // SAFETY: `slice` was allocated with `layout`.
        unsafe { alloc.deallocate(slice.cast(), layout) };
        assert_that!(MALLOC_CALLS.get(), eq(1));
        assert_that!(FREE_CALLS.get(), eq(1));
    }

    #[gtest]
    fn c_like_alloc_malloc_allocate_zeroed() {
        reset_calls();
        let alloc = counting_malloc_alloc();
        let layout = Layout::array::<u32>(4).unwrap();
        // The default `calloc` zeroes memory from `malloc`.
        let slice = alloc.allocate_zeroed(layout).expect("allocate_zeroed succeeds");
        assert_that!(MALLOC_CALLS.get(), eq(1));
        // SAFETY: `slice` has 16 bytes initialized to zero.
        assert_that!(unsafe { slice.as_ref() }, container_eq([0; 16]));

        // SAFETY: `slice` was allocated with `layout`.
        unsafe { alloc.deallocate(slice.cast(), layout) };
        assert_that!(FREE_CALLS.get(), eq(1));
    }

    #[gtest]
    fn c_like_alloc_calloc_allocate_and_deallocate() {
        reset_calls();
        let alloc = counting_calloc_alloc();
        let layout = Layout::array::<u32>(4).unwrap();
        // Both `allocate` and `allocate_zeroed` use `calloc`.
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(LAST_CALLOC_ARGS.get(), eq((16, 1)));
        let zeroed = alloc.allocate_zeroed(layout).expect("allocate_zeroed succeeds");
        assert_that!(CALLOC_CALLS.get(), eq(2));
        assert_that!(zeroed.len(), eq(16));
        // SAFETY: `zeroed` has 16 bytes initialized to zero.
        assert_that!(unsafe { zeroed.as_ref() }, container_eq([0; 16]));

        // SAFETY: `slice` and `zeroed` were allocated with `layout`.
        unsafe {
            alloc.deallocate(slice.cast(), layout);
            alloc.deallocate(zeroed.cast(), layout);
        }
        assert_that!(FREE_CALLS.get(), eq(2));
    }

    #[gtest]
    fn c_like_alloc_zero_size() {
        reset_calls();
        let alloc = counting_malloc_alloc();
        let layout0 = Layout::from_size_align(0, 8).unwrap();
        let layout16 = Layout::from_size_align(16, 8).unwrap();

        // Zero-sized allocations don't call the C functions.
        let slice = alloc.allocate(layout0).expect("allocation succeeds");
        assert_that!(slice.len(), eq(0));
        assert_that!(slice.cast::<u8>().as_ptr().addr() % layout0.align(), eq(0));
        let zeroed = alloc.allocate_zeroed(layout0).expect("allocate_zeroed succeeds");
        assert_that!(zeroed.len(), eq(0));
        assert_that!(MALLOC_CALLS.get(), eq(0));

        // Grow 0 -> 16
        // SAFETY: `slice` was allocated with `layout0`.
        let grown = unsafe { alloc.grow(slice.cast(), layout0, layout16) }.expect("grow succeeds");
        assert_that!(grown.len(), eq(16));
        assert_that!(MALLOC_CALLS.get(), eq(1));
        assert_that!(FREE_CALLS.get(), eq(0));

        // Shrink 16 -> 0
        // SAFETY: `grown` was allocated with `layout16`.
        let shrunk =
            unsafe { alloc.shrink(grown.cast(), layout16, layout0) }.expect("shrink succeeds");
        assert_that!(shrunk.len(), eq(0));
        assert_that!(shrunk.cast::<u8>().as_ptr().addr() % layout0.align(), eq(0));
        assert_that!(FREE_CALLS.get(), eq(1));

        // SAFETY: `shrunk` and `zeroed` were allocated with `layout0`.
        unsafe {
            alloc.deallocate(shrunk.cast(), layout0);
            alloc.deallocate(zeroed.cast(), layout0);
        }
        assert_that!(MALLOC_CALLS.get(), eq(1));
        assert_that!(FREE_CALLS.get(), eq(1));
    }

    #[gtest]
    fn c_like_alloc_grow_and_shrink() {
        reset_calls();
        let alloc = counting_malloc_alloc();
        let l1 = Layout::array::<u32>(2).unwrap();
        let l2 = Layout::array::<u32>(8).unwrap();
        let slice = alloc.allocate(l1).expect("allocation succeeds");
        // SAFETY: `slice` is valid for writes of 2 `u32`s and aligned for `u32`.
        unsafe { slice.cast::<[u32; 2]>().write([1, 2]) };

        // With the default `realloc`, growing allocates a new block, copies the contents, and
        // frees the old block.
        // SAFETY: `slice` was allocated with `l1`, and `l2.size() >= l1.size()`.
        let grown = unsafe { alloc.grow(slice.cast(), l1, l2) }.expect("grow succeeds");
        assert_that!(grown.len(), eq(32));
        // SAFETY: The first 2 `u32`s were copied from `slice`.
        assert_that!(unsafe { grown.cast::<[u32; 2]>().read() }, eq([1, 2]));
        assert_that!(MALLOC_CALLS.get(), eq(2));
        assert_that!(FREE_CALLS.get(), eq(1));

        // SAFETY: `grown` was allocated with `l2`, and `l1.size() <= l2.size()`.
        let shrunk = unsafe { alloc.shrink(grown.cast(), l2, l1) }.expect("shrink succeeds");
        assert_that!(shrunk.len(), eq(8));
        // SAFETY: The first 2 `u32`s were copied from `grown`.
        assert_that!(unsafe { shrunk.cast::<[u32; 2]>().read() }, eq([1, 2]));
        assert_that!(MALLOC_CALLS.get(), eq(3));
        assert_that!(FREE_CALLS.get(), eq(2));

        // `grow_zeroed` zeroes the new bytes, although `malloc` fills them with garbage.
        // SAFETY: `shrunk` was allocated with `l1`, and `l2.size() >= l1.size()`.
        let grown_zeroed =
            unsafe { alloc.grow_zeroed(shrunk.cast(), l1, l2) }.expect("grow_zeroed succeeds");
        // SAFETY: The first 2 `u32`s were copied from `shrunk`, and the other 6 are zeroed.
        assert_that!(
            unsafe { grown_zeroed.cast::<[u32; 8]>().read() },
            eq([1, 2, 0, 0, 0, 0, 0, 0])
        );
        assert_that!(MALLOC_CALLS.get(), eq(4));
        assert_that!(FREE_CALLS.get(), eq(3));

        // SAFETY: `grown_zeroed` was allocated with `l2`.
        unsafe { alloc.deallocate(grown_zeroed.cast(), l2) };
        assert_that!(FREE_CALLS.get(), eq(4));
    }

    #[gtest]
    fn c_like_alloc_over_aligned_layout_fails() {
        reset_calls();
        let alloc = counting_malloc_alloc();
        let layout = Layout::from_size_align(16, MALLOC_ALIGN * 2).unwrap();
        assert_that!(alloc.allocate(layout), err(anything()));
        assert_that!(alloc.allocate_zeroed(layout), err(anything()));
        assert_that!(MALLOC_CALLS.get(), eq(0));
    }

    #[gtest]
    fn c_like_alloc_allocation_failure() {
        fn check_allocation_failure(alloc: &impl Allocator) {
            reset_calls();
            let layout = Layout::array::<u32>(4).unwrap();
            let larger = Layout::array::<u32>(8).unwrap();
            let slice = alloc.allocate(layout).expect("allocation succeeds");
            // SAFETY: `slice` is valid for writes of a `u32` and aligned for `u32`.
            unsafe { slice.cast::<u32>().write(42) };

            FAIL_ALLOCATIONS.set(true);
            assert_that!(alloc.allocate(layout), err(eq(AllocError)));
            assert_that!(alloc.allocate_zeroed(layout), err(eq(AllocError)));
            // SAFETY: `slice` was allocated with `layout`, and `larger` is not smaller.
            assert_that!(unsafe { alloc.grow(slice.cast(), layout, larger) }, err(eq(AllocError)));
            FAIL_ALLOCATIONS.set(false);

            // A failed `grow` leaves the block allocated and unchanged.
            assert_that!(FREE_CALLS.get(), eq(0));
            // SAFETY: `slice` is still allocated, and its first `u32` was written above.
            assert_that!(unsafe { slice.cast::<u32>().read() }, eq(42));
            // SAFETY: `slice` was allocated with `layout`.
            unsafe { alloc.deallocate(slice.cast(), layout) };
            assert_that!(FREE_CALLS.get(), eq(1));
        }

        check_allocation_failure(&counting_malloc_alloc());
        check_allocation_failure(&counting_calloc_alloc());
        check_allocation_failure(&malloc_and_calloc_alloc());
    }

    #[gtest]
    fn c_like_alloc_with_c_buf_ptr() {
        reset_calls();
        let mut ptr = CBufPtr::<i32, CLikeAlloc<CustomFns>>::null();
        let mut len: c_int = 0;
        // SAFETY: `ptr` is null and `len` is 0.
        let mut vec = unsafe { ptr.as_vec_mut_in(&mut len, counting_calloc_alloc()) };
        vec.push_back(1);
        vec.push_back(2);
        vec.push_back(3);
        assert_that!(vec.as_slice(), container_eq([1, 2, 3]));
        vec.clear();
        // Each push reallocates, and every allocation is freed.
        assert_that!(CALLOC_CALLS.get(), eq(3));
        assert_that!(FREE_CALLS.get(), eq(3));
    }

    #[gtest]
    fn c_like_alloc_with_cbox_cvec_and_c_buf_ptr() {
        reset_calls();
        let alloc = counting_calloc_alloc();

        let b = CBox::new_in(777i32, alloc);
        assert_that!(*b, eq(777));
        drop(b);
        assert_that!(CALLOC_CALLS.get(), eq(1));
        assert_that!(FREE_CALLS.get(), eq(1));

        let mut vec: CVec<i32, CLikeAlloc<CustomFns>> = CVec::new_in(alloc);
        vec.extend_from_slice(&[10, 20, 30]);
        let mut ptr1 = CBufPtr::from_boxed_slice(vec.into_boxed_slice());
        let mut len1 = 3usize;
        let mut ptr2 = CBufPtr::clone_and_leak_in(&[40, 50], alloc);
        let mut len2 = 2usize;
        {
            // SAFETY: `ptr1` and `ptr2` hold `len1` and `len2` elements allocated by `alloc`.
            let mut h1 = unsafe { ptr1.as_vec_mut_in(&mut len1, alloc) };
            let mut h2 = unsafe { ptr2.as_vec_mut_in(&mut len2, alloc) };
            h1.push_back(35);
            // `swap` checks that both handles use equal allocators.
            h1.swap(&mut h2);
            assert_that!(&*h1, container_eq([40, 50]));
            assert_that!(&*h2, container_eq([10, 20, 30, 35]));
            h1.clear();
            h2.clear();
        }
        assert_that!(ptr1.is_null(), eq(true));
        assert_that!(ptr2.is_null(), eq(true));
        // Every allocation was freed.
        assert_that!(FREE_CALLS.get(), eq(CALLOC_CALLS.get()));
    }

    #[gtest]
    fn c_like_alloc_eq() {
        let alloc = counting_malloc_alloc();
        let copy = alloc;
        assert_that!(copy, eq(alloc));
        // Separately created pointers to the same function may differ (Miri tests this), so
        // allocators from two `counting_malloc_alloc()` calls are not necessarily equal.
        // SAFETY: `counting_malloc` and `other_free` wrap `libc::malloc` and `libc::free`.
        let other_free_alloc = unsafe { CLikeAlloc::new_with_malloc(counting_malloc, other_free) };
        assert_that!(other_free_alloc, not(eq(alloc)));
        // An allocator with a `calloc` function differs from one with a `malloc` function.
        assert_that!(counting_calloc_alloc(), not(eq(alloc)));
    }

    #[gtest]
    fn c_like_alloc_custom_fns_with_context() {
        reset_calls();
        let alloc = ctx_alloc(7);
        let l1 = Layout::array::<u32>(2).unwrap();
        let l2 = Layout::array::<u32>(8).unwrap();
        let slice = alloc.allocate(l1).expect("allocation succeeds");
        assert_that!(LAST_MALLOC_SIZE.get(), eq(8));
        assert_that!(LAST_CTX.get(), eq(7));
        // SAFETY: `slice` is valid for writes of 2 `u32`s and aligned for `u32`.
        unsafe { slice.cast::<[u32; 2]>().write([1, 2]) };

        LAST_CTX.set(0);
        // SAFETY: `slice` was allocated with `l1`, and `l2.size() >= l1.size()`.
        let grown = unsafe { alloc.grow(slice.cast(), l1, l2) }.expect("grow succeeds");
        // SAFETY: The first 2 `u32`s were copied from `slice`.
        assert_that!(unsafe { grown.cast::<[u32; 2]>().read() }, eq([1, 2]));
        assert_that!(LAST_CTX.get(), eq(7));

        // The default `calloc` zeroes memory from `malloc`, which fills it with garbage.
        let zeroed = alloc.allocate_zeroed(l1).expect("allocate_zeroed succeeds");
        // SAFETY: `zeroed` has 8 bytes initialized to zero.
        assert_that!(unsafe { zeroed.as_ref() }, container_eq([0; 8]));

        LAST_CTX.set(0);
        // SAFETY: `grown` was allocated with `l2`, and `zeroed` with `l1`.
        unsafe {
            alloc.deallocate(grown.cast(), l2);
            alloc.deallocate(zeroed.cast(), l1);
        }
        assert_that!(LAST_CTX.get(), eq(7));
        assert_that!(MALLOC_CALLS.get(), eq(3));
        assert_that!(CALLOC_CALLS.get(), eq(0));
        assert_that!(FREE_CALLS.get(), eq(3));
    }

    #[gtest]
    fn c_like_alloc_custom_fns_with_malloc_and_calloc() {
        reset_calls();
        let alloc = malloc_and_calloc_alloc();
        let layout = Layout::array::<u32>(4).unwrap();
        // `allocate` uses `malloc`, and `allocate_zeroed` uses `calloc`.
        let slice = alloc.allocate(layout).expect("allocation succeeds");
        assert_that!(MALLOC_CALLS.get(), eq(1));
        assert_that!(CALLOC_CALLS.get(), eq(0));
        let zeroed = alloc.allocate_zeroed(layout).expect("allocate_zeroed succeeds");
        assert_that!(MALLOC_CALLS.get(), eq(1));
        assert_that!(CALLOC_CALLS.get(), eq(1));
        assert_that!(LAST_CALLOC_ARGS.get(), eq((16, 1)));
        // SAFETY: `zeroed` has 16 bytes initialized to zero.
        assert_that!(unsafe { zeroed.as_ref() }, container_eq([0; 16]));

        // SAFETY: `slice` and `zeroed` were allocated with `layout`.
        unsafe {
            alloc.deallocate(slice.cast(), layout);
            alloc.deallocate(zeroed.cast(), layout);
        }
        assert_that!(FREE_CALLS.get(), eq(2));
    }

    #[gtest]
    fn c_like_alloc_custom_fns_with_realloc() {
        reset_calls();
        let alloc = realloc_alloc();
        let l1 = Layout::array::<u32>(2).unwrap();
        let l2 = Layout::array::<u32>(8).unwrap();
        let slice = alloc.allocate(l1).expect("allocation succeeds");
        // SAFETY: `slice` is valid for writes of 2 `u32`s and aligned for `u32`.
        unsafe { slice.cast::<[u32; 2]>().write([1, 2]) };

        // `grow` and `shrink` use `realloc` instead of allocating, copying, and freeing.
        // SAFETY: `slice` was allocated with `l1`, and `l2.size() >= l1.size()`.
        let grown = unsafe { alloc.grow(slice.cast(), l1, l2) }.expect("grow succeeds");
        assert_that!(grown.len(), eq(32));
        assert_that!(LAST_REALLOC_SIZE.get(), eq(32));
        // SAFETY: `realloc` preserved the first 2 `u32`s.
        assert_that!(unsafe { grown.cast::<[u32; 2]>().read() }, eq([1, 2]));
        // SAFETY: `grown` was allocated with `l2`, and `l1.size() <= l2.size()`.
        let shrunk = unsafe { alloc.shrink(grown.cast(), l2, l1) }.expect("shrink succeeds");
        assert_that!(shrunk.len(), eq(8));
        assert_that!(LAST_REALLOC_SIZE.get(), eq(8));
        // SAFETY: `realloc` preserved the first 2 `u32`s.
        assert_that!(unsafe { shrunk.cast::<[u32; 2]>().read() }, eq([1, 2]));
        assert_that!(REALLOC_CALLS.get(), eq(2));
        assert_that!(MALLOC_CALLS.get(), eq(1));
        assert_that!(FREE_CALLS.get(), eq(0));

        // A failed `realloc` leaves the block allocated and unchanged.
        FAIL_ALLOCATIONS.set(true);
        // SAFETY: `shrunk` was allocated with `l1`, and `l2.size() >= l1.size()`.
        assert_that!(unsafe { alloc.grow(shrunk.cast(), l1, l2) }, err(eq(AllocError)));
        FAIL_ALLOCATIONS.set(false);
        // SAFETY: `shrunk` is still allocated, and its first 2 `u32`s were written above.
        assert_that!(unsafe { shrunk.cast::<[u32; 2]>().read() }, eq([1, 2]));

        // Shrinking to zero bytes frees the block instead.
        let layout0 = Layout::from_size_align(0, l1.align()).unwrap();
        // SAFETY: `shrunk` was allocated with `l1`, and `layout0.size() <= l1.size()`.
        let empty = unsafe { alloc.shrink(shrunk.cast(), l1, layout0) }.expect("shrink succeeds");
        assert_that!(empty.len(), eq(0));
        assert_that!(REALLOC_CALLS.get(), eq(3));
        assert_that!(FREE_CALLS.get(), eq(1));
    }

    #[gtest]
    fn c_like_alloc_custom_fns_eq() {
        let alloc = ctx_alloc(1);
        let copy = alloc;
        assert_that!(copy, eq(alloc));
        #[allow(clippy::clone_on_copy)]
        let clone = alloc.clone();
        assert_that!(clone, eq(alloc));
        // Allocators compare equal if their `AllocFns` do.
        assert_that!(ctx_alloc(1), eq(alloc));
        assert_that!(ctx_alloc(2), not(eq(alloc)));
    }

    #[gtest]
    fn c_like_alloc_custom_fns_with_c_buf_ptr_swap() {
        reset_calls();
        let alloc = ctx_alloc(7);
        let mut ptr1 = CBufPtr::clone_and_leak_in(&[1, 2], alloc);
        let mut len1 = 2usize;
        let mut ptr2 = CBufPtr::clone_and_leak_in(&[3], alloc);
        let mut len2 = 1usize;
        {
            // SAFETY: `ptr1` and `ptr2` hold `len1` and `len2` elements allocated by `alloc`.
            let mut h1 = unsafe { ptr1.as_vec_mut_in(&mut len1, alloc) };
            let mut h2 = unsafe { ptr2.as_vec_mut_in(&mut len2, alloc) };
            // `swap` checks that both handles use equal allocators, which compares the contexts.
            h1.swap(&mut h2);
            assert_that!(&*h1, container_eq([3]));
            assert_that!(&*h2, container_eq([1, 2]));
            h1.clear();
            h2.clear();
        }
        // Every allocation was freed, with the context.
        assert_that!(FREE_CALLS.get(), eq(MALLOC_CALLS.get()));
        assert_that!(LAST_CTX.get(), eq(7));
    }

    #[gtest]
    fn c_like_alloc_with_miniz_like_hooks() {
        // Allocation hooks with the signatures of miniz's `mz_alloc_func` and `mz_free_func`, as in
        // the `AllocFns` example. They record their `opaque` pointer.
        #[derive(Clone, Copy)]
        struct MzHooks {
            zalloc: unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void,
            zfree: unsafe extern "C" fn(*mut c_void, *mut c_void),
            opaque: *mut c_void,
        }

        // SAFETY: The only `MzHooks` below holds `zalloc` and `zfree`, which wrap `malloc` and
        // `free`.
        unsafe impl AllocFns for MzHooks {
            fn malloc(&self, size: usize) -> *mut c_void {
                unsafe { (self.zalloc)(self.opaque, 1, size) }
            }

            unsafe fn free(&self, ptr: *mut c_void) {
                unsafe { (self.zfree)(self.opaque, ptr) }
            }
        }

        unsafe extern "C" fn zalloc(opaque: *mut c_void, items: usize, size: usize) -> *mut c_void {
            LAST_CTX.set(opaque.addr());
            unsafe { counting_malloc(items * size) }
        }

        unsafe extern "C" fn zfree(opaque: *mut c_void, address: *mut c_void) {
            LAST_CTX.set(opaque.addr());
            unsafe { counting_free(address) }
        }

        reset_calls();
        let mut state = 0u8;
        let opaque = (&raw mut state).cast::<c_void>();
        let alloc = CLikeAlloc::new(MzHooks { zalloc, zfree, opaque });
        let boxed = CBox::new_in(42, alloc);
        assert_that!(*boxed, eq(42));
        assert_that!(LAST_MALLOC_SIZE.get(), eq(4));
        assert_that!(LAST_CTX.get(), eq(opaque.addr()));

        LAST_CTX.set(0);
        drop(boxed);
        assert_that!(FREE_CALLS.get(), eq(1));
        assert_that!(LAST_CTX.get(), eq(opaque.addr()));
    }
}

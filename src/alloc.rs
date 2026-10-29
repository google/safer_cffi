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

/// Zero-sized allocator backed by standard C library `malloc`, `calloc`, `realloc`, and `free`,
/// for memory whose ownership crosses the FFI boundary.
///
/// All `LibcAlloc` values are equivalent allocators: memory allocated through one may be
/// deallocated, grown, or shrunk through any other.
///
/// Beyond the [`Allocator`] contract, unsafe code may rely on the following:
///
/// - Every non-zero-sized block comes straight from `libc::malloc`, `libc::calloc`, or
///   `libc::realloc`, with no header or offset. C code may `free` or `realloc` it. Conversely,
///   `deallocate`, `grow`, and `shrink` use only `ptr` and the layout sizes, never
///   `old_layout.align()`, so a block from C's `malloc` may be passed to them with any layout whose
///   size is within the block.
/// - A block is aligned to its layout even when the layout is smaller than its alignment, which
///   plain `malloc(size)` does not guarantee (see `malloc_size`).
/// - A layout with alignment above `MALLOC_ALIGN` is refused with `AllocError`, never served
///   misaligned.
/// - Zero-sized layouts never reach the C allocation functions; the dangling pointer they get must
///   not be passed to `free`. `deallocate` handles this itself; code that frees through C must
///   check the size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibcAlloc;

// SAFETY: The `Allocator` contract asks for the following. All `LibcAlloc` values are equivalent
// (see the `LibcAlloc` docs), which is consistent because no method reads `self`.
//
// - Currently allocated blocks point to valid memory until they are invalidated: non-zero-sized
//   blocks live in the C heap, and `LibcAlloc` releases a block only when it is passed to
//   `deallocate`, a reallocating method, or `deallocate_by_ptr`; zero-sized blocks are dangling
//   pointers that are never dereferenced. `LibcAlloc` holds no state.
// - Returned blocks satisfy the allocation invariants of `core::ptr` and are wholly disjoint: a
//   non-zero-sized block is a prefix of a live object from `malloc`, `calloc`, or `realloc`, which
//   is an allocation and, by C23 7.24.3, disjoint from any other object; a zero-sized block
//   contains no addresses.
// - No method unwinds: they call only `malloc`, `calloc`, `realloc`, and `free`, which are C
//   functions and do not unwind, plus code that does not panic when the method's safety
//   preconditions hold. `LibcAlloc` is `Copy`, so it has no drop glue.
// - The methods access no currently allocated memory except the block passed to `deallocate`,
//   `grow`, `grow_zeroed`, or `shrink`.
// - The methods are correct: zero-sized layouts get `dangling_slice`, which is aligned, and
//   `malloc_size` (see its safety-usable invariant) picks a request size that is at least
//   `layout.size()` and makes the C result aligned to `layout.align()`; `allocate_zeroed` gets
//   zeroed memory from `calloc`, and `grow_zeroed` is `Allocator`'s provided implementation on top
//   of it; `resize` returns `Err` only before calling `realloc` or when `realloc` returns null,
//   which leaves the old block unchanged; and `deallocate`, `grow`, and `shrink` use only `ptr` and
//   the layout sizes, so they accept every fitting layout.
unsafe impl Allocator for LibcAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }
        let size = malloc_size(layout).ok_or(AllocError)?;

        // SAFETY: `malloc` has no preconditions. If the result is non-null it is a fresh block,
        // valid for reads and writes of `size` bytes, and by `malloc_size`'s safety-usable
        // invariant `size >= layout.size()` and the block is aligned to `layout.align()`.
        let ptr = unsafe { libc::malloc(size) as *mut u8 };
        NonNull::new(ptr).map(|p| NonNull::slice_from_raw_parts(p, layout.size())).ok_or(AllocError)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }
        let size = malloc_size(layout).ok_or(AllocError)?;

        // SAFETY: `calloc` has no preconditions. If the result is non-null it is a fresh block of
        // `size` zeroed bytes, and by `malloc_size`'s safety-usable invariant `size >=
        // layout.size()` and the block is aligned to `layout.align()`.
        let ptr = unsafe { libc::calloc(size, 1) as *mut u8 };

        let non_null = NonNull::new(ptr).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(non_null, layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: `free` requires a pointer from `malloc`/`calloc`/`realloc` that has not been
            // freed. By the `deallocate` contract `ptr` is currently allocated by this allocator
            // and `layout` fits it, so `layout.size() != 0` means the block is non-zero-sized,
            // and every such block is a live C block (see the `LibcAlloc` docs). Afterwards `ptr`
            // is invalid, as `deallocate` promises.
            unsafe { libc::free(ptr.as_ptr().cast::<core::ffi::c_void>()) };
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

impl LibcAlloc {
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
            // SAFETY: `ptr` is currently allocated via this allocator with a fitting `old_layout`,
            // as this function's `# Safety` contract requires and `deallocate` needs. Afterwards
            // any C block at `ptr` is deallocated.
            unsafe { self.deallocate(ptr, old_layout) };
            return Ok(dangling_slice(new_layout));
        }
        if old_layout.size() == 0 {
            // `ptr` is a dangling pointer that owns nothing, so there is nothing to move or free.
            return self.allocate(new_layout);
        }
        let size = malloc_size(new_layout).ok_or(AllocError)?;

        // SAFETY: `libc::realloc` requires a pointer from `malloc`, `calloc`, or `realloc` that has
        // not been freed. By this function's `# Safety` contract, `ptr` is currently allocated via
        // this allocator with a fitting `old_layout`, and `old_layout.size() != 0` here, so it is a
        // live C block (see `LibcAlloc` docs). By C23 7.24.3 and `malloc_size`'s safety-usable
        // invariant, if the result is non-null then `ptr` is deallocated and the new block is valid
        // for reads and writes of `size >= new_layout.size()` bytes, aligned to
        // `new_layout.align()`, with the first `min(old_layout.size(), size)` bytes preserved; if
        // the result is null, `ptr` remains allocated and unchanged.
        let new_ptr = unsafe { libc::realloc(ptr.as_ptr().cast::<c_void>(), size) };
        NonNull::new(new_ptr.cast::<u8>())
            .map(|p| NonNull::slice_from_raw_parts(p, new_layout.size()))
            .ok_or(AllocError)
    }
}

/// A `malloc`-like C allocation function: `void *malloc(size_t size)`.
///
/// See [`CLikeAlloc::new_with_malloc`] for the contract it must meet.
pub type MallocFn = unsafe extern "C" fn(size: size_t) -> *mut c_void;

/// A `calloc`-like C allocation function: `void *calloc(size_t nmemb, size_t size)`.
///
/// See [`CLikeAlloc::new_with_calloc`] for the contract it must meet.
pub type CallocFn = unsafe extern "C" fn(nmemb: size_t, size: size_t) -> *mut c_void;

/// A `free`-like C deallocation function: `void free(void *ptr)`.
///
/// See [`CLikeAlloc::new_with_malloc`] and [`CLikeAlloc::new_with_calloc`] for the contract it
/// must meet.
pub type FreeFn = unsafe extern "C" fn(ptr: *mut c_void);

/// The allocation function of a [`CLikeAlloc`].
#[derive(Clone, Copy, Debug)]
enum AllocFn {
    Malloc(MallocFn),
    Calloc(CallocFn),
}

/// Allocator backed by custom C allocation functions, such as the memory hooks that many C
/// libraries let their users install, for memory whose ownership crosses the FFI boundary.
///
/// Instead of being a zero-sized type that calls the C standard library like [`LibcAlloc`], a
/// `CLikeAlloc` stores pointers to a `malloc`- or `calloc`-like allocation function and a
/// `free`-like deallocation function. Create one with
/// [`new_with_malloc`](Self::new_with_malloc) or [`new_with_calloc`](Self::new_with_calloc).
///
/// As there is no `realloc` function, [`grow`](Allocator::grow) and
/// [`shrink`](Allocator::shrink) allocate a new block, copy the contents, and free the old block.
/// Memory allocated through a `CLikeAlloc` can be deallocated through any copy of it, including on
/// another thread (`CLikeAlloc` is [`Send`] and [`Sync`]).
///
/// Two `CLikeAlloc`s compare equal if they store the same function pointers (via
/// [`core::ptr::fn_addr_eq`]). Equal allocators are equivalent (memory allocated through one may be
/// deallocated through the other), though separately created pointers to the same function may
/// compare unequal; copies of one allocator always compare equal.
///
/// `CLikeAlloc` does not implement [`DropByPtrAllocator`] because freeing memory requires the
/// stored `free` function, so it cannot be used with [`OwnedCBufPtr`](crate::OwnedCBufPtr).
/// Instead, pass it to methods such as [`CBufPtr::as_vec_mut_in`](crate::CBufPtr::as_vec_mut_in).
///
/// Beyond the [`Allocator`] contract, unsafe code may rely on the following:
///
/// - Every non-zero-sized block comes straight from the `malloc` or `calloc` function passed to the
///   constructor, with no header or offset. C code may release it with the corresponding `free`
///   function. Conversely, `deallocate`, `grow`, and `shrink` use only `ptr` and the layout sizes,
///   never `old_layout.align()`, so a block from the underlying allocation function may be passed
///   to them with any layout whose size is within the block.
/// - A block is aligned to its layout even when the layout is smaller than its alignment, which a
///   plain `malloc(size)` does not guarantee (see `malloc_size`).
/// - A layout with alignment above `MALLOC_ALIGN` is refused with `AllocError`, never served
///   misaligned.
/// - Zero-sized layouts never reach the C allocation functions; the dangling pointer they get must
///   not be passed to `free`. `deallocate` handles this itself; code that frees through C must
///   check the size.
///
/// # Example
///
/// ```
/// use core::ffi::c_int;
/// use safer_cffi::{CBufPtr, CLikeAlloc};
///
/// // SAFETY: `libc::calloc` and `libc::free` are C's `calloc` and `free`.
/// let alloc = unsafe { CLikeAlloc::new_with_calloc(libc::calloc, libc::free) };
///
/// let mut items: CBufPtr<c_int, CLikeAlloc> = CBufPtr::null();
/// let mut item_len: c_int = 0;
/// // SAFETY: `items` is null and `item_len` is 0.
/// let mut items_vec = unsafe { items.as_vec_mut_in(&mut item_len, alloc) };
/// items_vec.push_back(1);
/// items_vec.push_back(2);
/// assert_eq!(items_vec.as_slice(), [1, 2]);
/// items_vec.clear();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CLikeAlloc {
    // Safety invariant: `alloc_fn` and `free_fn` meet the `# Safety` contract of
    // `CLikeAlloc::new_with_malloc` if `alloc_fn` is `AllocFn::Malloc`, and of
    // `CLikeAlloc::new_with_calloc` if it is `AllocFn::Calloc`.
    alloc_fn: AllocFn,
    free_fn: FreeFn,
}

impl PartialEq for CLikeAlloc {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let same_alloc_fn = match (self.alloc_fn, other.alloc_fn) {
            (AllocFn::Malloc(a), AllocFn::Malloc(b)) => core::ptr::fn_addr_eq(a, b),
            (AllocFn::Calloc(a), AllocFn::Calloc(b)) => core::ptr::fn_addr_eq(a, b),
            _ => false,
        };
        same_alloc_fn && core::ptr::fn_addr_eq(self.free_fn, other.free_fn)
    }
}

impl Eq for CLikeAlloc {}

impl CLikeAlloc {
    /// Creates an allocator that allocates memory with `malloc` and frees it with `free`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `malloc` and `free` behave like the C standard library
    /// functions of the same name (C23 7.24.3) whenever this allocator or a copy of it calls them.
    /// In particular:
    ///
    /// - For every `n > 0`, `malloc(n)` returns either null or a pointer to a new allocation of at
    ///   least `n` bytes, aligned as C requires of `malloc(n)`: suitably for any object type with a
    ///   fundamental alignment requirement and a size of at most `n` bytes. The allocation stays
    ///   valid for reads and writes, and nothing else accesses it, until it is passed to `free`.
    /// - `free(ptr)`, for a non-null `ptr` that `malloc` returned and that has not been freed yet,
    ///   deallocates that allocation.
    /// - Both functions can be called from any thread, including concurrently, and `free` accepts
    ///   allocations made on other threads. This is required because `CLikeAlloc` is `Send` and
    ///   `Sync`.
    /// - Neither function unwinds.
    #[inline]
    #[must_use]
    pub const unsafe fn new_with_malloc(malloc: MallocFn, free: FreeFn) -> Self {
        // The safety invariant of `CLikeAlloc` for `AllocFn::Malloc` is this function's
        // `# Safety` contract, which the caller upholds.
        Self { alloc_fn: AllocFn::Malloc(malloc), free_fn: free }
    }

    /// Creates an allocator that allocates memory with `calloc` and frees it with `free`.
    ///
    /// As there is no `malloc` function, all allocations are zero-initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `calloc` and `free` behave like the C standard library
    /// functions of the same name (C23 7.24.3) whenever this allocator or a copy of it calls them.
    /// In particular:
    ///
    /// - For every `n > 0`, `calloc(n, 1)` returns either null or a pointer to a new allocation of
    ///   at least `n` zero-initialized bytes, aligned as C requires of `calloc(n, 1)`: suitably for
    ///   any object type with a fundamental alignment requirement and a size of at most `n` bytes.
    ///   The allocation stays valid for reads and writes, and nothing else accesses it, until it is
    ///   passed to `free`.
    /// - `free(ptr)`, for a non-null `ptr` that `calloc` returned and that has not been freed yet,
    ///   deallocates that allocation.
    /// - Both functions can be called from any thread, including concurrently, and `free` accepts
    ///   allocations made on other threads. This is required because `CLikeAlloc` is `Send` and
    ///   `Sync`.
    /// - Neither function unwinds.
    #[inline]
    #[must_use]
    pub const unsafe fn new_with_calloc(calloc: CallocFn, free: FreeFn) -> Self {
        // The safety invariant of `CLikeAlloc` for `AllocFn::Calloc` is this function's
        // `# Safety` contract, which the caller upholds.
        Self { alloc_fn: AllocFn::Calloc(calloc), free_fn: free }
    }
}

// SAFETY: The `Allocator` contract asks for the following. `CLikeAlloc`s that compare equal are
// equivalent (see the `CLikeAlloc` docs), which is consistent because they store the same
// functions and the methods read nothing else from `self`.
//
// - Currently allocated blocks point to valid memory until they are invalidated: non-zero-sized
//   blocks come from `self.alloc_fn` and, by `CLikeAlloc`'s field safety invariant, remain valid
//   and unaccessed by anything else until passed to `self.free_fn`, which only `deallocate` does
//   and which invalidates the block; zero-sized blocks are dangling pointers that are never
//   dereferenced.
// - Returned blocks satisfy the allocation invariants of `core::ptr` and are wholly disjoint: by
//   the field safety invariant, which requires C23 7.24.3 behavior, a non-zero-sized block is a
//   prefix of a new allocation from `self.alloc_fn`, disjoint from any other object; a zero-sized
//   block contains no addresses.
// - No method unwinds: they call only `self.alloc_fn` and `self.free_fn`, which by the field safety
//   invariant do not unwind and may be called from any thread, plus code that does not panic when
//   the method's safety preconditions hold. `CLikeAlloc` is `Copy`, so it has no drop glue.
// - The methods access no currently allocated memory except the block passed to `deallocate`,
//   `grow`, `grow_zeroed`, or `shrink`, and the block that `allocate_zeroed` zeroes before
//   returning it.
// - The methods are correct: zero-sized layouts get `dangling_slice`, which is aligned, and by
//   `malloc_size`'s safety-usable invariant and the field safety invariant, every non-zero-sized
//   block has at least `layout.size()` bytes and is aligned to `layout.align()`; `allocate_zeroed`
//   returns zeroed blocks, from `calloc` or by zeroing a `malloc` block; and `deallocate` checks
//   only `layout.size() != 0`, while `grow`, `grow_zeroed`, and `shrink` use `Allocator`'s provided
//   implementations, which build on `allocate`, `allocate_zeroed`, `deallocate`, and
//   `copy_nonoverlapping` on `*mut u8`, so all of them accept every fitting layout and never
//   inspect `old_layout.align()`.
unsafe impl Allocator for CLikeAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling_slice(layout));
        }
        let size = malloc_size(layout).ok_or(AllocError)?;

        let ptr = match self.alloc_fn {
            AllocFn::Malloc(malloc) => {
                // SAFETY: By `CLikeAlloc`'s safety invariant, `malloc` meets the `# Safety`
                // contract of `CLikeAlloc::new_with_malloc`, whose only call precondition is
                // `n > 0`. `layout.size() != 0` here and, by `malloc_size`'s safety-usable
                // invariant, `size >= layout.size()`, so `size > 0`.
                unsafe { malloc(size) }
            }
            AllocFn::Calloc(calloc) => {
                // SAFETY: By `CLikeAlloc`'s safety invariant, `calloc` meets the `# Safety`
                // contract of `CLikeAlloc::new_with_calloc`, whose only call precondition for
                // `calloc(n, 1)` is `n > 0`. `layout.size() != 0` here and, by `malloc_size`'s
                // safety-usable invariant, `size >= layout.size()`, so `size > 0`.
                unsafe { calloc(size, 1) }
            }
        };
        NonNull::new(ptr.cast::<u8>())
            .map(|p| NonNull::slice_from_raw_parts(p, layout.size()))
            .ok_or(AllocError)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match self.alloc_fn {
            // `AllocFn::Calloc` already zero-initializes in `allocate`.
            AllocFn::Calloc(_) => self.allocate(layout),
            AllocFn::Malloc(_) => {
                let slice = self.allocate(layout)?;
                if !slice.is_empty() {
                    // SAFETY: `write_bytes` requires a pointer valid for writes of `count` bytes
                    // and aligned for `u8` (alignment 1, which every pointer satisfies). `slice`
                    // was just returned by `self.allocate(layout)`, which meets the `Allocator`
                    // contract (see the SAFETY comment on this impl), so it is a currently
                    // allocated block of `slice.len()` bytes that is valid for writes and not yet
                    // shared. Afterwards all `slice.len()` bytes are zero-initialized.
                    unsafe { slice.cast::<u8>().as_ptr().write_bytes(0, slice.len()) };
                }
                Ok(slice)
            }
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: By `CLikeAlloc`'s safety invariant, `self.free_fn` meets the `# Safety`
            // contract of `CLikeAlloc::new_with_malloc` or `CLikeAlloc::new_with_calloc`, requiring
            // a non-null pointer returned by `self.alloc_fn` that has not yet been freed. By
            // `deallocate`'s `# Safety` contract, `ptr` is currently allocated via this allocator
            // (that is, via an equivalent one, which stores the same functions) and `layout` fits
            // it, so `layout.size() != 0` means `ptr` came from `self.alloc_fn` and has not been
            // freed since (see `CLikeAlloc` docs). Afterwards `ptr` is deallocated, as
            // `deallocate` promises.
            unsafe { (self.free_fn)(ptr.as_ptr().cast::<c_void>()) };
        }
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
        // SAFETY: `libc::free` requires a pointer from `malloc`, `calloc`, or `realloc` that has
        // not been freed. By `deallocate_by_ptr`'s `# Safety` contract, `ptr` is currently
        // allocated via `LibcAlloc` with a non-zero size, and every such block is a live C block
        // from `libc::malloc`, `libc::calloc`, or `libc::realloc` (see `LibcAlloc` docs). The
        // contract also forbids any active aliases or later access through `ptr`.
        unsafe { libc::free(ptr.as_ptr().cast::<c_void>()) };
    }
}

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
        assert_traits::<CLikeAlloc>();
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
        static FAIL_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    }

    fn reset_calls() {
        MALLOC_CALLS.set(0);
        CALLOC_CALLS.set(0);
        FREE_CALLS.set(0);
        LAST_MALLOC_SIZE.set(0);
        LAST_CALLOC_ARGS.set((0, 0));
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

    /// `libc::free` under another name, for an allocator that differs only in its `free`.
    unsafe extern "C" fn other_free(ptr: *mut c_void) {
        unsafe { libc::free(ptr) };
    }

    fn counting_malloc_alloc() -> CLikeAlloc {
        // SAFETY: `counting_malloc` and `counting_free` wrap `libc::malloc` and `libc::free`.
        unsafe { CLikeAlloc::new_with_malloc(counting_malloc, counting_free) }
    }

    fn counting_calloc_alloc() -> CLikeAlloc {
        // SAFETY: `counting_calloc` and `counting_free` wrap `libc::calloc` and `libc::free`.
        unsafe { CLikeAlloc::new_with_calloc(counting_calloc, counting_free) }
    }

    #[gtest]
    fn c_like_alloc_with_libc_functions() {
        // The constructors are `const`, so allocators can be stored in statics.
        // SAFETY: These are the C standard library's allocation functions.
        static LIBC_MALLOC: CLikeAlloc =
            unsafe { CLikeAlloc::new_with_malloc(libc::malloc, libc::free) };
        // SAFETY: These are the C standard library's allocation functions.
        static LIBC_CALLOC: CLikeAlloc =
            unsafe { CLikeAlloc::new_with_calloc(libc::calloc, libc::free) };

        for alloc in [LIBC_MALLOC, LIBC_CALLOC] {
            let b = CBox::new_in([1u64, 2, 3], alloc);
            assert_that!(*b, eq([1, 2, 3]));
        }
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

        // Without `realloc`, growing allocates a new block, copies the contents, and frees the old
        // block.
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
        let layout = Layout::array::<u32>(4).unwrap();
        let larger = Layout::array::<u32>(8).unwrap();
        for alloc in [counting_malloc_alloc(), counting_calloc_alloc()] {
            reset_calls();
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
    }

    #[gtest]
    fn c_like_alloc_with_c_buf_ptr() {
        reset_calls();
        let mut ptr = CBufPtr::<i32, CLikeAlloc>::null();
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

        let mut vec: CVec<i32, CLikeAlloc> = CVec::new_in(alloc);
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
        assert_that!(counting_calloc_alloc(), not(eq(alloc)));
        // SAFETY: `counting_malloc` and `other_free` wrap `libc::malloc` and `libc::free`.
        let other_free_alloc = unsafe { CLikeAlloc::new_with_malloc(counting_malloc, other_free) };
        assert_that!(other_free_alloc, not(eq(alloc)));
    }
}

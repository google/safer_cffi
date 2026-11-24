// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A library for creating safer C APIs from Rust.
//!
//! This crate provides a tracker that manages the lifecycle of Rust objects, allowing
//! them to be passed to C as opaque handles and safely reclaimed later.
//!
//! ## Tracking Objects
//!
//! [`OpaqueTracker`] uses synthetic IDs to track objects ([`Handle<T>`]). Use it to provide opaque
//! pointers to C code.
//!
//! A handle is not an address, so C code cannot read the fields of a tracked object. If the C
//! header defines the fields of a struct, pass the struct itself instead, as `Option<&mut T>` or
//! `Option<Box<T>>`.
//!
//! ## Shared Borrows
//!
//! By default, a tracker holds each object in a `Box<T>` and only hands out exclusive borrows
//! ([`OpaqueTracker::borrow_mut`]): borrowing an object fails while another borrow of it is alive.
//!
//! To let read-only entry points use the same object at the same time, give the tracker `Arc<T>`
//! as its storage type (`OpaqueTracker<T, Arc<T>>`), and borrow with
//! [`OpaqueTracker::borrow`] wherever no mutable access is needed.
//!
//! Trackers do not hold `Arc<T>` by default: such a tracker can only be a `static` if `T` is
//! `Sync`, and its exclusive borrows are slightly more expensive.
//!
//! ## C Slices and Vectors
//!
//! Many C structs contain `(*mut T, L)` field pairs representing dynamically-sized
//! arrays (where `L` is an integer length type such as `c_int` or `usize`).
//! [`CBufPtr`], [`OwnedCBufPtr`], and [`CVecRefMut`] provide safe handles over these pairs:
//!
//! - **[`CBufPtr`]**: A `#[repr(transparent)]` wrapper around `*mut T` for use in `#[repr(C)]`
//!   struct definitions. Encodes the allocator type (defaulting to [`LibcAlloc`]). Provides
//!   [`with_len`](CBufPtr::with_len) to get a `&[T]` slice, [`with_len_mut`](CBufPtr::with_len_mut)
//!   to get a mutable `&mut [T]` slice, [`as_vec_mut`](CBufPtr::as_vec_mut) to create a mutable
//!   `CVecRefMut` handle, and [`clone_and_leak`](CBufPtr::clone_and_leak) to clone a Rust slice
//!   into a C-allocated buffer.
//!
//! - **[`OwnedCBufPtr`]**: A `#[repr(transparent)]` wrapper around [`CBufPtr`] for `Copy` types
//!   that automatically frees the allocated buffer on drop. Dereferences to `CBufPtr` to provide
//!   all its slice and vector accessor methods. Use
//!   [`clone_from_slice`](OwnedCBufPtr::clone_from_slice) to clone a Rust slice into a new owned
//!   buffer, e.g. to implement `Clone` for the containing struct.
//!
//! - **[`CVecRefMut`]** (growable vector handle): Provides mutable slice access via
//!   [`DerefMut`](core::ops::DerefMut), plus [`push_back`](CVecRefMut::push_back),
//!   [`try_push_back`](CVecRefMut::try_push_back), [`clear`](CVecRefMut::clear), and
//!   [`swap`](CVecRefMut::swap) for array mutation.
//!
//! All vector operations default to the **C allocator** (`malloc`/`free`) for allocations,
//! ensuring compatibility with memory managed across the FFI boundary, and support custom
//! [`Allocator`] implementations via `as_vec_mut_in`, `clone_and_leak_in`, and
//! `clone_from_slice_in`. To allocate with custom C allocation functions, such as memory hooks
//! installed by C callers, use [`CLikeAlloc`] instead of the default, zero-sized [`LibcAlloc`].

pub(crate) mod alloc;
pub(crate) mod c_buf;
pub(crate) mod c_str;
pub(crate) mod c_vec;
pub(crate) mod errors;
pub(crate) mod handle;
#[cfg(test)]
pub(crate) mod testing;
pub(crate) mod tracker;

pub use alloc::{
    AllocFns, CBox, CLikeAlloc, CVec, CallocFn, CustomFns, DropByPtrAllocator, EquivalentAlloc,
    FreeFn, LibcAlloc, LibcFns, MallocFn,
};
pub use c_buf::{CBufLen, CBufPtr, OwnedCBufPtr};
pub use c_str::CStrRef;
pub use c_vec::CVecRefMut;
pub use core::alloc::Allocator;
pub use errors::{AllocError, TrackerError};
pub use handle::Handle;
#[allow(deprecated)]
pub use tracker::Tracker;
pub use tracker::{ArcTrackerStorage, OpaqueTracker, TrackedRef, TrackedRefMut, TrackerStorage};

// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A library for creating safer C APIs from Rust.
//!
//! This crate provides trackers that manage the lifecycle of Rust objects, allowing
//! them to be passed to C as handles or pointers and safely reclaimed later.
//!
//! ## Choosing a Tracker
//!
//! There are two tracker variants available:
//!
//! - **Opaque Tracker** (`OpaqueTracker`): Uses synthetic IDs to track objects (`Handle<T>`).
//!   Use this to provide opaque pointers to C code. This is faster and safer than the
//!   raw tracker.
//!
//! - **Raw Tracker** (`RawTracker`): Uses raw memory addresses (`*mut T`) as keys.
//!   Use this only if you need to interoperate with existing C code that requires the actual
//!   pointer value, e.g. for field access without helpers.
//!
//!   The raw tracker comes with two caveats:
//!
//!   - The raw tracker cannot be used if the API allows for objects to be created on the C
//!     side. All objects must be created from the Rust side.
//!   - It does not prevent ABA problems. That is, if an object is deallocated and a new object
//!     is allocated in the same memory location, pointers to the old object will now silently
//!     point to the new object.
//!
//! ## C Slices and Vectors
//!
//! Many C structs contain `(*mut T, L)` field pairs representing dynamically-sized
//! arrays (where `L` is an integer length type such as `c_int` or `usize`).
//! [`CSlicePtr`] and [`CVecRefMut`] provide safe handles over these pairs:
//!
//! - **[`CSlicePtr`]**: A `#[repr(transparent)]` wrapper around `*mut T` for use in
//!   `#[repr(C)]` struct definitions. Encodes the allocator type (defaulting to [`LibcAlloc`]).
//!   Provides [`with_len`](CSlicePtr::with_len) to get a `&[T]` slice,
//!   [`with_len_mut`](CSlicePtr::with_len_mut) to get a mutable `&mut [T]` slice,
//!   [`with_len_vec_mut`](CSlicePtr::with_len_vec_mut) to create a mutable `CVecRefMut`
//!   handle, and [`clone_and_leak`](CSlicePtr::clone_and_leak) to clone a Rust slice into a C-allocated buffer.
//!
//! - **[`CVecRefMut`]** (growable vector handle): Provides mutable slice access via
//!   [`DerefMut`](core::ops::DerefMut), plus [`push_back`](CVecRefMut::push_back),
//!   [`try_push_back`](CVecRefMut::try_push_back), [`clear`](CVecRefMut::clear),
//!   [`replace`](CVecRefMut::replace), and [`swap`](CVecRefMut::swap) for array mutation.
//!
//! All vector operations default to the **C allocator** (`malloc`/`free`) for allocations,
//! ensuring compatibility with memory managed across the FFI boundary, and support custom
//! [`Allocator`] implementations via `with_len_vec_mut_in` and `clone_and_leak_in`.

pub(crate) mod alloc;
pub(crate) mod c_slice;
pub(crate) mod c_str;
pub(crate) mod c_vec;
pub(crate) mod errors;
pub(crate) mod handle;
pub(crate) mod opaque;
pub(crate) mod raw;
#[cfg(test)]
pub(crate) mod testing;
pub(crate) mod tracker;

pub use alloc::{CBox, LibcAlloc};
pub use allocator_api2::alloc::Allocator;
pub use c_slice::{CSliceLen, CSlicePtr};
pub use c_str::CStrRef;
pub use c_vec::CVecRefMut;
pub use errors::{AllocError, TrackerError};
pub use handle::Handle;
pub use opaque::OpaqueTracker;
pub use raw::RawTracker;
pub use tracker::{Tracked, Tracker};

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
//! ## C Slices
//!
//! Many C structs contain `(*mut T, c_int)` field pairs representing dynamically-sized
//! arrays. [`CSlicePtr`] and [`CSliceRefMut`] provide safe handles over these pairs:
//!
//! - **[`CSlicePtr`]**: A `#[repr(transparent)]` wrapper around `*mut T` for use in
//!   `#[repr(C)]` struct definitions. Provides [`with_len`](CSlicePtr::with_len) to get
//!   a `&[T]` slice, [`with_len_mut`](CSlicePtr::with_len_mut) to create a mutable handle,
//!   and [`clone_and_leak`](CSlicePtr::clone_and_leak) to clone a Rust slice into a C-allocated buffer.
//!
//! - **[`CSliceRefMut`]** (exclusive): Provides mutable slice access via
//!   [`DerefMut`](core::ops::DerefMut), plus [`add`](CSliceRefMut::add) and
//!   [`clear`](CSliceRefMut::clear) for array mutation.
//!
//! All use the **C allocator** (`malloc`/`free`) for allocations,
//! ensuring compatibility with memory managed across the FFI boundary.

pub(crate) mod c_slice;
pub(crate) mod c_str;
pub(crate) mod errors;
pub(crate) mod handle;
pub(crate) mod opaque;
pub(crate) mod raw;
pub(crate) mod tracker;

pub use c_slice::{CSlicePtr, CSliceRefMut};
pub use c_str::CStrRef;
pub use errors::{AllocError, TrackerError};
pub use handle::Handle;
pub use opaque::OpaqueTracker;
pub use raw::RawTracker;
pub use tracker::{Tracked, Tracker};

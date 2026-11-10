// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Like `custom_allocator.rs`, but the allocation functions also get a context pointer from the C
//! caller, like `opaque` in zlib's `z_stream`. It typically points to the caller's own state, such
//! as a memory pool; the compressor only passes it on.

use safer_cffi::{AllocFns, CBox, CLikeAlloc};
use std::ffi::c_void;

// ---------------------- Compressor C Struct ----------------------

/// A compressor that allocates its item, like any other state, with `self.allocator()`.
struct Compressor {
    /// The box stores a copy of the allocator, so dropping it frees its memory with `free_fn`.
    item: Option<CBox<[u8; 5], ExampleAlloc>>,
    // Safety invariant: `alloc_fn(opaque, size)` and `free_fn(opaque, ptr)` behave like C's
    // `malloc(size)` and `free(ptr)`, including alignment.
    alloc_fn: AllocFn,
    free_fn: FreeFn,
    opaque: *mut c_void,
}

/// `void* (*alloc_fn)(void* opaque, size_t size)`
type AllocFn = unsafe extern "C" fn(opaque: *mut c_void, size: usize) -> *mut c_void;
/// `void (*free_fn)(void* opaque, void* ptr)`
type FreeFn = unsafe extern "C" fn(opaque: *mut c_void, ptr: *mut c_void);
/// The allocator of a [`Compressor`], which calls its `alloc_fn` and `free_fn` with its `opaque`.
type ExampleAlloc = CLikeAlloc<CompressorAllocFns>;

impl Compressor {
    /// Creates a compressor without an item.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `alloc_fn(opaque, size)` and `free_fn(opaque, ptr)` behave like
    /// C's `malloc(size)` and `free(ptr)`, including alignment, for as long as the compressor or
    /// any memory it allocated exists.
    unsafe fn new(alloc_fn: AllocFn, free_fn: FreeFn, opaque: *mut c_void) -> Self {
        Self { item: None, alloc_fn, free_fn, opaque }
    }
}

// ---------------------- Allocator Implementation ----------------------

impl Compressor {
    /// Returns an allocator that calls `alloc_fn` and `free_fn` with `opaque`.
    fn allocator(&self) -> ExampleAlloc {
        // SAFETY: `CompressorAllocFns` has the same field invariant as `Compressor`.
        ExampleAlloc::new(CompressorAllocFns {
            alloc_fn: self.alloc_fn,
            free_fn: self.free_fn,
            opaque: self.opaque,
        })
    }
}

/// The allocation functions of a compressor, together with the context pointer to pass to them.
///
/// This needs to be a separate struct from `Compressor` to be copyable.
#[derive(Clone, Copy)]
struct CompressorAllocFns {
    // Safety invariant: `alloc_fn(opaque, size)` and `free_fn(opaque, ptr)` behave like C's
    // `malloc(size)` and `free(ptr)`, including alignment.
    alloc_fn: AllocFn,
    free_fn: FreeFn,
    opaque: *mut c_void,
}

// SAFETY: By the field invariant, `malloc` and `free` behave like C's `malloc` and `free`, which
// also means that they do not unwind. Copies hold the same functions and context pointer, so they
// accept each other's allocations.
unsafe impl AllocFns for CompressorAllocFns {
    fn malloc(&self, size: usize) -> *mut c_void {
        // SAFETY: By the field invariant, `alloc_fn(opaque, size)` can be called like
        // `malloc(size)`, which has no preconditions.
        unsafe { (self.alloc_fn)(self.opaque, size) }
    }

    unsafe fn free(&self, ptr: *mut c_void) {
        // SAFETY: By the field invariant, `free_fn(opaque, ptr)` can be called like `free(ptr)`,
        // and the caller ensures that `ptr` may be passed to it.
        unsafe { (self.free_fn)(self.opaque, ptr) }
    }
}

// ---------------------- Toy Example ----------------------

#[cfg(test)]
#[allow(missing_safety_doc)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    unsafe extern "C" fn log_malloc(opaque: *mut c_void, size: usize) -> *mut c_void {
        println!("Allocating {size} bytes with context {opaque:p}");
        unsafe { libc::malloc(size) }
    }

    unsafe extern "C" fn log_free(opaque: *mut c_void, ptr: *mut c_void) {
        println!("Freeing {ptr:p} with context {opaque:p}");
        unsafe { libc::free(ptr) }
    }

    #[gtest]
    fn allocates_item_with_alloc_fn() {
        let mut context = 0;
        let mut compressor =
            unsafe { Compressor::new(log_malloc, log_free, (&raw mut context).cast()) };

        // Calls `log_malloc` for the box. Dropping the compressor calls `log_free`.
        compressor.item = Some(CBox::new_in(*b"hello", compressor.allocator()));

        expect_that!(compressor.item.as_deref(), some(eq(b"hello")));
    }
}

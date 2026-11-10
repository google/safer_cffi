// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A compressor that allocates memory with allocation functions from its C caller, like the memory
//! hooks that many C libraries let their users install. For simplicity, the test passes the C
//! standard library's `malloc` and `free`.

use googletest::prelude::*;
use safer_cffi::{CBox, CLikeAlloc, CustomFns, FreeFn, MallocFn};

/// The allocator of a [`Compressor`], which calls its `alloc_fn` and `free_fn`.
type ExampleAlloc = CLikeAlloc<CustomFns>;

/// A compressor that allocates its item, like any other state, with `self.allocator()`.
struct Compressor {
    /// The box stores a copy of the allocator, so dropping it frees its memory with `free_fn`.
    item: Option<CBox<[u8; 5], ExampleAlloc>>,
    // Safety invariant: `alloc_fn` and `free_fn` behave like C's `malloc` and `free`, including
    // alignment and thread safety.
    alloc_fn: MallocFn,
    free_fn: FreeFn,
}

impl Compressor {
    /// Creates a compressor without an item.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `alloc_fn` and `free_fn` behave like C's `malloc` and `free`,
    /// including alignment and thread safety (see [`CLikeAlloc::new_with_malloc`]).
    unsafe fn new(alloc_fn: MallocFn, free_fn: FreeFn) -> Self {
        Self { item: None, alloc_fn, free_fn }
    }

    /// Returns an allocator that allocates memory with `alloc_fn` and frees it with `free_fn`.
    fn allocator(&self) -> ExampleAlloc {
        // SAFETY: `new_with_malloc` requires `malloc` and `free` to behave like the C standard
        // library functions of the same name, including alignment and thread safety. By the field
        // invariant, `self.alloc_fn` and `self.free_fn` do.
        unsafe { ExampleAlloc::new_with_malloc(self.alloc_fn, self.free_fn) }
    }
}

#[gtest]
fn allocates_item_with_alloc_fn() {
    // SAFETY: `Compressor::new` requires `alloc_fn` and `free_fn` to behave like C's `malloc` and
    // `free`, including alignment and thread safety. `libc::malloc` and `libc::free` are the C
    // standard library's `malloc` and `free`.
    let mut compressor = unsafe { Compressor::new(libc::malloc, libc::free) };

    // Allocates the box with `libc::malloc`; dropping the compressor frees it with `libc::free`.
    compressor.item = Some(CBox::new_in(*b"hello", compressor.allocator()));

    expect_that!(compressor.item.as_deref(), some(eq(b"hello")));
}

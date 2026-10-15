// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use safer_cffi::{CVecRefMut, OwnedCBufPtr};

#[repr(C)]
pub struct IntArray {
    // Safety invariant: Either `items` is null and `item_len == 0`, or `items` points to a live,
    // uniquely owned `LibcAlloc` allocation of `Layout::array::<i32>((item_len as usize).max(1))`
    // with `0 <= item_len` initialized `i32` elements (`(item_len as usize) * size_of::<i32>() <=
    // isize::MAX as usize`). Fields are `pub` only for `c_library_from_rust_signatures_test`'s
    // `offset_of!` check until RFC 3458 `unsafe` fields are stable; the crate target has private
    // visibility.
    pub items: OwnedCBufPtr<i32>,
    // Safety invariant: `item_len >= 0` equals the exact number of initialized `i32` elements in
    // the allocation owned by `items` (and is `0` when `items` is null).
    pub item_len: i32,
}

impl IntArray {
    pub fn items(&self) -> &[i32] {
        // SAFETY: By the field safety invariant of `IntArray`, `self.item_len >= 0`, and if
        // `self.items` is non-null and `self.item_len > 0`, `self.items` points to `self.item_len`
        // consecutive initialized, aligned `i32` elements in a single live allocation that is not
        // mutated for the duration of the `&self` borrow.
        unsafe { self.items.with_len(self.item_len) }
    }

    pub fn items_mut(&mut self) -> &mut [i32] {
        // SAFETY: By the field safety invariant of `IntArray`, `self.item_len >= 0`, and if
        // `self.items` is non-null and `self.item_len > 0`, `self.items` points to `self.item_len`
        // consecutive initialized, aligned `i32` elements in a single live allocation exclusively
        // borrowed via `&mut self`.
        unsafe { self.items.with_len_mut(self.item_len) }
    }

    pub fn items_vec_mut(&mut self) -> CVecRefMut<'_, i32, i32> {
        // SAFETY: By the field safety invariant of `IntArray`, `self.item_len >= 0`, and either
        // `self.items` is null with `self.item_len == 0` or `self.items` points to a uniquely owned
        // `LibcAlloc` allocation of `Layout::array::<i32>((self.item_len as usize).max(1))` with
        // `self.item_len` initialized `i32` elements. `CVecRefMut` preserves this coupled invariant
        // across all mutations (`push_back`, `clear`, `swap`).
        unsafe { self.items.as_vec_mut(&mut self.item_len) }
    }
}

impl Clone for IntArray {
    fn clone(&self) -> Self {
        // Safety invariant maintained: `OwnedCBufPtr::clone_from_slice` returns a null pointer when
        // `self.item_len == 0` or a fresh `LibcAlloc` allocation of `self.item_len` cloned `i32`
        // elements.
        Self { items: OwnedCBufPtr::clone_from_slice(self.items()), item_len: self.item_len }
    }
}

// Example FFI functions.
//
// We focus on `CBufPtr` here, ideally you want to manage `IntArray` with a tracker.

// SAFETY: `create_array` is the unique C ABI symbol declared in `c_buf_ptr.h` and does not collide
// with any other symbol in this binary.
#[unsafe(no_mangle)]
pub extern "C" fn create_array() -> Option<Box<IntArray>> {
    Some(Box::new(IntArray { items: OwnedCBufPtr::null(), item_len: 0 }))
}

// SAFETY: `free_array` is the unique C ABI symbol declared in `c_buf_ptr.h` and does not collide
// with any other symbol in this binary.
#[unsafe(no_mangle)]
pub extern "C" fn free_array(array: Option<Box<IntArray>>) {
    drop(array);
}

// SAFETY: `append_to_array` is the unique C ABI symbol declared in `c_buf_ptr.h` and does not
// collide with any other symbol in this binary.
#[unsafe(no_mangle)]
pub extern "C" fn append_to_array(array: Option<&mut IntArray>, item: i32) {
    if let Some(arr) = array {
        arr.items_vec_mut().push_back(item);
    }
}

// SAFETY: `sum_array` is the unique C ABI symbol declared in `c_buf_ptr.h` and does not collide
// with any other symbol in this binary.
#[unsafe(no_mangle)]
pub extern "C" fn sum_array(array: Option<&IntArray>) -> i32 {
    array.map(|arr| arr.items().iter().sum()).unwrap_or(0)
}

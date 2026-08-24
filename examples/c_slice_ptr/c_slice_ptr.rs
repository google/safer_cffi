// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use safer_cffi::{CSlicePtr, CSliceRefMut};

#[repr(C)]
pub struct IntArray {
    // Safety invariant: the length of this array is `item_len`.
    pub items: CSlicePtr<i32>,
    pub item_len: i32,
}

impl IntArray {
    pub fn items(&self) -> &[i32] {
        // SAFETY: the length of `items` is `item_len`.
        unsafe { self.items.with_len(self.item_len) }
    }

    pub fn items_mut(&mut self) -> CSliceRefMut<'_, i32, i32> {
        // SAFETY: the length of `items` is `item_len`.
        unsafe { self.items.with_len_mut(&mut self.item_len) }
    }
}

impl Drop for IntArray {
    fn drop(&mut self) {
        self.items_mut().clear();
    }
}

// Example FFI functions.
//
// We focus on `CSlicePtr` here, ideally you want to manage `IntArray` with a tracker.

#[unsafe(no_mangle)]
pub extern "C" fn create_array() -> Option<Box<IntArray>> {
    Some(Box::new(IntArray { items: CSlicePtr::null(), item_len: 0 }))
}

#[unsafe(no_mangle)]
pub extern "C" fn free_array(array: Option<Box<IntArray>>) {
    drop(array);
}

#[unsafe(no_mangle)]
pub extern "C" fn append_to_array(array: Option<&mut IntArray>, item: i32) {
    if let Some(arr) = array {
        arr.items_mut().add(item);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_array(array: Option<&IntArray>) -> i32 {
    array.map(|arr| arr.items().iter().sum()).unwrap_or(0)
}

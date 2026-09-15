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
    // Safety invariant: the length of this array is `item_len`, and the capacity is `item_cap`.
    /// Do not access this field directly, use `items_mut` instead.
    /// TODO: Mark as `unsafe` once https://github.com/rust-lang/rfcs/blob/master/text/3458-unsafe-fields.md is stable.
    pub items: OwnedCBufPtr<u8>,
    /// Do not access this field directly, use `items_mut` instead.
    /// TODO: Mark as `unsafe` once https://github.com/rust-lang/rfcs/blob/master/text/3458-unsafe-fields.md is stable.
    pub item_len: i32,
    /// Do not access this field directly, use `items_mut` instead.
    /// TODO: Mark as `unsafe` once https://github.com/rust-lang/rfcs/blob/master/text/3458-unsafe-fields.md is stable.
    pub item_cap: i32,
}

impl IntArray {
    pub fn items_mut(&mut self) -> CVecRefMut<'_, u8, i32> {
        // SAFETY: the length of `items` is `item_len`, and the capacity is `item_cap`.
        unsafe { self.items.as_vec_mut_with_cap(&mut self.item_len, &mut self.item_cap) }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn append_to_array(array: Option<&mut IntArray>, item: u8) {
    if let Some(arr) = array {
        arr.items_mut().push_back(item);
    }
}

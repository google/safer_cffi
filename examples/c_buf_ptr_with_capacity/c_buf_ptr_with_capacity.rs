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
    // Safety invariant: Either `items` is null and `item_len == 0 && item_cap == 0`, or `items`
    // points to a live, uniquely owned `LibcAlloc` allocation of
    // `Layout::array::<u8>((item_cap as usize).max(1))` (`0 <= item_len <= item_cap <=
    // isize::MAX`) with the first `item_len` elements initialized. (`pub` only for
    // `c_library_from_rust_signatures_test`'s `offset_of!` check; the crate target has private
    // visibility.)
    /// Do not access this field directly, use `items_mut` instead.
    /// TODO: Mark as `unsafe` once https://github.com/rust-lang/rfcs/blob/master/text/3458-unsafe-fields.md is stable.
    pub items: OwnedCBufPtr<u8>,
    // Safety invariant: `0 <= item_len <= item_cap`, representing the number of initialized
    // leading `u8` elements in `items`.
    /// Do not access this field directly, use `items_mut` instead.
    /// TODO: Mark as `unsafe` once https://github.com/rust-lang/rfcs/blob/master/text/3458-unsafe-fields.md is stable.
    pub item_len: i32,
    // Safety invariant: `item_cap >= item_len >= 0` equals the allocated capacity (in `u8`
    // elements) of the `LibcAlloc` buffer owned by `items` (and is `0` when `items` is null).
    /// Do not access this field directly, use `items_mut` instead.
    /// TODO: Mark as `unsafe` once https://github.com/rust-lang/rfcs/blob/master/text/3458-unsafe-fields.md is stable.
    pub item_cap: i32,
}

impl IntArray {
    pub fn items_mut(&mut self) -> CVecRefMut<'_, u8, i32> {
        // SAFETY: By the field safety invariants of `IntArray`, `0 <= self.item_len <=
        // self.item_cap`, and either `self.items` is null with `self.item_cap == 0` or `self.items`
        // points to a uniquely owned `LibcAlloc` allocation of
        // `Layout::array::<u8>((self.item_cap as usize).max(1))` with `self.item_len` initialized
        // `u8` elements. `CVecRefMut` preserves this invariant across all mutations.
        unsafe { self.items.as_vec_mut_with_cap(&mut self.item_len, &mut self.item_cap) }
    }
}

// SAFETY: `append_to_array` is the unique C ABI symbol declared in `c_buf_ptr_with_capacity.h` and
// does not collide with any other symbol in this binary.
#[unsafe(no_mangle)]
pub extern "C" fn append_to_array(array: Option<&mut IntArray>, item: u8) {
    if let Some(arr) = array {
        arr.items_mut().push_back(item);
    }
}

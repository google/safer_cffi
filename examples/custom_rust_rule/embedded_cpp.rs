// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use inline_cpp_macro::{global_cpp, inline_cpp};

global_cpp! {
    #include <cstdint>

    #include "third_party/absl/numeric/bits.h"
}

#[unsafe(no_mangle)]
pub extern "C" fn bit_ceil(x: u32) -> u32 {
    let cpp_bit_ceil = inline_cpp! {
        (uint32_t x) -> uint32_t {
          return absl::bit_ceil(x);
        }
    };
    cpp_bit_ceil(x)
}

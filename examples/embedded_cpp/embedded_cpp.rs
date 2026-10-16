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

    #include "security/ise_memory_safety/safer_cffi/examples/embedded_cpp/math_helper.h"
}

/// Computes the `n`-th Fibonacci number in Rust.
#[unsafe(no_mangle)]
pub extern "C" fn fibonacci(n: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 1..n {
        let next = a + b;
        a = b;
        b = next;
    }
    b
}

/// Checks whether `x` is a Fibonacci number using Binet's identity in Rust and
/// an embedded C++ helper for the perfect-square check.
#[unsafe(no_mangle)]
pub extern "C" fn is_fibonacci(x: i32) -> bool {
    if x < 0 {
        return false;
    }
    let Some(five_x2) = x.checked_mul(x).and_then(|x2| x2.checked_mul(5)) else {
        return false;
    };
    let Some(plus_four) = five_x2.checked_add(4) else {
        return false;
    };
    let minus_four = five_x2 - 4;
    is_perfect_square(plus_four) || is_perfect_square(minus_four)
}

fn is_perfect_square(x: i32) -> bool {
    let check = inline_cpp! {
        (int32_t val) -> bool {
          return safer_cffi_examples::IsPerfectSquare(val);
        }
    };
    check(x)
}

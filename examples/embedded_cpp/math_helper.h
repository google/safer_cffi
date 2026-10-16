/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE or
 * https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE or https://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.
 */

#ifndef SAFER_CFFI_EXAMPLES_EMBEDDED_CPP_MATH_HELPER_H_
#define SAFER_CFFI_EXAMPLES_EMBEDDED_CPP_MATH_HELPER_H_

#include <cmath>
#include <cstdint>

namespace safer_cffi_examples {

inline bool IsPerfectSquare(int32_t x) {
  if (x < 0) {
    return false;
  }
  int32_t y = static_cast<int32_t>(std::sqrt(static_cast<double>(x)));
  return y * y == x;
}

}  // namespace safer_cffi_examples

#endif  // SAFER_CFFI_EXAMPLES_EMBEDDED_CPP_MATH_HELPER_H_

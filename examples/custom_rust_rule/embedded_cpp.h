/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE or
 * https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE or https://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.
 */

#ifndef SAFER_CFFI_EXAMPLES_CUSTOM_RUST_RULE_EMBEDDED_CPP_H_
#define SAFER_CFFI_EXAMPLES_CUSTOM_RUST_RULE_EMBEDDED_CPP_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t bit_ceil(uint32_t x);

#ifdef __cplusplus
}
#endif

#endif  // SAFER_CFFI_EXAMPLES_CUSTOM_RUST_RULE_EMBEDDED_CPP_H_

/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE or
 * https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE or https://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.
 */

#ifndef SAFER_CFFI_EXAMPLES_C_BUF_PTR_WITH_CAPACITY_C_BUF_PTR_WITH_CAPACITY_H_
#define SAFER_CFFI_EXAMPLES_C_BUF_PTR_WITH_CAPACITY_C_BUF_PTR_WITH_CAPACITY_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IntArray {
  uint8_t* items;
  int32_t item_len;
  int32_t item_cap;
} IntArray;

void append_to_array(IntArray* array, uint8_t item);

#ifdef __cplusplus
}
#endif

#endif  // SAFER_CFFI_EXAMPLES_C_BUF_PTR_WITH_CAPACITY_C_BUF_PTR_WITH_CAPACITY_H_

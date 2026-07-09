/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE or
 * https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE or https://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.
 */

#ifndef SAFER_CFFI_EXAMPLES_OPAQUE_TRACKER_OPAQUE_TRACKER_H_
#define SAFER_CFFI_EXAMPLES_OPAQUE_TRACKER_OPAQUE_TRACKER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Counter Counter;

Counter* new_counter(void);
void free_counter(Counter* counter);
uint64_t increase_counter(Counter* counter);

#ifdef __cplusplus
}
#endif

#endif  // SAFER_CFFI_EXAMPLES_OPAQUE_TRACKER_OPAQUE_TRACKER_H_

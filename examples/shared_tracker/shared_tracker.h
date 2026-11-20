/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE or
 * https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE or https://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.
 */

#ifndef SAFER_CFFI_EXAMPLES_SHARED_TRACKER_SHARED_TRACKER_H_
#define SAFER_CFFI_EXAMPLES_SHARED_TRACKER_SHARED_TRACKER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SharedCounter SharedCounter;

SharedCounter* new_shared_counter(void);
void free_shared_counter(SharedCounter* counter);

// May be called for the same counter from several threads at the same time.
uint64_t get_shared_counter(const SharedCounter* counter);

// Fails (returns 0) while another call is using the same counter.
uint64_t increase_shared_counter(SharedCounter* counter);

#ifdef __cplusplus
}
#endif

#endif  // SAFER_CFFI_EXAMPLES_SHARED_TRACKER_SHARED_TRACKER_H_

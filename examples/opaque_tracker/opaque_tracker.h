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

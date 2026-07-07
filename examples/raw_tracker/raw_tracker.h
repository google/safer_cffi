#ifndef SAFER_CFFI_EXAMPLES_RAW_TRACKER_RAW_TRACKER_H_
#define SAFER_CFFI_EXAMPLES_RAW_TRACKER_RAW_TRACKER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MyStruct must only be created using `new_struct`.
// It MUST NOT be initialized manually on the C side.
typedef struct MyStruct {
  uint64_t field;
} MyStruct;

MyStruct* new_struct(void);
void free_struct(MyStruct* s);
void print_struct(MyStruct* s);

#ifdef __cplusplus
}
#endif

#endif  // SAFER_CFFI_EXAMPLES_RAW_TRACKER_RAW_TRACKER_H_

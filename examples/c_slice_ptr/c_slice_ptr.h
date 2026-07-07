#ifndef SAFER_CFFI_EXAMPLES_C_SLICE_PTR_C_SLICE_PTR_H_
#define SAFER_CFFI_EXAMPLES_C_SLICE_PTR_C_SLICE_PTR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IntArray {
  int32_t* items;
  int32_t item_count;
} IntArray;

IntArray* create_array(void);
void free_array(IntArray* array);
void append_to_array(IntArray* array, int32_t item);
int32_t sum_array(const IntArray* array);

#ifdef __cplusplus
}
#endif

#endif  // SAFER_CFFI_EXAMPLES_C_SLICE_PTR_C_SLICE_PTR_H_

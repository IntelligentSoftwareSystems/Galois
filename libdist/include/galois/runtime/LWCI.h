#pragma once
#ifdef GALOIS_USE_LWCI
#include "lc.h"
extern lch* mv;

/**
 * Element-wise sum of 2 arrays.
 *
 * @tparam Ty type of elements contained in the arrays
 *
 * @param dst destination array to write to
 * @param src source array to read from
 * @param count Size of array in bytes
 */
template<typename Ty>
void ompi_op_sum(void* dst, void* src, size_t count) {
  Ty* dst_ty = (Ty*) dst;
  Ty* src_ty = (Ty*) src;
  for (size_t i = 0; i < (count/sizeof(Ty)); ++i) {
    dst_ty[i] += src_ty[i];
  }
}

/**
 * Element-wise max of 2 arrays.
 *
 * @tparam Ty type of elements contained in the arrays
 *
 * @param dst destination array to write to
 * @param src source array to read from
 * @param count Size of array in bytes
 */
template<typename Ty>
void ompi_op_max(void* dst, void* src, size_t count) {
  Ty* dst_ty = (Ty*) dst;
  Ty* src_ty = (Ty*) src;
  for (size_t i = 0; i < (count / sizeof(Ty)); ++i) {
    if (dst_ty[i] < src_ty[i]) {
      dst_ty[i] = src_ty[i];
    }
  }
}

/**
 * Element-wise min of 2 arrays.
 *
 * @tparam Ty type of elements contained in the arrays
 *
 * @param dst destination array to write to
 * @param src source array to read from
 * @param count Size of array in bytes
 */
template<typename Ty>
void ompi_op_min(void* dst, void* src, size_t count) {
  Ty* dst_ty = (Ty*) dst;
  Ty* src_ty = (Ty*) src;
  for (size_t i = 0; i < (count / sizeof(Ty)); ++i) {
    if (dst_ty[i] > src_ty[i]) {
      dst_ty[i] = src_ty[i];
    }
  }
}

#endif

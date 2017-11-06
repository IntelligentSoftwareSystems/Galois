#pragma once
#ifdef GALOIS_USE_LWCI
#include "lc.h"
extern lch* mv;

template<typename Ty>
void ompi_op_sum(void* dst, void* src, size_t count) {
  Ty* dst_ty = (Ty*) dst;
  Ty* src_ty = (Ty*) src;
  for (size_t i = 0; i < (count/sizeof(Ty)); ++i) {
    dst_ty[i] += src_ty[i];
  }
}
#endif

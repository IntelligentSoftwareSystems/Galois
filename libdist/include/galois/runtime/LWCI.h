/** LWCI Reduce Functions -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#pragma once
#ifdef GALOIS_USE_LWCI
#include "lc.h"
extern lch* mv;

namespace galois {
namespace runtime {
namespace internal {

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

} // end internal namespace
} // end runtime namespace
} // end galois namespace
#endif

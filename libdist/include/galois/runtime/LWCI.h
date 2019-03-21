/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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

/**
 * @file LWCI.h
 *
 * LWCI header that includes lc.h (LCI library) and internal helper functions
 * on arrays.
 */

#pragma once
#ifdef GALOIS_USE_LWCI
#include "lc.h"

extern lc_ep lc_col_ep;
extern lc_ep lc_p2p_ep[3];

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
template <typename Ty>
void ompi_op_sum(void* dst, void* src, size_t count) {
  Ty* dst_ty = (Ty*)dst;
  Ty* src_ty = (Ty*)src;
  for (size_t i = 0; i < (count / sizeof(Ty)); ++i) {
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
template <typename Ty>
void ompi_op_max(void* dst, void* src, size_t count) {
  Ty* dst_ty = (Ty*)dst;
  Ty* src_ty = (Ty*)src;
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
template <typename Ty>
void ompi_op_min(void* dst, void* src, size_t count) {
  Ty* dst_ty = (Ty*)dst;
  Ty* src_ty = (Ty*)src;
  for (size_t i = 0; i < (count / sizeof(Ty)); ++i) {
    if (dst_ty[i] > src_ty[i]) {
      dst_ty[i] = src_ty[i];
    }
  }
}

} // namespace internal
} // namespace runtime
} // namespace galois
#endif

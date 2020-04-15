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

#ifndef GALOIS_RUNTIME_EXECUTOR_ORDERED_H
#define GALOIS_RUNTIME_EXECUTOR_ORDERED_H

#include "galois/config.h"

namespace galois {
namespace runtime {

// TODO(ddn): Pull in and integrate in executors from exp

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_impl(Iter GALOIS_UNUSED(beg),
                           Iter GALOIS_UNUSED(end),
                           const Cmp& GALOIS_UNUSED(cmp),
                           const NhFunc& GALOIS_UNUSED(nhFunc),
                           const OpFunc& GALOIS_UNUSED(opFunc),
                           const char* GALOIS_UNUSED(loopname)) {
  GALOIS_DIE("not yet implemented");
}

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc,
          typename StableTest>
void for_each_ordered_impl(Iter GALOIS_UNUSED(beg),
                           Iter GALOIS_UNUSED(end),
                           const Cmp& GALOIS_UNUSED(cmp),
                           const NhFunc& GALOIS_UNUSED(nhFunc),
                           const OpFunc& GALOIS_UNUSED(opFunc),
                           const StableTest& GALOIS_UNUSED(stabilityTest),
                           const char* GALOIS_UNUSED(loopname)) {
  GALOIS_DIE("not yet implemented");
}

} // end namespace runtime
} // end namespace galois

#endif

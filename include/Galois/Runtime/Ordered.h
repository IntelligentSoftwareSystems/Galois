/** Ordered execution -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_ORDERED_H
#define GALOIS_RUNTIME_ORDERED_H

#include "Galois/Runtime/Deterministic.h"

namespace Galois {
template<typename IterTy, typename Function1Ty, typename Function2Ty, typename ComparatorTy>
static inline void for_each_ordered(IterTy b, IterTy e, Function1Ty f1, Function2Ty f2, ComparatorTy comp, const char* loopname = 0) {
  typedef typename std::iterator_traits<IterTy>::value_type T;
  typedef GaloisRuntime::Deterministic::OrderedOptions<T,Function1Ty,Function2Ty,ComparatorTy> OptionsTy;
  typedef GaloisRuntime::Deterministic::Executor<OptionsTy> WorkTy;

  OptionsTy options(f1, f2, comp);
  WorkTy W(options, loopname);
  GaloisRuntime::Initializer<IterTy, WorkTy> init(b, e, W);
  for_each_det_impl(init, W);
}
}

#endif

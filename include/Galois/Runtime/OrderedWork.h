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
#ifndef GALOIS_RUNTIME_ORDERED_WORK_H
#define GALOIS_RUNTIME_ORDERED_WORK_H

#include "Galois/Runtime/DeterministicWork.h"
#include "Galois/Runtime/LCordered.h"

namespace GaloisRuntime {


template <typename NhFunc, typename OpFunc>
struct OrderedTraits {

  static const bool NeedsPush = !Galois::does_not_need_push<OpFunc>::value;

  static const bool HasFixedNeighborhood = Galois::has_fixed_neighborhood<NhFunc>::value;

};


template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_impl (Iter beg, Iter end, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, const char* loopname) {
  if (!OrderedTraits<NhFunc, OpFunc>::NeedsPush && OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    // TODO: Remove-only/DAG executor
    GALOIS_ERROR(true, "Remove-only executor not implemented yet");

  } else if (OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    for_each_ordered_lc (beg, end, cmp, nhFunc, opFunc, loopname);

  } else {
    for_each_ordered_2p (beg, end, cmp, nhFunc, opFunc, loopname);
  }
}


template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc, typename StableTest>
void for_each_ordered_impl (Iter beg, Iter end, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, StableTest stabilityTest, const char* loopname) {

  if (!OrderedTraits<NhFunc, OpFunc>::NeedsPush && OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    GALOIS_ERROR(true, "no-adds + fixed-neighborhood == stable-source");

  }
  else if (OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    for_each_ordered_lc (beg, end, cmp, nhFunc, opFunc, stabilityTest, loopname);

  } else {
    GALOIS_ERROR(true, "two-phase executor for unstable-source algorithms not implemented yet");
    // TODO: implement following
    // for_each_ordered_2p (beg, end, cmp, nhFunc, opFunc, stabilityTest, loopname); 
  }
}

} // end namespace GaloisRuntime

#endif // GALOIS_RUNTIME_ORDERED_WORK_H

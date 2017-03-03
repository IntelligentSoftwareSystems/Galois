/** Ordered execution -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_ORDERED_H
#define GALOIS_RUNTIME_EXECUTOR_ORDERED_H

namespace Galois {
namespace Runtime {

// TODO(ddn): Pull in and integrate in executors from exp

#if 0
template <typename NhFunc, typename OpFunc>
struct OrderedTraits {
  static const bool NeedsPush = !Galois::DEPRECATED::does_not_need_push<OpFunc>::value;
  static const bool HasFixedNeighborhood = Galois::DEPRECATED::has_fixed_neighborhood<NhFunc>::value;
};
#endif

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_impl(Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname) {
#if 0
  if (!OrderedTraits<NhFunc, OpFunc>::NeedsPush && OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    // TODO: Remove-only/DAG executor
    gDie("Remove-only executor not implemented yet");
  } else if (OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    for_each_ordered_lc (beg, end, cmp, nhFunc, opFunc, loopname);
  } else {
    for_each_ordered_2p (beg, end, cmp, nhFunc, opFunc, loopname);
  }
#else
  gDie("not yet implemented");
#endif
}


template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc, typename StableTest>
void for_each_ordered_impl(Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const StableTest& stabilityTest, const char* loopname) {
#if 0
  if (!OrderedTraits<NhFunc, OpFunc>::NeedsPush && OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    gDie("no-adds + fixed-neighborhood == stable-source");
  } else if (OrderedTraits<NhFunc, OpFunc>::HasFixedNeighborhood) {
    for_each_ordered_lc (beg, end, cmp, nhFunc, opFunc, stabilityTest, loopname);
  } else {
    gDie("two-phase executor for unstable-source algorithms not implemented yet");
    // TODO: implement following
    // for_each_ordered_2p (beg, end, cmp, nhFunc, opFunc, stabilityTest, loopname); 
  }
#else
  gDie("not yet implemented");
#endif
}

} // end namespace Runtime
} // end namespace Galois

#endif

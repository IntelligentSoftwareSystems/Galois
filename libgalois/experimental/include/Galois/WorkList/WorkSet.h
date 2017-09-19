/** Scalable priority worklist -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */

#ifndef GALOIS_WORKLIST_WORKSET_H
#define GALOIS_WORKLIST_WORKSET_H

#include "Galois/WorkList/Chunked.h"
#include "Galois/WorkList/WorkListHelpers.h"
#include "Galois/PriorityQueue.h"
#include "Galois/UnorderedSet.h"
#include "Galois/TwoLevelSet.h"
#include "Galois/WorkList/Separator.h"
#include "Galois/Timer.h"

namespace Galois {
namespace WorkList {

namespace detail {

template<typename T,
	 typename Scheduler = dChunkedFIFO<64, T>, 
         typename Set = Galois::ThreadSafeOrderedSet<T> > 
struct WorkSetMaster : private boost::noncopyable {
private:
  Scheduler scheduler;
  Set set;
  Galois::Statistic* duplicate;

public:
  typedef T value_type;
  template<typename _T>
  using retype = WorkSetMaster<_T, typename Scheduler::template retype<_T>, typename Set::template retype<_T> >;

  WorkSetMaster() { duplicate = new Galois::Statistic("SchedulerDuplicates"); }

  template<typename... Args>
  WorkSetMaster(Galois::WorkList::Separator dummy, Args... args): scheduler(std::forward<Args>(args)...)
  {
    duplicate = new Galois::Statistic("SchedulerDuplicates");
  }

  ~WorkSetMaster() { delete duplicate; }

  void push(const value_type& val) {
    if(set.push(val)) {
      scheduler.push(val);
    } else {
      *duplicate += 1;
    }
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  Galois::optional<value_type> pop() {
    auto defaultRetVal = Galois::optional<value_type>();
    Galois::optional<value_type> retval = scheduler.pop();

    if(retval == defaultRetVal)
      return defaultRetVal;

    set.remove(retval.get());
    return retval;
  }
};

}  // end namespace detail

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedOrderedSetFIFO = detail::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, Galois::ThreadSafeOrderedSet<T> >;
GALOIS_WLCOMPILECHECK(dChunkedOrderedSetFIFO);

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedUnorderedSetFIFO = detail::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, Galois::ThreadSafeUnorderedSet<T> >;
GALOIS_WLCOMPILECHECK(dChunkedUnorderedSetFIFO);

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedTwoLevelHashFIFO = detail::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, Galois::ThreadSafeTwoLevelHash<T> >;
GALOIS_WLCOMPILECHECK(dChunkedTwoLevelHashFIFO);

template<int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedTwoLevelSetFIFO = detail::WorkSetMaster<T, dChunkedFIFO<ChunkSize,T,Concurrent>, Galois::ThreadSafeTwoLevelSet<T> >;
GALOIS_WLCOMPILECHECK(dChunkedTwoLevelSetFIFO);

} // end namespace WorkList
} // end namespace Galois

#endif

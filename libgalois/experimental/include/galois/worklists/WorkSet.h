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

#ifndef GALOIS_WORKLIST_WORKSET_H
#define GALOIS_WORKLIST_WORKSET_H

#include "galois/worklists/Chunk.h"
#include "galois/worklists/WorkListHelpers.h"
#include "galois/PriorityQueue.h"
#include "galois/UnorderedSet.h"
#include "galois/TwoLevelSet.h"
#include "galois/worklists/Separator.h"
#include "galois/Timer.h"

namespace galois {
namespace worklists {

namespace internal {

template <typename T, typename Scheduler = PerSocketChunkFIFO<64, T>,
          typename Set = galois::ThreadSafeOrderedSet<T>>
struct WorkSetMaster : private boost::noncopyable {
private:
  Scheduler scheduler;
  Set set;
  galois::Statistic* duplicate;

public:
  typedef T value_type;
  template <typename _T>
  using retype = WorkSetMaster<_T, typename Scheduler::template retype<_T>,
                               typename Set::template retype<_T>>;

  WorkSetMaster() { duplicate = new galois::Statistic("SchedulerDuplicates"); }

  template <typename... Args>
  WorkSetMaster(galois::worklists::Separator dummy, Args... args)
      : scheduler(std::forward<Args>(args)...) {
    duplicate = new galois::Statistic("SchedulerDuplicates");
  }

  ~WorkSetMaster() { delete duplicate; }

  void push(const value_type& val) {
    if (set.push(val)) {
      scheduler.push(val);
    } else {
      *duplicate += 1;
    }
  }

  template <typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template <typename RangeTy>
  void push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  galois::optional<value_type> pop() {
    auto defaultRetVal                  = galois::optional<value_type>();
    galois::optional<value_type> retval = scheduler.pop();

    if (retval == defaultRetVal)
      return defaultRetVal;

    set.remove(retval.get());
    return retval;
  }
};

} // end namespace internal

template <int ChunkSize = 64, typename T = int, bool Concurrent = true>
using PerSocketChunkOrderedSetFIFO =
    internal::WorkSetMaster<T, PerSocketChunkFIFO<ChunkSize, T, Concurrent>,
                            galois::ThreadSafeOrderedSet<T>>;
GALOIS_WLCOMPILECHECK(PerSocketChunkOrderedSetFIFO);

template <int ChunkSize = 64, typename T = int, bool Concurrent = true>
using PerSocketChunkUnorderedSetFIFO =
    internal::WorkSetMaster<T, PerSocketChunkFIFO<ChunkSize, T, Concurrent>,
                            galois::ThreadSafeUnorderedSet<T>>;
GALOIS_WLCOMPILECHECK(PerSocketChunkUnorderedSetFIFO);

template <int ChunkSize = 64, typename T = int, bool Concurrent = true>
using PerSocketChunkTwoLevelHashFIFO =
    internal::WorkSetMaster<T, PerSocketChunkFIFO<ChunkSize, T, Concurrent>,
                            galois::ThreadSafeTwoLevelHash<T>>;
GALOIS_WLCOMPILECHECK(PerSocketChunkTwoLevelHashFIFO);

template <int ChunkSize = 64, typename T = int, bool Concurrent = true>
using PerSocketChunkTwoLevelSetFIFO =
    internal::WorkSetMaster<T, PerSocketChunkFIFO<ChunkSize, T, Concurrent>,
                            galois::ThreadSafeTwoLevelSet<T>>;
GALOIS_WLCOMPILECHECK(PerSocketChunkTwoLevelSetFIFO);

} // end namespace worklists
} // end namespace galois

#endif

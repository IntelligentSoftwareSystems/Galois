/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef GALOIS_WORKLIST_MARKING_SET_H
#define GALOIS_WORKLIST_MARKING_SET_H

#include "galois/worklists/Chunk.h"
#include "galois/worklists/WorkListHelpers.h"
#include "galois/worklists/Separator.h"
#include "galois/Timer.h"

namespace galois {
namespace worklists {

namespace internal {

template <typename T, typename Marker,
          typename Scheduler = PerSocketChunkFIFO<64, T>>
struct MarkingWorkSetMaster : private boost::noncopyable {
private:
  Scheduler scheduler;
  Marker marker;
  galois::Statistic* duplicate;

public:
  typedef T value_type;
  template <typename _T>
  using retype =
      MarkingWorkSetMaster<_T, Marker, typename Scheduler::template retype<_T>>;

  MarkingWorkSetMaster(const Marker& m = Marker()) : marker(m) {
    duplicate = new galois::Statistic("SchedulerDuplicates");
  }

  template <typename... Args>
  MarkingWorkSetMaster(const Marker& m, Separator dummy, Args... args)
      : scheduler(std::forward<Args>(args)...), marker(m),
        duplicate(new galois::Statistic("SchedulerDuplicates")) {}

  ~MarkingWorkSetMaster() { delete duplicate; }

  void push(const value_type& val) {
    bool* mark = marker(val);
    while (true) {
      bool inSet = *mark;
      if (inSet) {
        *duplicate += 1;
        break;
      }
      if (__sync_bool_compare_and_swap(mark, inSet, true)) {
        scheduler.push(val);
        break;
      }
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

    bool* mark = marker(retval.get());
    while (true) {
      bool inSet = *mark;
      if (!inSet || __sync_bool_compare_and_swap(mark, inSet, false))
        break;
    }
    return retval;
  }
};

} // end namespace internal

template <typename Marker, int ChunkSize = 64, typename T = int,
          bool Concurrent = true>
using PerSocketChunkMarkingSetFIFO = internal::MarkingWorkSetMaster<
    T, Marker, PerSocketChunkFIFO<ChunkSize, T, Concurrent>>;
GALOIS_WLCOMPILECHECK(PerSocketChunkMarkingSetFIFO);

} // end namespace worklists
} // end namespace galois

#endif

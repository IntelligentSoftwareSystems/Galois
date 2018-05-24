/** Scalable priority worklist -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */

#ifndef GALOIS_WORKLIST_MARKING_SET_H
#define GALOIS_WORKLIST_MARKING_SET_H

#include "galois/worklists/Chunked.h"
#include "galois/worklists/WorkListHelpers.h"
#include "galois/worklists/Separator.h"
#include "galois/Timer.h"

namespace galois {
namespace worklists {

namespace internal {

template<typename T, typename Marker,
	 typename Scheduler = dChunkedFIFO<64, T> >
struct MarkingWorkSetMaster : private boost::noncopyable {
private:
  Scheduler scheduler;
  Marker marker;
  galois::Statistic* duplicate;

public:
  typedef T value_type;
  template<typename _T>
  using retype = MarkingWorkSetMaster<_T, Marker, typename Scheduler::template retype<_T> >;

  MarkingWorkSetMaster(const Marker& m = Marker()): marker(m)
  {
    duplicate = new galois::Statistic("SchedulerDuplicates");
  }

  template<typename... Args>
  MarkingWorkSetMaster(const Marker& m, Separator dummy, Args... args)
    :scheduler(std::forward<Args>(args)...), marker(m), duplicate(new galois::Statistic("SchedulerDuplicates"))
  {
  }

  ~MarkingWorkSetMaster() { delete duplicate; }

  void push(const value_type& val) {
    bool* mark = marker(val);
    while(true) {
      bool inSet = *mark;
      if(inSet) {
        *duplicate += 1;
        break;
      }
      if(__sync_bool_compare_and_swap(mark, inSet, true)) {
        scheduler.push(val);
        break;
      }
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

  galois::optional<value_type> pop() {
    auto defaultRetVal = galois::optional<value_type>();
    galois::optional<value_type> retval = scheduler.pop();

    if(retval == defaultRetVal)
      return defaultRetVal;

    bool* mark = marker(retval.get());
    while(true) {
      bool inSet = *mark;
      if(!inSet || __sync_bool_compare_and_swap(mark, inSet, false))
        break;
    }
    return retval;
  }
};

}  // end namespace internal

template<typename Marker, int ChunkSize=64, typename T=int, bool Concurrent=true>
using dChunkedMarkingSetFIFO = internal::MarkingWorkSetMaster<T, Marker, dChunkedFIFO<ChunkSize,T,Concurrent> >;
GALOIS_WLCOMPILECHECK(dChunkedMarkingSetFIFO);

} // end namespace worklists
} // end namespace galois

#endif

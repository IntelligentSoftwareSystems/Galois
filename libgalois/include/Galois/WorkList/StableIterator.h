/** Stable Iterator worklist -*- C++ -*-
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
 * @section description
 * This dereferences iterators lazily.  This is only safe if they are not
 * invalidated by the operator.  This gives the effect of a do all loop
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_WORKLIST_STABLEITERATOR_H
#define GALOIS_WORKLIST_STABLEITERATOR_H

#include "Galois/gstl.h"
#include "Galois/WorkList/Chunked.h"

namespace galois {
namespace WorkList {

/**
 * Low-overhead worklist when initial range is not invalidated by the
 * operator.
 *
 * @tparam Steal     Try workstealing on initial ranges
 * @tparam Container Worklist to manage work enqueued by the operator
 * @tparam Iterator  (inferred by library)
 */
template<bool Steal = false, typename Container = dChunkedFIFO<>, typename Iterator=int*>
struct StableIterator {
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  typedef Iterator iterator;

  //! change the type the worklist holds
  template<typename _T>
  using retype =  StableIterator<Steal, typename Container::template retype<_T>, Iterator>;

  template<bool b>
  using rethread = StableIterator<Steal, typename Container::template rethread<b>, Iterator>;

  template<typename _iterator>
  struct with_iterator { typedef StableIterator<Steal, Container, _iterator> type; };

  template<bool _steal>
  struct with_steal { typedef StableIterator<_steal, Container, Iterator> type; };

  template<typename _container>
  struct with_container { typedef StableIterator<Steal, _container, Iterator> type; };

private:
  struct shared_state {
    Iterator stealBegin;
    Iterator stealEnd;
    Substrate::SimpleLock stealLock;
    bool stealAvail;
  };

  struct state {
    Substrate::CacheLineStorage<shared_state> stealState;
    Iterator localBegin;
    Iterator localEnd;
    unsigned int nextVictim;
    unsigned int numStealFailures;
    
    void populateSteal() {
      if (Steal && localBegin != localEnd) {
	shared_state& s = stealState.data;
	s.stealLock.lock();
	s.stealEnd = localEnd;
	s.stealBegin = localEnd = galois::split_range(localBegin, localEnd);
	if (s.stealBegin != s.stealEnd)
          s.stealAvail = true;
	s.stealLock.unlock();
      }
    }
  };

  Substrate::PerThreadStorage<state> TLDS;
  Container inner;

  bool doSteal(state& dst, state& src, bool wait) {
    shared_state& s = src.stealState.data;
    if (s.stealAvail) {
      if (wait) {
        s.stealLock.lock();
      } else if (!s.stealLock.try_lock()) {
        return false;
      }
      if (s.stealBegin != s.stealEnd) {
	dst.localBegin = s.stealBegin;
	s.stealBegin = dst.localEnd = galois::split_range(s.stealBegin, s.stealEnd);
        s.stealAvail = s.stealBegin != s.stealEnd;
      }
      s.stealLock.unlock();
    }
    return dst.localBegin != dst.localEnd;
  }

  //pop already failed, try again with stealing
  galois::optional<value_type> pop_steal(state& data) {
    //always try stealing self
    if (doSteal(data, data, true))
      return *data.localBegin++;
    //only try stealing one other
    if (doSteal(data, *TLDS.getRemote(data.nextVictim), false)) {
      //share the wealth
      if (data.nextVictim != Substrate::ThreadPool::getTID())
	data.populateSteal();
      return *data.localBegin++;
    }
    ++data.nextVictim;
    ++data.numStealFailures;
    data.nextVictim %= Runtime::activeThreads;
    return galois::optional<value_type>();
  }

public:
  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(const RangeTy& r) {
    state& data = *TLDS.getLocal();
    auto lp = r.local_pair();
    data.localBegin = lp.first;
    data.localEnd = lp.second;
    data.nextVictim = Substrate::ThreadPool::getTID();
    data.numStealFailures = 0;
    data.populateSteal();
  }

  //! pop a value from the queue.
  galois::optional<value_type> pop() {
    state& data = *TLDS.getLocal();
    if (data.localBegin != data.localEnd)
      return *data.localBegin++;

    galois::optional<value_type> item;
    if (Steal && 2 * data.numStealFailures > Runtime::activeThreads)
      if ((item = pop_steal(data)))
        return item;
    if ((item = inner.pop()))
      return item;
    if (Steal)
      return pop_steal(data);
    return item;
  }

  void push(const value_type& val) {
    inner.push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }
};
GALOIS_WLCOMPILECHECK(StableIterator)

}
}
#endif

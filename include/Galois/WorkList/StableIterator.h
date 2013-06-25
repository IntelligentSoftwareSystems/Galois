/** Stable Iterator worklist -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @description
 * This dereferences iterators lazily.  This is only safe if they are not
 * invalidated by the operator
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_WORKLIST_STABLEITERATOR_H
#define GALOIS_WORKLIST_STABLEITERATOR_H

#include "Galois/gstl.h"

namespace Galois {
namespace WorkList {

template<typename Iterator=int*, bool Steal = false>
struct StableIterator {
  typedef typename std::iterator_traits<Iterator>::value_type value_type;

  //! change the concurrency flag
  template<bool _concurrent>
  struct rethread { typedef StableIterator<Iterator, Steal> type; };
  
  //! change the type the worklist holds
  template<typename _T>
  struct retype { typedef StableIterator<Iterator, Steal> type; };

  template<typename _iterator>
  struct with_iterator { typedef StableIterator<_iterator, Steal> type; };

  template<bool _steal>
  struct with_steal { typedef StableIterator<Iterator, _steal> type; };

private:
  struct shared_state {
    Iterator stealBegin;
    Iterator stealEnd;
    Runtime::LL::SimpleLock<true> stealLock;
    bool stealAvail;
    void resetAvail() {
      if (stealBegin != stealEnd)
	stealAvail = true;
    }
  };

  struct state {
    Runtime::LL::CacheLineStorage<shared_state> stealState;
    Iterator localBegin;
    Iterator localEnd;
    unsigned int nextVictim;
    
    void populateSteal() {
      if (Steal && localBegin != localEnd) {
	shared_state& s = stealState.data;
	s.stealLock.lock();
	s.stealEnd = localEnd;
	s.stealBegin = localEnd = Galois::split_range(localBegin, localEnd);
	s.resetAvail();
	s.stealLock.unlock();
      }
    }
  };

  Runtime::PerThreadStorage<state> TLDS;

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
	s.stealBegin = dst.localEnd = Galois::split_range(s.stealBegin, s.stealEnd);
	s.resetAvail();
      }
      s.stealLock.unlock();
    }
    return dst.localBegin != dst.localEnd;
  }

  //pop already failed, try again with stealing
  Galois::optional<value_type> pop_steal(state& data) {
    //always try stealing self
    if (doSteal(data, data, true))
      return *data.localBegin++;
    //only try stealing one other
    if (doSteal(data, *TLDS.getRemote(data.nextVictim), false)) {
      //share the wealth
      if (data.nextVictim != Runtime::LL::getTID())
	data.populateSteal();
      return *data.localBegin++;
    }
    ++data.nextVictim;
    data.nextVictim %= Runtime::activeThreads;
    return Galois::optional<value_type>();
  }

public:
  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(const RangeTy& r) {
    state& data = *TLDS.getLocal();
    data.localBegin = r.local_begin();
    data.localEnd = r.local_end();
    data.nextVictim = Runtime::LL::getTID();
    data.populateSteal();
  }

  //! pop a value from the queue.
  Galois::optional<value_type> pop() {
    state& data = *TLDS.getLocal();
    if (data.localBegin != data.localEnd)
      return *data.localBegin++;
    if (Steal)
      return pop_steal(data);
    return Galois::optional<value_type>();
  }

  void push(const value_type& val) {
    abort();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    abort();
  }
};
GALOIS_WLCOMPILECHECK(StableIterator)

}
}
#endif

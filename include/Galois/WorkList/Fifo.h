/** FIFO worklist -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_WORKLIST_FIFO_H
#define GALOIS_WORKLIST_FIFO_H

#include "Galois/Runtime/ll/PaddedLock.h"
#include "WLCompileCheck.h"
#include <deque>

namespace Galois {
namespace WorkList {

//! Simple FIFO worklist (not scalable).
template<typename T = int, bool Concurrent = true>
struct FIFO : private boost::noncopyable, private Runtime::LL::PaddedLock<Concurrent>  {
  template<bool _concurrent>
  struct rethread { typedef FIFO<T, _concurrent> type; };

  template<typename _T>
  struct retype { typedef FIFO<_T, Concurrent> type; };

private:
  std::deque<T> wl;

  using Runtime::LL::PaddedLock<Concurrent>::lock;
  using Runtime::LL::PaddedLock<Concurrent>::try_lock;
  using Runtime::LL::PaddedLock<Concurrent>::unlock;

public:
  typedef T value_type;

  void push(const value_type& val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    wl.insert(wl.end(),b,e);
    unlock();
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    if (Runtime::LL::getTID() == 0)
      push(range.begin(), range.end());
  }

  void steal(FIFO& victim) {
    if (!Runtime::LL::TryLockPairOrdered(*this, victim))
      return;
    typename std::deque<T>::iterator split = split_range(victim.wl.begin(), wl.victim.end());
    wl.insert(wl.end(), victim.wl.begin(), split);
    victim.wl.erase(victim.wl.begin(), split);
    UnLockPairOrdered(*this, victim);
  }

  Galois::optional<value_type> pop() {
    Galois::optional<value_type> retval;
    lock();
    if (!wl.empty()) {
      retval = wl.front();
      wl.pop_front();
    }
    unlock();
    return retval;
  }
};
GALOIS_WLCOMPILECHECK(FIFO)


} // end namespace WorkList
} // end namespace Galois

#endif

/** Galois Simple Parallel Loop -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Implementation of the do all loop. Includes various 
 * specializations to operators to reduce runtime overhead.
 * Doesn't do Galoisish things
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_EXECUTOR_DOALL_H
#define GALOIS_RUNTIME_EXECUTOR_DOALL_H

//#include "Galois/gstl.h"
//#include "Galois/gtuple.h"
//#include "Galois/Traits.h"
//#include "Galois/Statistic.h"
//#include "Galois/Substrate/Barrier.h"
//#include "Galois/Runtime/Support.h"
//#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/Statistics.h"
#include "Galois/Runtime/PtrLock.h"
#include "Galois/Runtime/Blocking.h"
#include "Galois/Runtime/PerThreadStorage.h"

//#include <algorithm>
//#include <mutex>
//#include <tuple>

namespace Galois {
namespace Runtime {


// TODO(ddn): Tune stealing. DMR suffers when stealing is on
template<class FunctionTy, class RangeTy>
class DoAllExecutor {
  typedef typename RangeTy::local_iterator iterator;
  FunctionTy F;
  RangeTy range;
  const char* loopname;
  unsigned activeThreads;

  /*
    starving queue
    threads without work go in it.  When all threads are in the queue, stop
  */

  unsigned exec(iterator b, iterator e) {
    unsigned n = 0;
    while (b != e) {
      ++n;
      F(*b++);
    }
    return n;
  }

  struct msg {
    std::atomic<bool> ready;
    bool exit;
    iterator b, e;
    msg* next;
  };
  PtrLock<msg> head;
  std::atomic<unsigned> waiting;
  
  bool wait(iterator& b, iterator& e) {
    //else, add ourselves to the queue
    msg self;
    self.ready = false;
    self.exit = false;
    do {
      self.next = head.getValue();
    } while(head.CAS(self.next, &self));
    auto old = waiting.fetch_add(1);
    //we are the last, signal everyone to exit
    if (old + 1 == activeThreads) {
      msg* m = head.getValue();
      while (m) {
        auto mn = m->next;
        m->exit = true;
        m->next = nullptr;
        m->ready = true;
        m = mn;
      }
    }
    //wait for signal
    while(!self.ready) { asmPause(); }
    if (self.exit)
      return false;
    b = self.b;
    e = self.e;
    return true;
  }

  unsigned tryDonate(iterator& b, iterator& e) {
    if (!waiting)
      return 0;
    head.lock();
    msg* other = head.getValue();
    head.unlock_and_set(other->next);
    --waiting;
    other->next = nullptr;
    auto mid = split_range(b,e);
    other->b = mid;
    other->e = e;
    auto retval = std::distance(mid, e);
    e = mid;
    other->ready = true;
    return retval;
  }
  
public:
  DoAllExecutor(const FunctionTy& _F, const RangeTy& r, unsigned atv, const char* ln)
    :F(_F), range(r), loopname(ln), activeThreads(atv)
  {
    reportLoopInstance(loopname);
  }

  void operator()() {
    //Assume the copy constructor on the functor is readonly
    iterator begin = range.local_begin();
    iterator end = range.local_end();
        
    unsigned long stat_iterations = 0;
    unsigned long stat_donations = 0;

    do {
      do {
        auto mid = split_range(begin,end);
        stat_iterations += exec(begin, mid);
        begin = mid;
        stat_donations += tryDonate(begin,end);
      } while (begin != end);
    } while (wait(begin, end));
    
    reportStat(loopname, "Iterations", stat_iterations, ThreadPool::getTID());
    reportStat(loopname, "Donations", stat_donations, ThreadPool::getTID());
  }
};



template<typename RangeTy, typename FunctionTy>
void do_all_impl(const RangeTy& range, const FunctionTy& f, unsigned activeThreads, const char* loopname = 0, bool steal = false) {
  reportLoopInstance(loopname);
  beginSampling(loopname);
  if (steal) {
    DoAllExecutor<FunctionTy, RangeTy> W(f, range, activeThreads, loopname);
    ThreadPool::getThreadPool().run(activeThreads, std::ref(W));
  } else {
    FunctionTy f_cpy (f);
    ThreadPool::getThreadPool().run(activeThreads, [&f_cpy, &range, &loopname] () {
        auto begin = range.local_begin();
        auto end = range.local_end();
        auto num = std::distance(begin,end);
        while (begin != end)
          f_cpy(*begin++);
        reportState(loopname, "Iterations", num, ThreadPool::getTID());
      });
  }
  endSampling();
}



} // end namespace Runtime
} // end namespace Galois

#endif

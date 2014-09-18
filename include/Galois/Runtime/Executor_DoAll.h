/** Galois Simple Parallel Loop -*- C++ -*-
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
 * Implementation of the do all loop. Includes various 
 * specializations to operators to reduce runtime overhead.
 * Doesn't do Galoisish things
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_DOALL_H
#define GALOIS_RUNTIME_EXECUTOR_DOALL_H

#include "Galois/gstl.h"
#include "Galois/gtuple.h"
#include "Galois/Traits.h"
#include "Galois/Statistic.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Range.h"

#include <algorithm>
#include <mutex>
#include <tuple>

namespace Galois {
namespace Runtime {

// TODO(ddn): Tune stealing. DMR suffers when stealing is on
// TODO: add loopname + stats
template<class FunctionTy, class RangeTy>
class DoAllExecutor {
  typedef typename RangeTy::local_iterator iterator;
  FunctionTy F;
  RangeTy range;
  const char* loopname;

  struct state {
    iterator stealBegin;
    iterator stealEnd;
    LL::SimpleLock stealLock;
    std::atomic<bool> avail;

    state(): avail(false) { stealLock.lock(); }

    void populateSteal(iterator& begin, iterator& end) {
      if (std::distance(begin, end) > 1) {
        avail = true;
        stealEnd = end;
        stealBegin = end = Galois::split_range(begin, end);
      }
      stealLock.unlock();
    }

    bool doSteal(iterator& begin, iterator& end, int minSteal) {
      if (avail) {
        std::lock_guard<LL::SimpleLock> lg(stealLock);
        if (!avail)
          return false;

        if (stealBegin != stealEnd) {
          begin = stealBegin;
          if (std::distance(stealBegin, stealEnd) < 2*minSteal)
            end = stealBegin = stealEnd;
          else
            end = stealBegin = Galois::split_range(stealBegin, stealEnd);
          if (stealBegin == stealEnd)
            avail = false;
          return begin != end;
        }
      }
      return false;
    }
  };

  PerThreadStorage<state> TLDS;

  GALOIS_ATTRIBUTE_NOINLINE
  bool trySteal(state& local, iterator& begin, iterator& end, int minSteal) {
    //First try stealing from self
    if (local.doSteal(begin, end, minSteal))
      return true;
    //Then try stealing from neighbors
    unsigned myID = LL::getTID();
    unsigned myPkg = LL::getPackageForThread(myID);
    //try package neighbors
    for (unsigned x = 0; x < activeThreads; ++x) {
      if (x != myID && LL::getPackageForThread(x) == myPkg) {
        if (TLDS.getRemote(x)->doSteal(begin, end, minSteal)) {
          if (std::distance(begin, end) > minSteal) {
            local.stealLock.lock();
            local.populateSteal(begin, end);
          }
          return true;
        }
      }
    }
    //try some random
    // auto num = (activeThreads + 7) / 8;
    // for (unsigned x = 0; x < num; ++x)
    //   if (TLDS.getRemote()->doSteal(begin, end))
    //     return true;
    return false;
  }

public:
  DoAllExecutor(const FunctionTy& _F, const RangeTy& r, const char* ln)
    :F(_F), range(r), loopname(ln)
  { }

  void operator()() {
    //Assume the copy constructor on the functor is readonly
    iterator begin = range.local_begin();
    iterator end = range.local_end();
    int minSteal = std::distance(begin,end) / 8;
    state& tld = *TLDS.getLocal();

    tld.populateSteal(begin,end);

    do {
      while (begin != end)
        F(*begin++);
    } while (trySteal(tld, begin, end, minSteal));
  }
};

template<typename RangeTy, typename FunctionTy>
void do_all_impl(const RangeTy& range, const FunctionTy& f, const char* loopname = 0, bool steal = false) {
  if (steal) {
    DoAllExecutor<FunctionTy, RangeTy> W(f, range, loopname);
    getSystemThreadPool().run(activeThreads, std::ref(W));
  } else {
    getSystemThreadPool().run(activeThreads, [&f, &range] () {
        auto begin = range.local_begin();
        auto end = range.local_end();
        while (begin != end)
          f(*begin++);
      });
  }
}

template<typename RangeTy, typename FunctionTy, typename TupleTy>
void do_all_gen(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");

  auto dtpl = std::tuple_cat(tpl,
      get_default_trait_values(tpl,
        std::make_tuple(loopname_tag{}, do_all_steal_tag{}),
        std::make_tuple(loopname{}, do_all_steal<>{})));

  do_all_impl(
      r, fn,
      get_by_supertype<loopname_tag>(dtpl).getValue(),
      get_by_supertype<do_all_steal_tag>(dtpl).getValue());
}


} // end namespace Runtime
} // end namespace Galois

#endif

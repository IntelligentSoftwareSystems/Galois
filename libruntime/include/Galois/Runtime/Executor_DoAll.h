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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
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

#include "Galois/gstl.h"
#include "Galois/gtuple.h"
#include "Galois/Traits.h"
#include "Galois/Statistic.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Range.h"

#include <algorithm>
#include <mutex>
#include <tuple>

namespace Galois {
namespace Runtime {

// TODO(ddn): Tune stealing. DMR suffers when stealing is on
// TODO: add loopname + stats
template<typename FunctionTy, typename RangeTy, typename ArgsTy>
class DoAllExecutor {

  static const bool combineStats = exists_by_supertype<combine_stats_by_name_tag, ArgsTy>::value;
  static const bool STEAL = get_type_by_supertype<do_all_steal_tag, ArgsTy>::type::value;

  typedef typename RangeTy::local_iterator iterator;
  FunctionTy F;
  RangeTy range;
  const char* loopname;

  struct state {
    iterator stealBegin;
    iterator stealEnd;
    Substrate::SimpleLock stealLock;
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
        std::lock_guard<Substrate::SimpleLock> lg(stealLock);
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

  Substrate::PerThreadStorage<state> TLDS;

  GALOIS_ATTRIBUTE_NOINLINE
  bool trySteal(state& local, iterator& begin, iterator& end, int minSteal) {
    //First try stealing from self
    if (local.doSteal(begin, end, minSteal))
      return true;
    //Then try stealing from neighbors
    unsigned myID = Substrate::ThreadPool::getTID();
    unsigned myPkg = Substrate::ThreadPool::getPackage();
    auto& tp = Substrate::ThreadPool::getThreadPool();
    //try package neighbors
    for (unsigned x = 0; x < activeThreads; ++x) {
      if (x != myID && tp.getPackage(x) == myPkg) {
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
  DoAllExecutor(const FunctionTy& _F, const RangeTy& r, const ArgsTy& args)
    :
      F(_F), 
      range(r), 
      loopname(get_by_supertype<loopname_tag>(args).getValue())
  {
    if (!combineStats) {
      reportLoopInstance(loopname);
    }
  }

  void operator()() {
    //Assume the copy constructor on the functor is readonly
    iterator begin = range.local_begin();
    iterator end = range.local_end();

    if (!STEAL) {
      while (begin != end) {
        F(*begin++);
      }
    } else {
      int minSteal = std::distance(begin,end) / 8;
      state& tld = *TLDS.getLocal();
      tld.populateSteal(begin,end);

      do {
        while (begin != end) {
          F(*begin++);
        }
      } while (trySteal(tld, begin, end, minSteal));
    }
  }

};

template<typename RangeTy, typename FunctionTy, typename ArgsTy>
void do_all_impl(const RangeTy& range, const FunctionTy& f, const ArgsTy& args) {
  DoAllExecutor<FunctionTy, RangeTy, ArgsTy> W(f, range, args);
  Substrate::ThreadPool::getThreadPool().run(activeThreads, std::ref(W));
};

// template<typename RangeTy, typename FunctionTy, typename ArgsTy>
// void do_all_impl(const RangeTy& range, const FunctionTy& f, const ArgsTy& args) {
// 
  // DoAllExecutor<FunctionTy, RangeTy, ArgsTy> W(f, range, args);
  // Substrate::ThreadPool::getThreadPool().run(activeThreads, std::ref(W));

  // if (steal) {
    // DoAllExecutor<FunctionTy, RangeTy> W(f, range, loopname);
    // Substrate::ThreadPool::getThreadPool().run(activeThreads, std::ref(W));
  // } else {
    // FunctionTy f_cpy (f);
    // Substrate::ThreadPool::getThreadPool().run(activeThreads, [&f_cpy, &range] () {
        // auto begin = range.local_begin();
        // auto end = range.local_end();
        // while (begin != end)
          // f_cpy(*begin++);
      // });
  // }
// }

template<typename RangeTy, typename FunctionTy, typename TupleTy>
void do_all_gen(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");

  auto dtpl = std::tuple_cat(tpl,
      get_default_trait_values(tpl,
        std::make_tuple(loopname_tag{}, do_all_steal_tag{}),
        std::make_tuple(loopname{}, do_all_steal<>{})));

  do_all_impl( r, fn, dtpl);
}


} // end namespace Runtime
} // end namespace Galois

#endif

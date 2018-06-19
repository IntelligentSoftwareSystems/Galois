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

#ifndef GALOIS_RUNTIME_EXECUTOR_DOALL_OLD_H
#define GALOIS_RUNTIME_EXECUTOR_DOALL_OLD_H

#include "galois/gstl.h"
#include "galois/gtuple.h"
#include "galois/Traits.h"
#include "galois/Timer.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/runtime/Support.h"
#include "galois/runtime/Range.h"

#include <algorithm>
#include <mutex>
#include <tuple>

namespace galois {
namespace runtime {

// TODO(ddn): Tune stealing. DMR suffers when stealing is on
// TODO: add loopname + stats
template <typename FunctionTy, typename RangeTy, typename ArgsTy>
class DoAllExecutor {

  static const bool STEAL = exists_by_supertype<steal_tag, ArgsTy>::value;

  typedef typename RangeTy::local_iterator iterator;
  FunctionTy F;
  RangeTy range;
  const char* loopname;

  struct state {
    iterator stealBegin;
    iterator stealEnd;
    substrate::SimpleLock stealLock;
    std::atomic<bool> avail;

    state() : avail(false) { stealLock.lock(); }

    void populateSteal(iterator& begin, iterator& end) {
      if (std::distance(begin, end) > 1) {
        avail      = true;
        stealEnd   = end;
        stealBegin = end = galois::split_range(begin, end);
      }
      stealLock.unlock();
    }

    bool doSteal(iterator& begin, iterator& end, int minSteal) {
      if (avail) {
        std::lock_guard<substrate::SimpleLock> lg(stealLock);
        if (!avail)
          return false;

        if (stealBegin != stealEnd) {
          begin = stealBegin;
          if (std::distance(stealBegin, stealEnd) < 2 * minSteal)
            end = stealBegin = stealEnd;
          else
            end = stealBegin = galois::split_range(stealBegin, stealEnd);
          if (stealBegin == stealEnd)
            avail = false;
          return begin != end;
        }
      }
      return false;
    }
  };

  substrate::PerThreadStorage<state> TLDS;

  GALOIS_ATTRIBUTE_NOINLINE
  bool trySteal(state& local, iterator& begin, iterator& end, int minSteal) {
    // First try stealing from self
    if (local.doSteal(begin, end, minSteal))
      return true;
    // Then try stealing from neighbors
    unsigned myID  = substrate::ThreadPool::getTID();
    unsigned myPkg = substrate::ThreadPool::getSocket();
    auto& tp       = substrate::getThreadPool();
    // try socket neighbors
    for (unsigned x = 0; x < activeThreads; ++x) {
      if (x != myID && tp.getSocket(x) == myPkg) {
        if (TLDS.getRemote(x)->doSteal(begin, end, minSteal)) {
          if (std::distance(begin, end) > minSteal) {
            local.stealLock.lock();
            local.populateSteal(begin, end);
          }
          return true;
        }
      }
    }
    // try some random
    // auto num = (activeThreads + 7) / 8;
    // for (unsigned x = 0; x < num; ++x)
    //   if (TLDS.getRemote()->doSteal(begin, end))
    //     return true;
    return false;
  }

public:
  DoAllExecutor(const FunctionTy& _F, const RangeTy& r, const ArgsTy& args)
      : F(_F), range(r),
        loopname(get_by_supertype<loopname_tag>(args).getValue()) {}

  void operator()() {
    // Assume the copy constructor on the functor is readonly
    iterator begin = range.local_begin();
    iterator end   = range.local_end();

    if (!STEAL) {
      while (begin != end) {
        F(*begin++);
      }
    } else {
      int minSteal = std::distance(begin, end) / 8;
      state& tld   = *TLDS.getLocal();
      tld.populateSteal(begin, end);

      do {
        while (begin != end) {
          F(*begin++);
        }
      } while (trySteal(tld, begin, end, minSteal));
    }
  }
};

template <typename RangeTy, typename FunctionTy, typename ArgsTy>
void do_all_old_impl(const RangeTy& range, const FunctionTy& f,
                     const ArgsTy& args) {
  DoAllExecutor<FunctionTy, RangeTy, ArgsTy> W(f, range, args);
  substrate::getThreadPool().run(activeThreads, std::ref(W));
};

// template<typename RangeTy, typename FunctionTy, typename ArgsTy>
// void do_all_old_impl(const RangeTy& range, const FunctionTy& f, const ArgsTy&
// args) {
//
// DoAllExecutor<FunctionTy, RangeTy, ArgsTy> W(f, range, args);
// substrate::getThreadPool().run(activeThreads, std::ref(W));

// if (steal) {
// DoAllExecutor<FunctionTy, RangeTy> W(f, range, loopname);
// substrate::getThreadPool().run(activeThreads, std::ref(W));
// } else {
// FunctionTy f_cpy (f);
// substrate::getThreadPool().run(activeThreads, [&f_cpy, &range] () {
// auto begin = range.local_begin();
// auto end = range.local_end();
// while (begin != end)
// f_cpy(*begin++);
// });
// }
// }

// TODO: Need to decide whether user should provide num_run tag or
// num_run can be provided by loop instance which is guaranteed to be unique
template <typename RangeTy, typename FunctionTy, typename TupleTy>
void do_all_gen_old(const RangeTy& r, const FunctionTy& fn,
                    const TupleTy& tpl) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const*, TupleTy>::value,
                "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");

  auto dtpl = std::tuple_cat(
      tpl, get_default_trait_values(tpl, std::make_tuple(loopname_tag{}),
                                    std::make_tuple(loopname{})));

  do_all_old_impl(r, fn, dtpl);
}

} // end namespace runtime
} // end namespace galois

#endif // GALOIS_RUNTIME_EXECUTOR_DOALL_OLD_H

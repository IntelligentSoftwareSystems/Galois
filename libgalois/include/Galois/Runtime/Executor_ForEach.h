/** Galois scheduler and runtime -*- C++ -*-
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
 * Implementation of the Galois foreach iterator. Includes various 
 * specializations to operators to reduce runtime overhead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_EXECUTOR_FOREACH_H
#define GALOIS_RUNTIME_EXECUTOR_FOREACH_H

//#include "Galois/Mem.h"
// #include "Galois/Statistic.h"
// #include "Galois/Threads.h"
// #include "Galois/Traits.h"
// #include "Galois/Runtime/Substrate.h"
#include "Galois/Runtime/SyncContext.h"
// #include "Galois/Runtime/ForEachTraits.h"
// #include "Galois/Runtime/Range.h"
// #include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Termination.h"
// #include "Galois/Substrate/ThreadPool.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Runtime/Statistics.h"
#include "Galois/Runtime/GaloisConfig.h"

#include "Galois/WorkList/Chunked.h"
#include "Galois/WorkList/Simple.h"

// #include <algorithm>
// #include <functional>
// #include <memory>
// #include <utility>

namespace Galois {
//! Internal Galois functionality - Use at your own risk.
namespace Runtime {

template<typename value_type>
class AbortHandler {
  struct Item { value_type val;  int retries; };

  typedef WorkList::GFIFO<Item> AbortedList;
  PerThreadStorage<AbortedList> queues;
  bool useBasicPolicy;
  
  /**
   * Policy: serialize via tree over packages.
   */
  void basicPolicy(const Item& item) {
    auto& tp = ThreadPool::getThreadPool();
    unsigned package = tp.getPackage();
    queues.get(tp.getLeaderForPackage(package / 2))->push(item);
  }

  /**
   * Policy: retry work 2X locally, then serialize via tree on package (trying
   * twice at each level), then serialize via tree over packages.
   */
  void doublePolicy(const Item& item) {
    int retries = item.retries - 1;
    if ((retries & 1) == 1) {
      queues->push(item);
      return;
    } 
    
    unsigned tid = ThreadPool::getTID();
    auto& tp = ThreadPool::getThreadPool();
    unsigned package = ThreadPool::getPackage();
    unsigned leader = ThreadPool::getLeader();
    if (tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.get(next)->push(item);
    } else {
      queues.get(tp.getLeaderForPackage(package / 2))->push(item);
    }
  }

  /**
   * Policy: retry work 2X locally, then serialize via tree on package but
   * try at most 3 levels, then serialize via tree over packages.
   */
  void boundedPolicy(const Item& item) {
    int retries = item.retries - 1;
    if (retries < 2) {
      queues->push(item);
      return;
    } 
    
    unsigned tid = ThreadPool::getTID();
    auto& tp = ThreadPool::getThreadPool();
    unsigned package = ThreadPool::getPackage();
    unsigned leader = tp.getLeaderForPackage(package);
    if (retries < 5 && tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.get(next)->push(item);
    } else {
      queues.get(tp.getLeaderForPackage(package / 2))->push(item);
    }
  }

  /**
   * Retry locally only.
   */
  void eagerPolicy(const Item& item) {
    queues->push(item);
  }

public:
  AbortHandler() {
    // XXX(ddn): Implement smarter adaptive policy
    useBasicPolicy = ThreadPool::getThreadPool().getMaxPackages() > 2;
  }

  value_type& value(Item& item) const { return item.val; }
  value_type& value(value_type& val) const { return val; }

  void push(const value_type& val) {
    Item item = { val, 1 };
    queues->push(item);
  }

  void push(const Item& item) {
    Item newitem = { item.val, item.retries + 1 };
    if (useBasicPolicy)
      basicPolicy(newitem);
    else
      doublePolicy(newitem);
  }

  AbortedList* getQueue() { return queues.get(); }
};

//TODO(ddn): Implement wrapper to allow calling without UserContext
//TODO(ddn): Check for operators that implement both with and without context
template<class WorkListTy, class FunctionTy, bool needsStats, bool needsPush, bool needsAborts, bool needsPia, bool needsBreak>
class ForEachExecutor {

protected:
  typedef typename WorkListTy::value_type value_type; 

  struct ThreadLocalData {
    FunctionTy function;
    UserContextAccess<value_type> facing;
    SimpleRuntimeContext ctx;
    unsigned long stat_conflicts;
    unsigned long stat_iterations;
    unsigned long stat_pushes;
    const char* loopname;
    ThreadLocalData(const FunctionTy& fn, const char* ln)
      : function(fn), stat_conflicts(0), stat_iterations(0), stat_pushes(0), 
        loopname(ln)
    {}
    ~ThreadLocalData() {
      if (needsStats) {
        auto ID = ThreadPool::getTID();
        reportStat(loopname, "Conflicts", stat_conflicts, ID);
        reportStat(loopname, "Commits", stat_iterations - stat_conflicts, ID);
        reportStat(loopname, "Pushes", stat_pushes, ID);
        reportStat(loopname, "Iterations", stat_iterations, ID);
      }
    }
  };

  // NB: Place dynamically growing wl after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage

  AbortHandler<value_type> aborted; 
  TerminationDetection& term;

  WorkListTy wl;
  FunctionTy origFunction;
  const char* loopname;
  bool broke;

  inline void commitIteration(ThreadLocalData& tld) {
    if (needsPush) {
      //auto ii = tld.facing.getPushBuffer().begin();
      //auto ee = tld.facing.getPushBuffer().end();
      auto& pb = tld.facing.getPushBuffer();
      auto n = pb.size();
      if (n) {
	tld.stat_pushes += n;
        wl.push(pb.begin(), pb.end());
        pb.clear();
      }
    }
    if (needsPia)
      tld.facing.resetAlloc();
    if (needsAborts)
      tld.ctx.commitIteration();
    //++tld.stat_commits;
  }

  template<typename Item>
  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(const Item& item, ThreadLocalData& tld) {
    assert(needsAborts);
    tld.ctx.cancelIteration();
    ++tld.stat_conflicts;
    aborted.push(item);
    //clear push buffer
    if (needsPush)
      tld.facing.resetPushBuffer();
    //reset allocator
    if (needsPia)
      tld.facing.resetAlloc();
  }

  inline void doProcess(value_type& val, ThreadLocalData& tld) {
    if (needsAborts)
      tld.ctx.startIteration();
    ++tld.stat_iterations;
    tld.function(val, tld.facing.data());
    commitIteration(tld);
  }

  void runQueueSimple(ThreadLocalData& tld) {
    boost::optional<value_type> p;
    while ((p = wl.pop())) {
      doProcess(*p, tld);
    }
  }

  template<int limit, typename WL>
  void runQueue(ThreadLocalData& tld, WL& lwl) {
    boost::optional<typename WL::value_type> p;
    int num = 0;
#ifdef GALOIS_USE_LONGJMP
    if (setjmp(hackjmp) == 0) {
      while ((!limit || num++ < limit) && (p = lwl.pop())) {
	doProcess(aborted.value(*p), tld);
      }
    } else {
      clearReleasable();
      clearConflictLock();
      abortIteration(*p, tld);
    }
#else
    try {
      while ((!limit || num++ < limit) && (p = lwl.pop())) {
	doProcess(aborted.value(*p), tld);
      }
    } catch (ConflictFlag const& flag) {
      abortIteration(*p, tld);
    }
#endif
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleAborts(ThreadLocalData& tld) {
    runQueue<0>(tld, *aborted.getQueue());
  }

  void fastPushBack(typename UserContextAccess<value_type>::PushBufferTy& x) {
    wl.push(x.begin(), x.end());
    x.clear();
  }

  bool checkEmpty(WorkListTy&, ThreadLocalData&, ...) { return true; }

  template<typename WL>
  auto checkEmpty(WL& wl, ThreadLocalData& tld, int) -> decltype(wl.empty(), bool()) {
    return wl.empty();
  }

  template<bool couldAbort, bool isLeader>
  void go() {
    // Thread-local data goes on the local stack to be NUMA friendly
    ThreadLocalData tld(origFunction, loopname);
    if (needsBreak)
      tld.facing.setBreakFlag(&broke);
    if (couldAbort)
      setThreadContext(&tld.ctx);
    if (needsPush && !couldAbort)
      tld.facing.setFastPushBack(
          std::bind(&ForEachExecutor::fastPushBack, this, std::placeholders::_1));
    unsigned long old_iterations = 0;
    // while (true) {
      do {
        // Run some iterations
        if (couldAbort || needsBreak) {
          constexpr int __NUM = (needsBreak || isLeader) ? 64 : 0;
          runQueue<__NUM>(tld, wl);
          // Check for abort
          if (couldAbort)
            handleAborts(tld);
        } else { // No try/catch
          runQueueSimple(tld);
        }

        bool didWork = old_iterations != tld.stat_iterations;
        old_iterations = tld.stat_iterations;

        // Update node color and prop token
        term.localTermination(didWork);
        asmPause(); // Let token propagate
      } while (!term.globalTermination() && (!needsBreak || !broke));

      // if (checkEmpty(wl, tld, 0))
      //   break;
      // if (needsBreak && broke)
      //   break;
    //   term.initializeThread();
    //   barrier.wait();
    // }

    if (couldAbort)
      setThreadContext(0);
  }

public:
  template<typename... WArgsTy>
  ForEachExecutor(const FunctionTy& f, unsigned activeThreads, const char* loopname, std::tuple<WArgsTy...> wlargs)
    : term(getGaloisConfig().getTermination(activeThreads)),
      wl(wlargs),
      origFunction(f),
      loopname(loopname),
      broke(false) 
  {  }

  template<typename RangeTy>
  void initThread(const RangeTy& range) {
    wl.push_initial(range);
    term.initializeThread();
  }

  void operator()() {
    bool isLeader = ThreadPool::isLeader();
    if (needsAborts && isLeader)
      go<true, true>();
    else if (needsAborts && !isLeader)
      go<true, false>();
    else if (!needsAborts && isLeader)
      go<false, true>();
    else
      go<false, false>();
  }
};

template<typename WLTy>
constexpr auto has_with_iterator(int)
  -> decltype(std::declval<typename WLTy::template with_iterator<int*>::type>(), bool()) 
{
  return true;
}

template<typename>
constexpr auto has_with_iterator(...) -> bool {
  return false;
}

template<typename WLTy, typename IterTy, typename Enable = void>
struct reiterator {
  typedef WLTy type;
};

template<typename WLTy, typename IterTy>
struct reiterator<WLTy, IterTy,
  typename std::enable_if<has_with_iterator<WLTy>(0)>::type> 
{
  typedef typename WLTy::template with_iterator<IterTy>::type type;
};

// template<typename Fn, typename T>
// constexpr auto takes_context(int) 
//   -> decltype(std::declval<typename std::result_of<Fn(T&, UserContext<T>&)>::type>(), bool())
// {
//   return true;
// }

// template<typename Fn, typename T>
// constexpr auto takes_context(...) -> bool
// {
//   return false;
// }

// template<typename Fn, typename T, typename Enable = void>
// struct MakeTakeContext
// {
//   Fn fn;

//   void operator()(T& item, UserContext<T>& ctx) const {
//     fn(item);
//   }
// };

// template<typename Fn, typename T>
// struct MakeTakeContext<Fn, T, typename std::enable_if<takes_context<Fn, T>(0)>::type>
// {
//   Fn fn;

//   void operator()(T& item, UserContext<T>& ctx) const {
//     fn(item, ctx);
//   }
// };

// template<typename WorkListTy, typename T, typename RangeTy, typename FunctionTy, typename ArgsTy>
// auto for_each_impl_(const RangeTy& range, const FunctionTy& fn, const ArgsTy& args) 
//   -> typename std::enable_if<takes_context<FunctionTy, T>(0)>::type
// {
//   typedef ForEachExecutor<WorkListTy, FunctionTy, ArgsTy> WorkTy;
//   Barrier& barrier = getSystemBarrier();
//   WorkTy W(fn, args);
//   W.init(range);
//   getThreadPool().run(activeThreads,
//                             [&W, &range]() { W.initThread(range); },
//                             std::ref(barrier),
//                             std::ref(W));
// }

// template<typename WorkListTy, typename T, typename RangeTy, typename FunctionTy, typename ArgsTy>
// auto for_each_impl_(const RangeTy& range, const FunctionTy& fn, const ArgsTy& args) 
//   -> typename std::enable_if<!takes_context<FunctionTy, T>(0)>::type
// {
//   typedef MakeTakeContext<FunctionTy, T> WrappedFunction;
//   auto newArgs = std::tuple_cat(args,
//       get_default_trait_values(args,
//         std::make_tuple(does_not_need_push_tag {}),
//         std::make_tuple(does_not_need_push<> {})));
//   typedef ForEachExecutor<WorkListTy, WrappedFunction, decltype(newArgs)> WorkTy;
//   Barrier& barrier = getSystemBarrier();
//   WorkTy W(WrappedFunction {fn}, newArgs);
//   W.init(range);
//   getThreadPool().run(activeThreads,
//                             [&W, &range]() { W.initThread(range); },
//                             std::ref(barrier),
//                             std::ref(W));
// }

template<bool nStats, bool nPush, bool nAborts, bool nPia, bool nBreak>
struct FEOpts {
  constexpr static bool needsStats = nStats;
  constexpr static bool needsPush = nPush;
  constexpr static bool needsAborts = nAborts;
  constexpr static bool needsPia = nPia;
  constexpr static bool needsBreak = nBreak;
};

template<typename RangeTy, typename FunctionTy, typename BaseWorkListTy, typename... WLArgsTy, typename FEOpts>
void for_each_impl(const RangeTy& range, const FunctionTy& fn, unsigned activeThreads, const char* loopname, s_wl<BaseWorkListTy, WLArgsTy...>& wl, FEOpts opts) {

  typedef typename std::iterator_traits<typename RangeTy::iterator>::value_type value_type; 
  typedef typename reiterator<BaseWorkListTy, typename RangeTy::iterator>::type::template retype<value_type> WorkListTy;

  reportLoopInstance(loopname);

  typedef ForEachExecutor<WorkListTy, FunctionTy, 
                          FEOpts::needsStats, FEOpts::needsPush,
                          FEOpts::needsAborts, FEOpts::needsPia, 
                          FEOpts::needsBreak> WorkTy;

  auto& barrier = getGaloisConfig().getBarrier(activeThreads);
  WorkTy W(fn, activeThreads, loopname, wl.args);
  ThreadPool::getThreadPool().run(activeThreads,
                                  [&W, &range, &barrier]() { 
                                    W.initThread(range); 
                                    barrier();
                                    W();
                                    barrier();
                                  }
                                  );
}

// //! Normalize arguments to for_each
// template<typename RangeTy, typename FunctionTy, typename TupleTy>
// void for_each_gen(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl) {

//   static const bool forceNew = false;
//   static_assert(!forceNew || Runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsAborts, "old type trait");
//   static_assert(!forceNew || Runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsStats, "old type trait");
//   static_assert(!forceNew || Runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPush, "old type trait");
//   static_assert(!forceNew || !Runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsBreak, "old type trait");
//   static_assert(!forceNew || !Runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPIA, "old type trait");
//   if (forceNew) {
//     auto xtpl = std::tuple_cat(tpl, typename function_traits<FunctionTy>::type {});
//     Runtime::for_each_impl(r, fn,
//         std::tuple_cat(xtpl, 
//           get_default_trait_values(tpl,
//             std::make_tuple(loopname_tag {}, wl_tag {}),
//             std::make_tuple(loopname {}, wl<defaultWL>()))));
//   } else {
//     auto tags = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::tags_type {};
//     auto values = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::values_type {};
//     auto ttpl = get_default_trait_values(tpl, tags, values);
//     auto dtpl = std::tuple_cat(tpl, ttpl);
//     auto xtpl = std::tuple_cat(dtpl, typename function_traits<FunctionTy>::type {});
//     Runtime::for_each_impl(r, fn,
//         std::tuple_cat(xtpl,
//           get_default_trait_values(dtpl,
//             std::make_tuple(loopname_tag {}, wl_tag {}),
//             std::make_tuple(loopname {}, wl<defaultWL>()))));
//   }
// }

} // end namespace Runtime
} // end namespace Galois
#endif

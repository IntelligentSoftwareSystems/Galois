/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_RUNTIME_EXECUTOR_FOREACH_H
#define GALOIS_RUNTIME_EXECUTOR_FOREACH_H

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

#include "galois/config.h"
#include "galois/gIO.h"
#include "galois/Mem.h"
#include "galois/runtime/Context.h"
#include "galois/runtime/LoopStatistics.h"
#include "galois/runtime/OperatorReferenceTypes.h"
#include "galois/runtime/Range.h"
#include "galois/runtime/Statistics.h"
#include "galois/runtime/Substrate.h"
#include "galois/runtime/ThreadTimer.h"
#include "galois/runtime/UserContextAccess.h"
#include "galois/substrate/Termination.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/Threads.h"
#include "galois/Timer.h"
#include "galois/Traits.h"
#include "galois/worklists/Chunk.h"
#include "galois/worklists/Simple.h"

namespace galois {
//! Internal Galois functionality - Use at your own risk.
namespace runtime {

template <typename value_type>
class AbortHandler {
  struct Item {
    value_type val;
    int retries;
  };

  typedef worklists::GFIFO<Item> AbortedList;
  substrate::PerThreadStorage<AbortedList> queues;
  bool useBasicPolicy;

  /**
   * Policy: serialize via tree over sockets.
   */
  void basicPolicy(const Item& item) {
    auto& tp        = substrate::getThreadPool();
    unsigned socket = tp.getSocket();
    queues.getRemote(tp.getLeaderForSocket(socket / 2))->push(item);
  }

  /**
   * Policy: retry work 2X locally, then serialize via tree on socket (trying
   * twice at each level), then serialize via tree over sockets.
   */
  void doublePolicy(const Item& item) {
    int retries = item.retries - 1;
    if ((retries & 1) == 1) {
      queues.getLocal()->push(item);
      return;
    }

    unsigned tid    = substrate::ThreadPool::getTID();
    auto& tp        = substrate::getThreadPool();
    unsigned socket = substrate::ThreadPool::getSocket();
    unsigned leader = substrate::ThreadPool::getLeader();
    if (tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.getRemote(next)->push(item);
    } else {
      queues.getRemote(tp.getLeaderForSocket(socket / 2))->push(item);
    }
  }

  /**
   * Policy: retry work 2X locally, then serialize via tree on socket but
   * try at most 3 levels, then serialize via tree over sockets.
   */
  void boundedPolicy(const Item& item) {
    int retries = item.retries - 1;
    if (retries < 2) {
      queues.getLocal()->push(item);
      return;
    }

    unsigned tid    = substrate::ThreadPool::getTID();
    auto& tp        = substrate::getThreadPool();
    unsigned socket = substrate::ThreadPool::getSocket();
    unsigned leader = tp.getLeaderForSocket(socket);
    if (retries < 5 && tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.getRemote(next)->push(item);
    } else {
      queues.getRemote(tp.getLeaderForSocket(socket / 2))->push(item);
    }
  }

  /**
   * Retry locally only.
   */
  void eagerPolicy(const Item& item) { queues.getLocal()->push(item); }

public:
  AbortHandler() {
    // XXX(ddn): Implement smarter adaptive policy
    useBasicPolicy = substrate::getThreadPool().getMaxSockets() > 2;
  }

  value_type& value(Item& item) const { return item.val; }
  value_type& value(value_type& val) const { return val; }

  void push(const value_type& val) {
    Item item = {val, 1};
    queues.getLocal()->push(item);
  }

  void push(const Item& item) {
    Item newitem = {item.val, item.retries + 1};
    if (useBasicPolicy)
      basicPolicy(newitem);
    else
      doublePolicy(newitem);
  }

  AbortedList* getQueue() { return queues.getLocal(); }
};

// TODO(ddn): Implement wrapper to allow calling without UserContext
// TODO(ddn): Check for operators that implement both with and without context
template <class WorkListTy, class FunctionTy, typename ArgsTy>
class ForEachExecutor {
public:
  static constexpr bool needStats = galois::internal::NeedStats<ArgsTy>::value;
  static constexpr bool needsPush = !has_trait<no_pushes_tag, ArgsTy>();
  static constexpr bool needsAborts =
      !has_trait<disable_conflict_detection_tag, ArgsTy>();
  static constexpr bool needsPia   = has_trait<per_iter_alloc_tag, ArgsTy>();
  static constexpr bool needsBreak = has_trait<parallel_break_tag, ArgsTy>();
  static constexpr bool MORE_STATS =
      needStats && has_trait<more_stats_tag, ArgsTy>();

protected:
  typedef typename WorkListTy::value_type value_type;

  struct ThreadLocalBasics {
    UserContextAccess<value_type> facing;
    FunctionTy function;
    SimpleRuntimeContext ctx;

    explicit ThreadLocalBasics(FunctionTy fn) : facing(), function(fn), ctx() {}
  };

  using LoopStat = LoopStatistics<needStats>;

  struct ThreadLocalData : public ThreadLocalBasics, public LoopStat {

    ThreadLocalData(FunctionTy fn, const char* ln)
        : ThreadLocalBasics(fn), LoopStat(ln) {}
  };

  // RunQueueState factors out state within runQueue iterations to protect it
  // from being overwritten when using longjmp/setjmp.
  template <typename WL>
  struct RunQueueState {
    unsigned int num = 0;
    galois::optional<typename WL::value_type> item;
  };

  // NB: Place dynamically growing wl after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage

  AbortHandler<value_type> aborted;
  substrate::TerminationDetection& term;
  substrate::Barrier& barrier;

  WorkListTy wl;
  FunctionTy origFunction;
  const char* loopname;
  bool broke;

  PerThreadTimer<MORE_STATS> initTime;
  PerThreadTimer<MORE_STATS> execTime;

  inline void commitIteration(ThreadLocalData& tld) {
    if (needsPush) {
      // auto ii = tld.facing.getPushBuffer().begin();
      // auto ee = tld.facing.getPushBuffer().end();
      auto& pb = tld.facing.getPushBuffer();
      auto n   = pb.size();
      if (n) {
        tld.inc_pushes(n);
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

  template <typename Item>
  GALOIS_ATTRIBUTE_NOINLINE void abortIteration(const Item& item,
                                                ThreadLocalData& tld) {
    assert(needsAborts);
    tld.ctx.cancelIteration();
    tld.inc_conflicts();
    aborted.push(item);
    // clear push buffer
    if (needsPush)
      tld.facing.resetPushBuffer();
    // reset allocator
    if (needsPia)
      tld.facing.resetAlloc();
  }

  inline void doProcess(value_type& val, ThreadLocalData& tld) {
    if (needsAborts)
      tld.ctx.startIteration();

    tld.inc_iterations();
    tld.function(val, tld.facing.data());
    commitIteration(tld);
  }

  bool runQueueSimple(ThreadLocalData& tld) {
    galois::optional<value_type> p;
    bool didWork = false;
    while ((p = wl.pop())) {
      didWork = true;
      doProcess(*p, tld);
    }
    return didWork;
  }

  template <unsigned int limit, typename WL>
  void runQueueDispatch(ThreadLocalData& tld, WL& lwl, RunQueueState<WL>& s) {
#ifdef GALOIS_USE_LONGJMP_ABORT
    if (setjmp(execFrame) == 0) {
      while ((!limit || s.num < limit) && (s.item = lwl.pop())) {
        ++s.num;
        doProcess(aborted.value(*s.item), tld);
      }
    } else {
      clearConflictLock();
      abortIteration(*s.item, tld);
    }
#elif defined(GALOIS_USE_EXCEPTION_ABORT)
    try {
      while ((!limit || s.num < limit) && (s.item = lwl.pop())) {
        ++s.num;
        doProcess(aborted.value(*s.item), tld);
      }
    } catch (ConflictFlag const& flag) {
      clearConflictLock();
      abortIteration(*s.item, tld);
    }
#endif
  }

  template <unsigned int limit, typename WL>
  bool runQueue(ThreadLocalData& tld, WL& lwl) {
    RunQueueState<WL> s;
    runQueueDispatch<limit>(tld, lwl, s);
    return s.num > 0;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool handleAborts(ThreadLocalData& tld) {
    return runQueue<0>(tld, *aborted.getQueue());
  }

  void fastPushBack(typename UserContextAccess<value_type>::PushBufferTy& x) {
    wl.push(x.begin(), x.end());
    x.clear();
  }

  bool checkEmpty(WorkListTy&, ThreadLocalData&, ...) { return true; }

  template <typename WL>
  auto checkEmpty(WL& wl, ThreadLocalData&, int)
      -> decltype(wl.empty(), bool()) {
    return wl.empty();
  }

  template <bool couldAbort, bool isLeader>
  void go() {

    execTime.start();

    // Thread-local data goes on the local stack to be NUMA friendly
    ThreadLocalData tld(origFunction, loopname);
    if (needsBreak)
      tld.facing.setBreakFlag(&broke);
    if (couldAbort)
      setThreadContext(&tld.ctx);
    if (needsPush && !couldAbort)
      tld.facing.setFastPushBack(std::bind(&ForEachExecutor::fastPushBack, this,
                                           std::placeholders::_1));

    while (true) {
      do {
        bool didWork = false;

        // Run some iterations
        if (couldAbort || needsBreak) {
          constexpr int __NUM = (needsBreak || isLeader) ? 64 : 0;
          bool b              = runQueue<__NUM>(tld, wl);
          didWork             = b || didWork;
          // Check for abort
          if (couldAbort) {
            b       = handleAborts(tld);
            didWork = b || didWork;
          }
        } else { // No try/catch
          bool b  = runQueueSimple(tld);
          didWork = b || didWork;
        }

        // Update node color and prop token
        term.localTermination(didWork);
        substrate::asmPause(); // Let token propagate
      } while (!term.globalTermination() && (!needsBreak || !broke));

      if (checkEmpty(wl, tld, 0)) {
        execTime.stop();
        break;
      }

      if (needsBreak && broke) {
        execTime.stop();
        break;
      }

      term.initializeThread();
      barrier.wait();
    }

    if (couldAbort)
      setThreadContext(0);
  }

  struct T1 {};
  struct T2 {};

  template <typename... WArgsTy>
  ForEachExecutor(T2, FunctionTy f, const ArgsTy& args, WArgsTy... wargs)
      : term(substrate::getSystemTermination(activeThreads)),
        barrier(getBarrier(activeThreads)), wl(std::forward<WArgsTy>(wargs)...),
        origFunction(f), loopname(galois::internal::getLoopName(args)),
        broke(false), initTime(loopname, "Init"),
        execTime(loopname, "Execute") {}

  template <typename WArgsTy, size_t... Is>
  ForEachExecutor(T1, FunctionTy f, const ArgsTy& args, const WArgsTy& wlargs,
                  std::index_sequence<Is...>)
      : ForEachExecutor(T2{}, f, args, std::get<Is>(wlargs)...) {}

  template <typename WArgsTy>
  ForEachExecutor(T1, FunctionTy f, const ArgsTy& args, const WArgsTy&,
                  std::index_sequence<>)
      : ForEachExecutor(T2{}, f, args) {}

public:
  ForEachExecutor(FunctionTy f, const ArgsTy& args)
      : ForEachExecutor(
            T1{}, f, args, get_trait_value<wl_tag>(args).args,
            std::make_index_sequence<std::tuple_size<
                decltype(get_trait_value<wl_tag>(args).args)>::value>{}) {}

  template <typename RangeTy>
  void init(const RangeTy&) {}

  template <typename RangeTy>
  void initThread(const RangeTy& range) {

    initTime.start();

    wl.push_initial(range);
    term.initializeThread();

    initTime.stop();
  }

  void operator()() {
    bool isLeader   = substrate::ThreadPool::isLeader();
    bool couldAbort = needsAborts && activeThreads > 1;
    if (couldAbort && isLeader)
      go<true, true>();
    else if (couldAbort && !isLeader)
      go<true, false>();
    else if (!couldAbort && isLeader)
      go<false, true>();
    else
      go<false, false>();
  }
};

template <typename WLTy>
constexpr auto has_with_iterator(int)
    -> decltype(std::declval<
                    typename WLTy::template with_iterator<int*>::type>(),
                bool()) {
  return true;
}

template <typename>
constexpr auto has_with_iterator(...) -> bool {
  return false;
}

template <typename WLTy, typename IterTy, typename Enable = void>
struct reiterator {
  typedef WLTy type;
};

template <typename WLTy, typename IterTy>
struct reiterator<WLTy, IterTy,
                  typename std::enable_if<has_with_iterator<WLTy>(0)>::type> {
  typedef typename WLTy::template with_iterator<IterTy>::type type;
};

// TODO(ddn): Think about folding in range into args too
template <typename RangeTy, typename FunctionTy, typename ArgsTy>
void for_each_impl(const RangeTy& range, FunctionTy&& fn, const ArgsTy& args) {
  typedef typename std::iterator_traits<typename RangeTy::iterator>::value_type
      value_type;
  typedef typename get_trait_type<wl_tag, ArgsTy>::type::type BaseWorkListTy;
  typedef typename reiterator<BaseWorkListTy, typename RangeTy::iterator>::
      type ::template retype<value_type>
          WorkListTy;
  using FuncRefType =
      OperatorReferenceType<decltype(std::forward<FunctionTy>(fn))>;
  typedef ForEachExecutor<WorkListTy, FuncRefType, ArgsTy> WorkTy;

  auto& barrier      = getBarrier(activeThreads);
  FuncRefType fn_ref = fn;
  WorkTy W(fn_ref, args);
  W.init(range);
  substrate::getThreadPool().run(
      activeThreads, [&W, &range]() { W.initThread(range); }, std::ref(barrier),
      std::ref(W));
}

// TODO: Need to decide whether user should provide num_run tag or
// num_run can be provided by loop instance which is guaranteed to be unique

//! Normalize arguments to for_each
template <typename RangeTy, typename FunctionTy, typename TupleTy>
void for_each_gen(const RangeTy& r, FunctionTy&& fn, const TupleTy& tpl) {
  static_assert(!has_trait<char*, TupleTy>(), "old loopname");
  static_assert(!has_trait<char const*, TupleTy>(), "old loopname");
  static_assert(!has_trait<bool, TupleTy>(), "old steal");

  auto ftpl = std::tuple_cat(tpl, typename function_traits<FunctionTy>::type{});

  auto xtpl = std::tuple_cat(
      ftpl, get_default_trait_values(tpl, std::make_tuple(wl_tag{}),
                                     std::make_tuple(wl<defaultWL>())));

  constexpr bool TIME_IT = has_trait<loopname_tag, decltype(xtpl)>();
  CondStatTimer<TIME_IT> timer(galois::internal::getLoopName(xtpl));

  timer.start();

  runtime::for_each_impl(r, std::forward<FunctionTy>(fn), xtpl);

  timer.stop();
}

} // end namespace runtime
} // end namespace galois
#endif

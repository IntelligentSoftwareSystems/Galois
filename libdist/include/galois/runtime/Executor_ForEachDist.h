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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Implementation of the Galois foreach iterator. Includes various 
 * specializations to operators to reduce runtime overhead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

// TODO: remove dist stuff
#ifndef GALOIS_RUNTIME_EXECUTOR_FOREACH_DIST_H
#define GALOIS_RUNTIME_EXECUTOR_FOREACH_DIST_H

#include "galois/gtuple.h"
#include "galois/Mem.h"
#include "galois/Timer.h"
#include "galois/Threads.h"
#include "galois/Traits.h"
#include "galois/runtime/Substrate.h"
#include "galois/runtime/Context.h"
#include "galois/runtime/ForEachTraits.h"
#include "galois/runtime/Range.h"
#include "galois/runtime/Support.h"
#include "galois/Substrate/Termination.h"
#include "galois/Substrate/ThreadPool.h"
#include "galois/runtime/UserContextAccess.h"
#include "galois/worklists/Chunked.h"
#include "galois/worklists/Simple.h"

#include "galois/runtime/Network.h"
#include "galois/runtime/Serialize.h"

#include "galois/Bag.h"
#include "galois/DistBag.h"
//#include "galois/worklists/WorkListDist.h"
//#include "galois/worklists/WorkListWrapper.h"
#include "galois/worklists/WorkListDist.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <iostream>

namespace galois {
//! Internal Galois functionality - Use at your own risk.
namespace runtime {

static constexpr unsigned GALOIS_DEFAULT_CHUNK_SIZE = 32;
typedef worklists::dChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> defaultWL;

template<typename value_type>
class AbortHandler {
  struct Item { value_type val;  int retries; };

  typedef worklists::GFIFO<Item> AbortedList;
  substrate::PerThreadStorage<AbortedList> queues;
  bool useBasicPolicy;
  
  /**
   * Policy: serialize via tree over packages.
   */
  void basicPolicy(const Item& item) {
    auto& tp = substrate::getThreadPool();
    unsigned package = tp.getPackage();
    queues.getRemote(tp.getLeaderForPackage(package / 2))->push(item);
  }

  /**
   * Policy: retry work 2X locally, then serialize via tree on package (trying
   * twice at each level), then serialize via tree over packages.
   */
  void doublePolicy(const Item& item) {
    int retries = item.retries - 1;
    if ((retries & 1) == 1) {
      queues.getLocal()->push(item);
      return;
    } 
    
    unsigned tid = substrate::ThreadPool::getTID();
    auto& tp = substrate::getThreadPool();
    unsigned package = substrate::ThreadPool::getPackage();
    unsigned leader = substrate::ThreadPool::getLeader();
    if (tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.getRemote(next)->push(item);
    } else {
      queues.getRemote(tp.getLeaderForPackage(package / 2))->push(item);
    }
  }

  /**
   * Policy: retry work 2X locally, then serialize via tree on package but
   * try at most 3 levels, then serialize via tree over packages.
   */
  void boundedPolicy(const Item& item) {
    int retries = item.retries - 1;
    if (retries < 2) {
      queues.getLocal()->push(item);
      return;
    } 
    
    unsigned tid = substrate::ThreadPool::getTID();
    auto& tp = substrate::getThreadPool();
    unsigned package = substrate::ThreadPool::getPackage();
    unsigned leader = tp.getLeaderForPackage(package);
    if (retries < 5 && tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.getRemote(next)->push(item);
    } else {
      queues.getRemote(tp.getLeaderForPackage(package / 2))->push(item);
    }
  }

  /**
   * Retry locally only.
   */
  void eagerPolicy(const Item& item) {
    queues.getLocal()->push(item);
  }

public:
  AbortHandler() {
    // XXX(ddn): Implement smarter adaptive policy
    useBasicPolicy = substrate::getThreadPool().getMaxPackages() > 2;
  }

  value_type& value(Item& item) const { return item.val; }
  value_type& value(value_type& val) const { return val; }

  void push(const value_type& val) {
    Item item = { val, 1 };
    queues.getLocal()->push(item);
  }

  void push(const Item& item) {
    Item newitem = { item.val, item.retries + 1 };
    if (useBasicPolicy)
      basicPolicy(newitem);
    else
      doublePolicy(newitem);
  }

  AbortedList* getQueue() { return queues.getLocal(); }
};

//TODO(ddn): Implement wrapper to allow calling without UserContext
//TODO(ddn): Check for operators that implement both with and without context
template<class WorkListTy, class FunctionTy, typename ArgsTy>
class ForEachExecutor {
public:
  static const bool needsStats = !exists_by_supertype<does_not_need_stats_tag, ArgsTy>::value;
  static const bool needsPush = !exists_by_supertype<does_not_need_push_tag, ArgsTy>::value;
  static const bool needsAborts = !exists_by_supertype<does_not_need_aborts_tag, ArgsTy>::value;
  static const bool needsPia = exists_by_supertype<needs_per_iter_alloc_tag, ArgsTy>::value;
  static const bool needsBreak = exists_by_supertype<needs_parallel_break_tag, ArgsTy>::value;
  static const bool combineStats = exists_by_supertype<combine_stats_by_name_tag, ArgsTy>::value;

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
        unsigned tid = substrate::ThreadPool::getTID();
        reportStat(loopname, "Conflicts", stat_conflicts, tid);
        reportStat(loopname, "Commits", stat_iterations - stat_conflicts, tid);
        reportStat(loopname, "Pushes", stat_pushes, tid);
        reportStat(loopname, "Iterations", stat_iterations, tid);
      }
    }
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
    galois::optional<value_type> p;
    while ((p = wl.pop())) {
      doProcess(*p, tld);
    }
  }

  template<int limit, typename WL>
  void runQueue(ThreadLocalData& tld, WL& lwl) {
    galois::optional<typename WL::value_type> p;
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
    while (true) {
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
        substrate::asmPause(); // Let token propagate
      } while (!term.globalTermination() && (!needsBreak || !broke));

      if (checkEmpty(wl, tld, 0))
        break;
      if (needsBreak && broke)
        break;
      term.initializeThread();
      barrier.wait();
    }

    if (couldAbort)
      setThreadContext(0);
  }

  struct T1 {}; struct T2 {};

  template<typename... WArgsTy>
  ForEachExecutor(T2, const FunctionTy& f, const ArgsTy& args, WArgsTy... wargs):
    term(substrate::getSystemTermination(activeThreads)),
    barrier(getBarrier(activeThreads)),
    wl(std::forward<WArgsTy>(wargs)...),
    origFunction(f),
    loopname(get_by_supertype<loopname_tag>(args).value),
    broke(false) 
  {
    if (!combineStats) {
      reportLoopInstance(loopname);
    }
  }

  template<typename WArgsTy, int... Is>
  ForEachExecutor(T1, const FunctionTy& f, 
                  const ArgsTy& args, const WArgsTy& wlargs, int_seq<Is...>):
    ForEachExecutor(T2{}, f, args, std::get<Is>(wlargs)...) {}

  template<typename WArgsTy>
  ForEachExecutor(T1, const FunctionTy& f, 
                  const ArgsTy& args, const WArgsTy& wlargs, int_seq<>):
    ForEachExecutor(T2{}, f, args) {}

public:
  ForEachExecutor(const FunctionTy& f, const ArgsTy& args):
    ForEachExecutor(T1{}, f, args, 
                    get_by_supertype<wl_tag>(args).args, 
                    typename make_int_seq<std::tuple_size<decltype(get_by_supertype<wl_tag>(args).args)>::value>::type{}) {}
  
  template<typename RangeTy>
  void init(const RangeTy&) { }

  template<typename RangeTy>
  void initThread(const RangeTy& range) {
    wl.push_initial(range);
    term.initializeThread();
  }

  void operator()() {
    bool isLeader = substrate::ThreadPool::isLeader();
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

//TODO(ddn): Think about folding in range into args too
template<typename RangeTy, typename FunctionTy, typename ArgsTy>
void for_each_impl(const RangeTy& range, const FunctionTy& fn, const ArgsTy& args) {
  typedef typename std::iterator_traits<typename RangeTy::iterator>::value_type value_type; 
  typedef typename get_type_by_supertype<wl_tag, ArgsTy>::type::type BaseWorkListTy;
  typedef typename reiterator<BaseWorkListTy, typename RangeTy::iterator>::type
    ::template retype<value_type> WorkListTy;
  //typedef typename WorkListTy::value_type g;
  typedef ForEachExecutor<WorkListTy, FunctionTy, ArgsTy> WorkTy;

  auto& barrier = getBarrier(activeThreads);
  WorkTy W(fn, args);
  W.init(range);
  substrate::getThreadPool().run(activeThreads,
             [&W, &range]() { W.initThread(range); },
             std::ref(barrier),
             std::ref(W));
  //  for_each_impl_<WorkListTy, value_type>(range, fn, args);
}


//TODO(ggill): Try to remove for_each_impl_dist and just it with for_each_impl
template<typename RangeTy, typename FunctionTy, typename ArgsTy>
void for_each_impl_dist(const RangeTy& range, const FunctionTy& fn, const ArgsTy& args) {
  typedef typename get_type_by_supertype<wl_tag, ArgsTy>::type::type WorkListTy;
  typedef ForEachExecutor<WorkListTy, FunctionTy, ArgsTy> WorkTy;

  auto& barrier = getBarrier(activeThreads);
  WorkTy W(fn, args);

  W.init(range);
  substrate::getThreadPool().run(
             activeThreads,
             [&W, &range]() { W.initThread(range); },
             std::ref(barrier),
             std::ref(W));

  //  for_each_impl_<WorkListTy, value_type>(range, fn, args);
}


//! Normalize arguments to for_each
template<typename RangeTy, typename FunctionTy, typename TupleTy>
void for_each_gen(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");

  static const bool forceNew = false;
  static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsAborts, "old type trait");
  static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsStats, "old type trait");
  static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPush, "old type trait");
  static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsBreak, "old type trait");
  static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPIA, "old type trait");
  if (forceNew) {
    auto xtpl = std::tuple_cat(tpl, typename function_traits<FunctionTy>::type {});
    runtime::for_each_impl(r, fn,
        std::tuple_cat(xtpl, 
          get_default_trait_values(tpl,
            std::make_tuple(loopname_tag {}, wl_tag {}),
            std::make_tuple(loopname {}, wl<defaultWL>()))));
  } else {
    auto tags = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::tags_type {};
    auto values = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::values_type {};
    auto ttpl = get_default_trait_values(tpl, tags, values);
    auto dtpl = std::tuple_cat(tpl, ttpl);
    auto xtpl = std::tuple_cat(dtpl, typename function_traits<FunctionTy>::type {});
    runtime::for_each_impl(r, fn,
        std::tuple_cat(xtpl,
          get_default_trait_values(dtpl,
            std::make_tuple(loopname_tag {}, wl_tag {}),
            std::make_tuple(loopname {}, wl<defaultWL>()))));
  }
}

//Tag dispatching
template<typename RangeTy, typename FunctionTy, typename TupleTy>
void for_each_gen_dist(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl){

  auto dtpl = std::tuple_cat(tpl, get_default_trait_values(tpl, std::make_tuple(wl_tag{}), std::make_tuple(wl<defaultWL>())));
  for_each_gen_dist_impl(r, fn, dtpl, std::integral_constant<bool, exists_by_supertype<op_tag, TupleTy>::value>());
}

/** For distributed worklist **/
template<typename RangeTy, typename FunctionTy, typename TupleTy>
  void for_each_gen_dist_impl(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl, std::true_type) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");


  if(exists_by_supertype<op_tag, TupleTy>::value)
  {
    auto helper_fn = get_by_supertype<op_tag>(tpl);

    static const bool forceNew = false;
    static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsAborts, "old type trait");
    static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsStats, "old type trait");
    static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPush, "old type trait");
    static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsBreak, "old type trait");
    static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPIA, "old type trait");

     typedef typename get_type_by_supertype<wl_tag, TupleTy>::type::type BaseWorkListTy;
    typedef typename std::iterator_traits<typename RangeTy::iterator>::value_type value_type;


    typedef typename reiterator<BaseWorkListTy, typename RangeTy::iterator>::type
    ::template retype<value_type> WorkListTy;

    //typedef typename BaseWorkListTy::value_type value_type_base;
    /** Construct new worklist **/
    typedef galois::InsertBag<value_type> Bag;
    Bag bag;
    auto ytpl = get_tuple_without(wl_tag{}, tpl);
    auto ztpl = std::tuple_cat(ytpl, std::make_tuple(wl<galois::worklists::WLdistributed<WorkListTy>>(&bag)));
    auto xtpl = std::tuple_cat(ztpl, typename function_traits<FunctionTy>::type {});

    std::string loopName(get_by_supertype<loopname_tag>(tpl).value);
    //std::string timer_for_each_str("FOR_EACH_IMPL_" + loopName + "_" + std::to_string(helper_fn.get_run_num()));
    std::string timer_for_each_str("FOR_EACH_IMPL_" + loopName + "_" + helper_fn.get_run_identifier());
    galois::StatTimer Timer_for_each_impl(timer_for_each_str.c_str());

    helper_fn.set_num_iter(0);
    Timer_for_each_impl.start();
    runtime::for_each_impl_dist(r, fn,
        std::tuple_cat(xtpl,
          get_default_trait_values(ztpl,
            std::make_tuple(loopname_tag {}, wl_tag {}),
            std::make_tuple(loopname {}, wl<defaultWL>()))));
    Timer_for_each_impl.stop();
    galois::runtime::reportStat("(NULL)", "NUM_WORK_ITEMS_" + (helper_fn.get_run_identifier()), r.end() - r.begin(), 0);

    typedef galois::DGBag<value_type, decltype(helper_fn)> DBag;
    DBag dbag(helper_fn, loopName);
    auto &local_wl = DBag::get();

    // Sync
    helper_fn.sync_graph();

    dbag.set(bag);
#ifdef __GALOIS_DEBUG_WORKLIST__
    std::cout << "[" << galois::runtime::getSystemNetworkInterface().ID << "] worklist size : " << std::distance(bag.begin(), bag.end()) << "\n";
#endif
    dbag.sync();


    /** loop while work in the worklist **/
    unsigned num_iterations = 1;
    while(!dbag.canTerminate()) {
      helper_fn.set_num_iter(num_iterations);

      //std::cout << "["<< galois::runtime::getSystemNetworkInterface().ID <<"] Iter : " << num_iterations <<" Total items to work on : " << local_wl.size() << "\n";

      // call for_each again.
      Timer_for_each_impl.start();
      bag.clear();
      if(!local_wl.empty()){
        runtime::for_each_impl_dist(runtime::makeStandardRange(local_wl.begin(), local_wl.end()), fn,
            std::tuple_cat(xtpl,
                get_default_trait_values(ztpl,
                std::make_tuple(loopname_tag {}, wl_tag {}),
                std::make_tuple(loopname {}, wl<defaultWL>()))));
      }
      Timer_for_each_impl.stop();
      galois::runtime::reportStat("(NULL)", "NUM_WORK_ITEMS_" + (helper_fn.get_run_identifier()), local_wl.end() - local_wl.begin(), 0);

      // Sync
      helper_fn.sync_graph();

      dbag.set(bag);
#ifdef __GALOIS_DEBUG_WORKLIST__
      std::cout << "[" << galois::runtime::getSystemNetworkInterface().ID << "] worklist size : " << std::distance(bag.begin(), bag.end()) << "\n";
#endif
      dbag.sync();

      ++num_iterations;
    }
    galois::runtime::reportStat("(NULL)", "NUM_ITERATIONS_" + std::to_string(helper_fn.get_run_num()), (unsigned long)num_iterations, 0);

     //std::cout << "\n\n TERMINATING on : " << net.ID << "\n\n";

  }
  else{
    /**CHECK with exist with exist_by_supertype to call special gen_dist . which will extract from tupe OP. get_by_super<foo_tag> **/
    static const bool forceNew = false;
    static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsAborts, "old type trait");
    static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsStats, "old type trait");
    static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPush, "old type trait");
    static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsBreak, "old type trait");
    static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPIA, "old type trait");
    if (forceNew) {
      auto xtpl = std::tuple_cat(tpl, typename function_traits<FunctionTy>::type {});
      runtime::for_each_impl(r, fn,
          std::tuple_cat(xtpl,
          get_default_trait_values(tpl,
            std::make_tuple(loopname_tag {}, wl_tag {}),
            std::make_tuple(loopname {}, wl<defaultWL>()))));
  } else {
    auto tags = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::tags_type {};
    auto values = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::values_type {};
    auto ttpl = get_default_trait_values(tpl, tags, values);
    auto dtpl = std::tuple_cat(tpl, ttpl);
    auto xtpl = std::tuple_cat(dtpl, typename function_traits<FunctionTy>::type {});
    runtime::for_each_impl(r, fn,
        std::tuple_cat(xtpl,
          get_default_trait_values(dtpl,
            std::make_tuple(loopname_tag {}, wl_tag {}),
            std::make_tuple(loopname {}, wl<defaultWL>()))));
  }

  }
}

//basic template if no op_tag helper function is provided.
template<typename RangeTy, typename FunctionTy, typename TupleTy>
void for_each_gen_dist_impl(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl, std::false_type) {
  static const bool forceNew = false;

  static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsAborts, "old type trait");
  static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsStats, "old type trait");
  static_assert(!forceNew || runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPush, "old type trait");
  static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsBreak, "old type trait");
  static_assert(!forceNew || !runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPIA, "old type trait");
  if (forceNew) {
    auto xtpl = std::tuple_cat(tpl, typename function_traits<FunctionTy>::type {});
    runtime::for_each_impl(r, fn,
        std::tuple_cat(xtpl,
          get_default_trait_values(tpl,
            std::make_tuple(loopname_tag {}, wl_tag {}),
            std::make_tuple(loopname {}, wl<defaultWL>()))));
  } else {
    auto tags = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::tags_type {};
    auto values = typename DEPRECATED::ExtractForEachTraits<FunctionTy>::values_type {};
    auto ttpl = get_default_trait_values(tpl, tags, values);
    auto dtpl = std::tuple_cat(tpl, ttpl);
    auto xtpl = std::tuple_cat(dtpl, typename function_traits<FunctionTy>::type {});
    runtime::for_each_impl(r, fn,
        std::tuple_cat(xtpl,
          get_default_trait_values(dtpl,
            std::make_tuple(loopname_tag {}, wl_tag {}),
            std::make_tuple(loopname {}, wl<defaultWL>()))));
  }


}


} // end namespace runtime
} // end namespace galois
#endif

/** Galois scheduler and runtime -*- C++ -*-
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
 * Implementation of the Galois foreach iterator. Includes various 
 * specializations to operators to reduce runtime overhead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_PARALLELWORK_H
#define GALOIS_RUNTIME_PARALLELWORK_H

#include "Galois/Mem.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Runtime/ActiveThreads.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/WorkList/GFifo.h"
#include "Galois/Runtime/DistSupport.h"

#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <functional>
#include <unordered_set>
//#include <iostream>

namespace Galois {
//! Internal Galois functionality - Use at your own risk.
namespace Runtime {

namespace {

template <bool Enabled> 
class LoopStatistics {
  unsigned long conflicts;
  unsigned long iterations;
  const char* loopname;

public:
  explicit LoopStatistics(const char* ln) :conflicts(0), iterations(0), loopname(ln) { }
  ~LoopStatistics() {
    reportStat(loopname, "Conflicts", conflicts);
    reportStat(loopname, "Iterations", iterations);
  }
  inline void inc_iterations(int amount = 1) {
    iterations += amount;
  }
  inline void inc_conflicts() {
    ++conflicts;
  }
};


template <>
class LoopStatistics<false> {
public:
  explicit LoopStatistics(const char* ln) {}
  inline void inc_iterations(int amount = 1) const { }
  inline void inc_conflicts() const { }
};

template<typename value_type>
class AbortHandler {
  typedef WorkList::GFIFO<value_type> AbortedList;
  PerPackageStorage<AbortedList> queues;

public:
  void push(bool recursiveAbort, value_type val) {
    if (recursiveAbort)
      queues.getRemote(LL::getLeaderForPackage(LL::getPackageForSelf(LL::getTID()) / 2))->push(val);
    else
      queues.getLocal()->push(val);
  }

  boost::optional<value_type> pop() {
    return queues.getLocal()->pop();
  }
};

template<typename value_type>
class RemoteAbortHandler {
  WorkList::GFIFO<value_type> queues;
  std::multimap<fatPointer, value_type> waiting;
  std::multimap<value_type, fatPointer> holding;
  std::unordered_set<value_type> hasState;

  void arrive(fatPointer ptr) {
    assert(waiting.count(ptr));
    for (auto ii = waiting.lower_bound(ptr), ee = waiting.upper_bound(ptr); ii != ee; ++ii)
      queues.push(ii->second);
    waiting.erase(ptr);
  }

  int size() {
    return waiting.size();
  }

public:

  void push(value_type val, fatPointer ptr) {
    //    std::cerr << "pushing " << val << " to " << ptr.first << " " << ptr.second << " with " << waiting.size() << "\n";
    hasState.insert(val);
    bool skipNotify =  waiting.count(ptr);
    waiting.insert(std::make_pair(ptr, val));
    holding.insert(std::make_pair(val, ptr));
    auto& dir = getSystemDirectory();
    dir.setContended(ptr);
    if (!skipNotify)
      dir.notifyWhenAvailable(ptr, std::bind(&RemoteAbortHandler::arrive, this, std::placeholders::_1));
  }

  boost::optional<value_type> pop() {
    return queues.pop();
  }

  void commit(value_type val) {
    if (hasState.count(val)) {
      //std::cerr << "commit remote " << val << "\n";
      hasState.erase(val);
      auto& dir = getSystemDirectory();
      for (auto ii = holding.lower_bound(val), ee = holding.upper_bound(val); ii != ee; ++ii)
        dir.clearContended(ii->second);
      holding.erase(val);
    }
  }

  //doesn't check queue, just hidden work
  bool empty() {
    return waiting.empty();
  }
};


template<typename value_type, bool NeedsStats, bool NeedsPIA, bool NeedsAborts, bool NeedsPush>
class ThreadLocalExec {
  LoopStatistics<NeedsStats> stat;
  SimpleRuntimeContext context;
  
  inline void resetAlloc() {
    if (NeedsPIA) facing.resetAlloc();
  }
  
public:
  UserContextAccess<value_type> facing;

  ThreadLocalExec(const char* ln): stat(ln) {
    if (NeedsAborts)
      setThreadContext(&context);
  }

  ~ThreadLocalExec() {
    if (NeedsAborts)
      setThreadContext(nullptr);
  }
  
  //commit with no intention of continuing computation
  void commit_terminal() {
    if (NeedsAborts)
      context.commit_iteration();
    resetAlloc();
  }

  //commit with every intention of doing more work
  template<typename WLTy>
  inline void commit(WLTy& wl, value_type val) {
    if (NeedsPush) {
      auto ii = facing.getPushBuffer().begin();
      auto ee = facing.getPushBuffer().end();
      if (ii != ee) {
        wl.push(ii,ee);
        facing.resetPushBuffer();
      }
    }
    commit_terminal();
  }
  
  //abort an item
  GALOIS_ATTRIBUTE_NOINLINE
  void abort(value_type val) {
    assert(NeedsAborts);
    context.cancel_iteration();
    stat.inc_conflicts(); //Class specialization handles opt
    if (NeedsPush)
      facing.resetPushBuffer();
    resetAlloc();
  }

  //begin an iteration
  inline void start() {
    stat.inc_iterations();
    if (NeedsAborts)
      context.start_iteration();
  }
};

template<class WorkListTy, class T, class FunctionTy>
class ForEachWork {
protected:
  typedef T value_type;
  typedef typename WorkListTy::template retype<value_type> WLTy;
  typedef gdeque<std::pair<std::pair<value_type,Lockable*>,std::pair<unsigned,bool> > > ReceivedList;
  typedef ThreadLocalExec<value_type, ForEachTraits<FunctionTy>::NeedsStats, ForEachTraits<FunctionTy>::NeedsPIA, ForEachTraits<FunctionTy>::NeedsAborts, ForEachTraits<FunctionTy>::NeedsPush> ThreadLocalData;

  WLTy wl;
  FunctionTy& origFunction;
  const char* loopname;

  TerminationDetection& term;
  AbortHandler<value_type> aborted;
  RemoteAbortHandler<value_type> remote_aborted;
  LL::CacheLineStorage<bool> broke;

  inline void execItem(boost::optional<value_type>& p, FunctionTy& func, ThreadLocalData& tld) {
    tld.start();
    func(*p, tld.facing.data());
    tld.commit(wl, *p);
    remote_aborted.commit(*p);
  }

  GALOIS_ATTRIBUTE_NOINLINE
  template<typename WL>
  bool runQueueSimple(FunctionTy& func, ThreadLocalData& tld, WL& lwl) {
    boost::optional<value_type> p = lwl.pop();
    if (p) {
      do {
	execItem(p, func, tld);
      } while ((p = lwl.pop()));
      return true;
    }
    return false;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  template<unsigned limit, typename WL>
  bool runQueue(FunctionTy& func, ThreadLocalData& tld, WL& lwl, bool recursiveAbort) {
    boost::optional<value_type> p = lwl.pop();
    if (p) {
      unsigned runlimit = limit;
      do {
	try {
	  execItem(p, func, tld);
	} catch (const conflict_ex& ex) {
	  tld.abort(*p);
	  aborted.push(recursiveAbort, *p);
	} catch (const remote_ex& ex) {
	  tld.abort(*p);
	  //aborted.push(recursiveAbort, *p);
	  remote_aborted.push(*p, ex.ptr);
	}
	if (limit)
	  --runlimit;
      } while ((limit == 0 || runlimit != 0) && (p = lwl.pop()));
      return true;
    }
    return false;
  }

  void fastPushBack(typename UserContextAccess<value_type>::PushBufferTy& x) {
    wl.push(x.begin(), x.end());
    x.clear();
  }

  template<bool checkAbort>
  void go() {
    //Thread Local Data goes on the local stack
    //to be NUMA friendly
    ThreadLocalData tld(loopname);
    FunctionTy func(origFunction);

    if (false && ForEachTraits<FunctionTy>::NeedsPush && !ForEachTraits<FunctionTy>::NeedsAborts)
      tld.facing.setFastPushBack(std::bind(&ForEachWork::fastPushBack, this, std::placeholders::_1));

    bool didWork;
    try {
      do {
        didWork = false;
        //Run some iterations
        if (ForEachTraits<FunctionTy>::NeedsAborts)
          didWork = runQueue<checkAbort ? 32 : 0>(func, tld, wl, false);
        else //No try/catch
          didWork = runQueueSimple(func, tld, wl);
        //Check for break
        if (ForEachTraits<FunctionTy>::NeedsBreak && broke.data)
          break;
        //Check for abort, also guards random network work
        if (checkAbort) {
          didWork |= runQueue<32>(func, tld, aborted, true);
          didWork |= runQueue<32>(func, tld, remote_aborted, true);
          didWork |= !remote_aborted.empty();
          doNetworkWork();
        }
        // update node color and prop token
        term.localTermination(didWork);
      } while (!term.globalTermination());
    } catch (const break_ex& ex) {
      tld.commit_terminal();
      broke.data = true;
    }
  }

public:
  ForEachWork(FunctionTy& f, const char* l): origFunction(f), loopname(l), term(getSystemTermination()), broke(false) {
    //LL::gDebug("Type traits stats ", ForEachTraits<FunctionTy>::NeedsStats, " break ", ForEachTraits<FunctionTy>::NeedsBreak, " push ", ForEachTraits<FunctionTy>::NeedsPush, " PIA ", ForEachTraits<FunctionTy>::NeedsPIA, "Aborts ", ForEachTraits<FunctionTy>::NeedsAborts);
  }
  
  template<typename W>
  ForEachWork(W& w, FunctionTy& f, const char* l): wl(w), origFunction(f), loopname(l), term(getSystemTermination()), broke(false) { }

  template<typename RangeTy>
  void AddInitialWork(const RangeTy& range) { wl.push_initial(range); }

  void initThread(void) { term.initializeThread(); }

  // in the distributed case even with 1 thread there can be aborts
  void operator()() {
    if ((LL::isPackageLeaderForSelf(LL::getTID()) &&
         activeThreads > 1 && 
         ForEachTraits<FunctionTy>::NeedsAborts)
        ||
        (networkHostNum > 1 && LL::getTID() == 0))
      go<true>();
    else
      go<false>();
  }
};


template<typename WLTy, typename RangeTy, typename FunctionTy>
void for_each_impl(const RangeTy& range, FunctionTy f, const char* loopname) {
  typedef typename RangeTy::value_type T;
  typedef ForEachWork<WLTy, T, FunctionTy> WorkTy;

  assert(!inGaloisForEach);

  inGaloisForEach = true;

  WorkTy W(f, loopname);
  //RunCommand init(std::bind(&WorkTy::template AddInitialWork<RangeTy>, std::ref(W), range));
  RunCommand w[5] = {
    std::bind(&WorkTy::initThread, std::ref(W)),
    std::bind(&WorkTy::template AddInitialWork<RangeTy>, std::ref(W), range), 
    std::ref(getSystemBarrier()),
    std::ref(W),
    std::ref(getSystemBarrier())
  };
  getSystemThreadPool().run(&w[0], &w[5], activeThreads);
  inGaloisForEach = false;
}

template<typename FunctionTy>
struct WOnEach {
  FunctionTy& origFunction;
  WOnEach(FunctionTy& f): origFunction(f) { }
  void operator()(void) {
    FunctionTy fn(origFunction);
    fn(LL::getTID(), activeThreads);   
  }
};

template<typename FunctionTy>
void on_each_impl(FunctionTy fn, const char* loopname = 0) {
  RunCommand w[2] = {WOnEach<FunctionTy>(fn),
		     std::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2], activeThreads);
}

} // end namespace anonymous

void preAlloc_impl(int num);

} // end namespace Runtime
} // end namespace Galois

#endif


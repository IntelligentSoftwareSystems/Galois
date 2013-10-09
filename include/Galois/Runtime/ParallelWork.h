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
  std::deque<value_type> queues;
  LL::SimpleLock<true> Lock;
  std::map<fatPointer, std::set<value_type> > waiting;
  std::map<value_type, std::set<fatPointer> > holding;

  void arrive(fatPointer ptr) {
    //    assert(waiting.count(ptr));
    std::lock_guard<LL::SimpleLock<true>> lg(Lock);
    //LL::gDebug("Arrive notification ", ptr.first, ",", ptr.second, ": ", waiting.count(ptr));
    for (auto ii = waiting.lower_bound(ptr), ee = waiting.upper_bound(ptr); ii != ee; ++ii)
      queues.push(ii->second);
    waiting.erase(ptr);
  }

  int size() {
    return waiting.size();
  }

public:

  void push(value_type val) {
    LL::SLguard lg(Lock);
    queues.push_back(val);
  }

  void push(value_type val, fatPointer ptr) {
    //    std::cerr << "pushing " << val << " to " << ptr.first << " " << ptr.second << " with " << waiting.size() << "\n";
    bool skipNotify;
    auto& dir = getSystemDirectory();
    { //limit scope of guard
      LL::SLguard lg(Lock);
      skipNotify =  waiting[ptr].size();
      waiting[ptr].insert(val);
      if (holding[val].insert(ptr).second) //first insert, inc contended
        dir.setContended(ptr);
      queues.push_back(val);
    }
    //    if (!skipNotify)
    //      dir.notifyWhenAvailable(ptr, std::bind(&RemoteAbortHandler::arrive, this, std::placeholders::_1));
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> retval;
    if (!queues.empty()) {
      retval = queues.front();
      queues.pop_front();
    }
    return retval;
  }

  void commit(value_type val) {
    std::set<fatPointer> held;
    { //bound lock
      LL::SLguard lg(Lock);
      if (holding.count(val)) {
        holding[val].swap(held);
        holding.erase(val);
      }
    }
    //std::cerr << "commit remote " << val << "\n";
    auto& dir = getSystemDirectory();
    for (auto key : held)
      dir.clearContended(key);
  }

  void dump() {
    LL::gDebug("RAQ: waiting size ", waiting.size(), ", holding size ", holding.size(), ", queue size ", queues.size());
    if (waiting.size() < 10) {
      for (auto& k : waiting)
        LL::gDebug("RAQ waiting on: ", k.first.first, ",", k.first.second);
      for (auto& k : waiting)
        getSystemDirectory().queryObj(k.first);
    }
  }

  //doesn't check queue, just hidden work
  bool empty() {
    static std::chrono::system_clock::time_point oldtime = std::chrono::system_clock::now();
    std::lock_guard<LL::SimpleLock<true>> lg(Lock);
    // unsigned num = getSystemDirectory().countContended();
    // if (!waiting.empty() || num) {
    //   std::unordered_set<fatPointer, ptrHash> items;
    //   if (num < 200) {
    //     for (auto& a : waiting)
    //       items.insert(a.first);
    //   }
    //   std::cerr << NetworkInterface::ID << " Waiting on " << items.size() << " holding " << num << "\n";
    // }
    std::chrono::duration<int> onesec(1);
    if (std::chrono::system_clock::now() - oldtime > onesec) {
      oldtime = std::chrono::system_clock::now();
      getSystemDirectory().dump();
      dump();
    }
    return holding.empty();
  }
};


template<typename value_type, typename FunctionTy, typename WLTy>
class ThreadLocalExec {
  SimpleRuntimeContext context;
  UserContextAccess<value_type> facing;
  FunctionTy func;
  WLTy& wl;
  unsigned long conflicts; // all conflicts
  unsigned long iterations; // all iterations
  unsigned long remote; // remote conflicts

  const char* loopname;

  const bool NeedsStats  = ForEachTraits<FunctionTy>::NeedsStats;
  const bool NeedsPIA    = ForEachTraits<FunctionTy>::NeedsPIA;
  const bool NeedsPush   = ForEachTraits<FunctionTy>::NeedsPush;
  const bool NeedsAborts = ForEachTraits<FunctionTy>::NeedsAborts;


  inline void resetAlloc() { if (NeedsPIA) facing.resetAlloc(); }
  inline void inc_stat_iterations() { if (NeedsStats) ++iterations; }
  inline void inc_stat_conflicts()  { if (NeedsStats) ++conflicts;  }
  inline void context_start()  { if (NeedsAborts) context.startIteration();  }
  inline void context_commit() { if (NeedsAborts) context.commitIteration(); }
  inline void context_abort() { context.cancelIteration(); }

  inline void push_work() {
    if (NeedsPush) {
      auto ii = facing.getPushBuffer().begin();
      auto ee = facing.getPushBuffer().end();
      if (ii != ee) {
        wl.push(ii,ee);
        facing.resetPushBuffer();
      }
    }
  }

  inline void push_reset() { if (NeedsPush) facing.resetPushBuffer(); }

  void fastPushBack(typename UserContextAccess<value_type>::PushBufferTy& x) {
    wl.push(x.begin(), x.end());
    x.clear();
  }

public:
  ThreadLocalExec(FunctionTy& _func, WLTy& _wl, const char* ln)
    : func(_func), wl(_wl), 
      conflicts(0), iterations(0), remote(0), loopname(ln) {
    if (NeedsAborts)
      setThreadContext(&context);
    if (false && NeedsPush && !NeedsAborts)
      facing.setFastPushBack(std::bind(&ThreadLocalExec::fastPushBack, this, std::placeholders::_1));
  }

  ~ThreadLocalExec() {
    if (NeedsAborts)
      setThreadContext(nullptr);
    if (NeedsStats) {
      reportStat(loopname, "Conflicts", conflicts);
      reportStat(loopname, "Iterations", iterations);
      reportStat(loopname, "RemoteEx", remote);
    }
  }

  inline void inc_stat_remote()     { if (NeedsStats) ++remote;     }

  inline void execItem(value_type& p) {
    inc_stat_iterations();
    context_start();
    if (NeedsAborts) {
      try {
        func(p, facing.data());
      } catch (...) {
        context_abort();
        inc_stat_conflicts();
        push_reset();
        resetAlloc();
        throw;
      }
    } else {
      func(p, facing.data());
    }
    push_work();
    context_commit();
    resetAlloc();
    //remote_aborted.commit(*p);
  }
};

template<class WorkListTy, class T, class FunctionTy>
class ForEachWork {
protected:
  typedef T value_type;
  typedef typename WorkListTy::template retype<value_type> WLTy;
  typedef ThreadLocalExec<value_type, FunctionTy, WLTy> ThreadLocalData;

  WLTy wl;
  FunctionTy& origFunction;
  const char* loopname;

  TerminationDetection& term;
  AbortHandler<value_type> aborted;
  RemoteAbortHandler<value_type> remote_aborted;
  LL::CacheLineStorage<bool> broke;

  template<unsigned limit, typename WL>
  GALOIS_ATTRIBUTE_NOINLINE
  bool runQueue(ThreadLocalData& tld, WL& lwl, bool recursiveAbort, bool remoteAbort) {
    boost::optional<value_type> p = lwl.pop();
    if (p) {
      if (ForEachTraits<FunctionTy>::NeedsAborts) {
        unsigned runlimit = limit;
        do {
          try {
            tld.execItem(*p);
            if (remoteAbort)
              remote_aborted.commit(*p);
          } catch (const conflict_ex& ex) {
            if (remoteAbort) //if in the remote queue, stay in the remote queue
              remote_aborted.push(*p);
            else
              aborted.push(recursiveAbort, *p);
          } catch (const remote_ex& ex) {
            tld.inc_stat_remote();
            //aborted.push(recursiveAbort, *p);
            remote_aborted.push(*p, ex.ptr);
          }
          if (limit)
            --runlimit;
        } while ((limit == 0 || runlimit != 0) && (p = lwl.pop()));
        return true;
      } else {
        do {
          tld.execItem(*p);
        } while ((p = lwl.pop()));
        return true;
      }
    }
    return false;
  }

  template<bool checkAbort>
  void go() {
    //Thread Local Data goes on the local stack
    //to be NUMA friendly
    ThreadLocalData tld(origFunction, wl, loopname);

    bool didWork;
    try {
      do {
        didWork = false;
        //Run some iterations
        didWork = runQueue<checkAbort ? 1 /*32*/ : 0>(tld, wl, false, false);
        //Check for break
        if (ForEachTraits<FunctionTy>::NeedsBreak && broke.data)
          break;
        //Check for abort, also guards random network work
        if (checkAbort) {
          didWork |= runQueue<1>(tld, aborted, true, false);
          if (LL::getTID() == 0) {
            didWork |= runQueue<1>(tld, remote_aborted, true, true);
            didWork |= !remote_aborted.empty();
            doNetworkWork();
          } else {
            while (getSystemNetworkInterface().handleReceives()) {}
          }
        }
        // update node color and prop token
        term.localTermination(didWork);
      } while (!term.globalTermination());
    } catch (const break_ex& ex) {
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
    // if ((LL::isPackageLeaderForSelf(LL::getTID()) &&
    //      activeThreads > 1 && 
    //      ForEachTraits<FunctionTy>::NeedsAborts)
    //     ||
    //     (NetworkInterface::Num > 1 && LL::getTID() == 0))
    if (LL::isPackageLeaderForSelf(LL::getTID()) &&
        ((activeThreads > 1 && ForEachTraits<FunctionTy>::NeedsAborts)
         ||
         (NetworkInterface::Num > 1)))
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
  trace_loop_start(loopname);
  getSystemThreadPool().run(&w[0], &w[5], activeThreads);
  trace_loop_end(loopname);
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
  if (inGaloisForEach)
    GALOIS_DIE("Nested for_each not supported");

  inGaloisForEach = true;
  RunCommand w[2] = {WOnEach<FunctionTy>(fn),
		     std::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2], activeThreads);
  inGaloisForEach = false;
}

//! on each executor with simple barrier.
template<typename FunctionTy>
void on_each_simple_impl(FunctionTy fn, const char* loopname = 0) {
  if (inGaloisForEach)
    GALOIS_DIE("Nested for_each not supported");

  inGaloisForEach = true;
  Barrier* b = createSimpleBarrier();
  b->reinit(activeThreads);
  RunCommand w[2] = {WOnEach<FunctionTy>(fn),
		     std::ref(*b)};
  getSystemThreadPool().run(&w[0], &w[2], activeThreads);
  delete b;
  inGaloisForEach = false;
}

} // end namespace anonymous

void preAlloc_impl(int num);

} // end namespace Runtime
} // end namespace Galois

#endif


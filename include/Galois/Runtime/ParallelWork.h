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

#include <algorithm>
#include <functional>
#include <iostream>

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

  std::map<fatPointer, std::vector<value_type>> remote;
  LL::SimpleLock<true> Lock;

  void objArive(std::pair<uint32_t, Lockable*> ptr) {
    std::lock_guard<LL::SimpleLock<true>> L(Lock);
    for (auto ii = remote[ptr].begin(), ee = remote[ptr].end(); ii != ee; ++ii)
      push(false, *ii);
    remote.erase(ptr);
  }

  void dump() {
    for (auto ii = remote.begin(), ee = remote.end(); ii != ee; ++ii)
      LL::gInfo("waiting ", ii->first.first, ",", ii->first.second);
  }

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

  void push(bool recursiveAbort, value_type val, fatPointer p) {
    Lock.lock();
    bool skipPush = remote.count(p);
    remote[p].push_back(val);
    Lock.unlock();
    std::function<void(fatPointer)> f(std::bind(&AbortHandler::objArive, this, std::placeholders::_1));
 
    if (!skipPush) {
      if (p.first == networkHostID)
        getSystemLocalDirectory().notifyWhenAvailable(p, f);
      else
        getSystemRemoteDirectory().notifyWhenAvailable(p, f);
    }
  }

  bool hasHiddenWork() {
    std::lock_guard<LL::SimpleLock<true>> L(Lock);
    // if (!remote.empty())
    //   dump();
    return !remote.empty();
  }

};

template<class WorkListTy, class T, class FunctionTy>
class ForEachWork {
protected:
  typedef T value_type;
  typedef typename WorkListTy::template retype<value_type> WLTy;
  typedef gdeque<std::pair<std::pair<value_type,Lockable*>,std::pair<unsigned,bool> > > ReceivedList;

  struct ThreadLocalData {
    FunctionTy function;
    UserContextAccess<value_type> facing;
    SimpleRuntimeContext context;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    ThreadLocalData(const FunctionTy& fn, const char* ln): function(fn), stat(ln) {}
  };

  WLTy wl;
  FunctionTy& origFunction;
  const char* loopname;

  TerminationDetection& term;
  AbortHandler<value_type> aborted;
  LL::CacheLineStorage<bool> broke;

  inline void commitIteration(ThreadLocalData& tld) {
    if (ForEachTraits<FunctionTy>::NeedsPush) {
      auto ii = tld.facing.getPushBuffer().begin();
      auto ee = tld.facing.getPushBuffer().end();
      if (ii != ee) {
	wl.push(ii,ee);
	tld.facing.resetPushBuffer();
      }
    }
    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      tld.context.commit_iteration();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(value_type val, ThreadLocalData& tld) {
    assert(ForEachTraits<FunctionTy>::NeedsAborts);
    tld.context.cancel_iteration();
    tld.stat.inc_conflicts(); //Class specialization handles opt
    //clear push buffer
    if (ForEachTraits<FunctionTy>::NeedsPush)
      tld.facing.resetPushBuffer();
    //reset allocator
    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();
  }

  inline void execItem(boost::optional<value_type>& p, ThreadLocalData& tld) {
    tld.stat.inc_iterations(); //Class specialization handles opt
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      tld.context.start_iteration();
    tld.function(*p, tld.facing.data());
    commitIteration(tld);
    doNetworkWork();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleBreak(ThreadLocalData& tld) {
    commitIteration(tld);
    broke.data = true;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleLocalAbort(value_type val, ThreadLocalData& tld, Lockable* L, bool recursiveAbort) {
    abortIteration(val, tld);
    aborted.push(recursiveAbort, val);
    //FIXME: do a reverse lookup in the remote directory to see if
    //this is a remote object or not.  In the mean time, do nothing.
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleRemoteAbort(value_type val, ThreadLocalData& tld, uint32_t hostid, Lockable* item, bool recursiveAbort) {
    abortIteration(val, tld);
    aborted.push(recursiveAbort, val, std::make_pair(hostid, item));
    //FIXME: implement all the extra stuff
  }

  template<typename WL>
  bool runQueueSimple(ThreadLocalData& tld, WL& lwl) {
    boost::optional<value_type> p = lwl.pop();
    if (p) {
      do {
	execItem(p, tld);
      } while ((p = lwl.pop()));
      return true;
    }
    return false;
  }

  template<bool limit, typename WL>
  bool runQueue(ThreadLocalData& tld, WL& lwl, bool recursiveAbort) {
    boost::optional<value_type> p; 
    try {
      p = lwl.pop();
      if (p) {
	unsigned runlimit = 32; //FIXME: don't hardcode
	do {
	  execItem(p, tld);
	  if (limit)
            --runlimit;
	} while ((limit == 0 || runlimit != 0) && (p = lwl.pop()));
	return true;
      }
      return false;
    } catch (const conflict_ex& ex) {
      handleLocalAbort(*p, tld, ex.obj, recursiveAbort);
    } catch (const remote_ex& ex) {
      handleRemoteAbort(*p, tld, ex.owner, ex.actual, recursiveAbort);
    } catch (const break_ex&) {
      handleBreak(tld);
      return false;
    }
    return true;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool runAborts(ThreadLocalData& tld) {
    bool r = runQueue<false>(tld, aborted, true);
    while (r && runQueue<false>(tld, aborted, true)) {};
    return r || aborted.hasHiddenWork();
  }

  void fastPushBack(typename UserContextAccess<value_type>::PushBufferTy& x) {
    wl.push(x.begin(), x.end());
    x.clear();
  }

  template<bool checkAbort>
  void go() {
    //Thread Local Data goes on the local stack
    //to be NUMA friendly
    ThreadLocalData tld(origFunction, loopname);
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      setThreadContext(&tld.context);
    if (false && ForEachTraits<FunctionTy>::NeedsPush && !ForEachTraits<FunctionTy>::NeedsAborts) {
      tld.facing.setFastPushBack(std::bind(&ForEachWork::fastPushBack, std::ref(*this), std::placeholders::_1));
    }
    bool didAnyWork;
    do {
      didAnyWork = false;
      bool didWork;
      do {
        didWork = false;
        //Run some iterations
        if (ForEachTraits<FunctionTy>::NeedsBreak || ForEachTraits<FunctionTy>::NeedsAborts)
          didWork = runQueue<checkAbort | ForEachTraits<FunctionTy>::NeedsBreak>(tld, wl, false);
        else //No try/catch
          didWork = runQueueSimple(tld, wl);
        //Check for break
        if (ForEachTraits<FunctionTy>::NeedsBreak && broke.data)
          break;
        //Check for abort
        if (checkAbort)
          didWork |= runAborts(tld);
	didAnyWork |= didWork;
	doNetworkWork();
      } while (didWork);
      if (ForEachTraits<FunctionTy>::NeedsBreak && broke.data)
        break;
      // update node color and prop token
      term.localTermination(didAnyWork);
    } while (!term.globalTermination());
    //FIXME: termination needs to happen if stealing does, not if the above condtion holds
    //} while (!term.globalTermination());

    if (ForEachTraits<FunctionTy>::NeedsAborts)
      setThreadContext(0);
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
  getSystemRemoteObjects().clear();

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


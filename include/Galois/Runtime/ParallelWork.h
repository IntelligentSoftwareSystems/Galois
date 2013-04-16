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

template<class WorkListTy, class T, class FunctionTy>
class ForEachWork {
protected:
  typedef T value_type;
  typedef typename WorkListTy::template retype<value_type> WLTy;
  typedef WorkList::GFIFO<value_type> AbortedList;
  typedef WorkList::GFIFO<std::pair<value_type,Lockable*> > ReceivedList;

  struct ThreadLocalData {
    FunctionTy function;
    UserContextAccess<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    ThreadLocalData(const FunctionTy& fn, const char* ln): function(fn), stat(ln) {}
  };

  WLTy wl;
  FunctionTy& origFunction;
  const char* loopname;

  TerminationDetection& term;
  PerPackageStorage<AbortedList> aborted;
  ReceivedList received;
  LL::CacheLineStorage<bool> broke;

  inline void commitIteration(ThreadLocalData& tld) {
    if (ForEachTraits<FunctionTy>::NeedsPush) {
      wl.push(tld.facing.getPushBuffer().begin(),
              tld.facing.getPushBuffer().end());
      tld.facing.resetPushBuffer();
    }
    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.commit_iteration();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(value_type val, ThreadLocalData& tld, bool recursiveAbort, bool local) {
    assert(ForEachTraits<FunctionTy>::NeedsAborts);
    tld.cnx.cancel_iteration();
    tld.stat.inc_conflicts(); //Class specialization handles opt
    if (local) {
      if (recursiveAbort)
        aborted.getRemote(LL::getLeaderForPackage(LL::getPackageForSelf(LL::getTID()) / 2))->push(val);
      else
        aborted.getLocal()->push(val);
    }
    //clear push buffer
    if (ForEachTraits<FunctionTy>::NeedsPush)
      tld.facing.resetPushBuffer();
    //reset allocator
    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();
  }

  inline void doProcess(boost::optional<value_type>& p, ThreadLocalData& tld) {
    tld.stat.inc_iterations(); //Class specialization handles opt
 // may havereceived remote objects already locked in the context
 /*
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.start_iteration();
  */
    tld.function(*p, tld.facing.data());
    commitIteration(tld);

    if ((Distributed::networkHostNum > 1) && (LL::getTID() == 0)) {
      getSystemLocalDirectory().makeProgress();
      getSystemRemoteDirectory().makeProgress();
      Distributed::getSystemNetworkInterface().handleReceives();
   }
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleBreak(ThreadLocalData& tld) {
    commitIteration(tld);
    broke.data = true;
  }

  bool runQueueSimple(ThreadLocalData& tld) {
    bool workHappened = false;
    boost::optional<value_type> p = wl.pop();
    if (p)
      workHappened = true;
    while (p) {
      doProcess(p, tld);
      p = wl.pop();
    }
    return workHappened;
  }

  void resch_recv(value_type val, Lockable *L) {
    received.push(std::make_pair(val,L));
  }

  template<typename WL>
  boost::optional<value_type> getProcess(WL& lwl) {
    boost::optional<value_type> p;
    boost::optional<std::pair<value_type,Lockable*> > rp;
    if (LL::getTID() == 0) {
      rp = received.pop();
      if(rp) {
        Lockable* L;
        L = rp->second;
        getAbortCnx().swap_acquire(L,getThreadContext());
        p = rp->first;
      }
      else {
	p = lwl.pop();
      }
    }
    else {
      p = lwl.pop();
    }
    return p;
  }

  template<bool limit, typename WL>
  bool runQueue(ThreadLocalData& tld, WL& lwl, bool recursiveAbort) {
    bool workHappened = false;
    boost::optional<value_type> p;
    p = getProcess(lwl);
    unsigned num = 0;
    if (p)
      workHappened = true;
    try {
      while (p) {
	doProcess(p, tld);
	if (limit) {
	  ++num;
	  if (num == 32)
	    break;
	}
        p = getProcess(lwl);
      }
    } catch (const conflict_ex& ex) {
      Distributed::getSystemLocalDirectory().recall(ex.obj, false);
      abortIteration(*p, tld, recursiveAbort, true);
    } catch (const remote_ex& ex) {
      // object is remote - only for Remote Directory now
      assert(ex.owner != Distributed::networkHostID);
      // add to the list of remoteObjects
      Distributed::getSystemRemoteObjects().insert(ex.obj,std::bind(&ForEachWork::resch_recv,this,*p,ex.obj));
      abortIteration(*p, tld, recursiveAbort, false);
    } catch (const break_ex&) {
      handleBreak(tld);
      return false;
    }

    if ((Distributed::networkHostNum > 1) && (LL::getTID() == 0)) {
      getSystemLocalDirectory().makeProgress();
      getSystemRemoteDirectory().makeProgress();
      Distributed::getSystemNetworkInterface().handleReceives();
    }
    return workHappened;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool handleAborts(ThreadLocalData& tld) {
    return runQueue<false>(tld, *aborted.getLocal(), true);
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
      setThreadContext(&tld.cnx);
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
          didWork = runQueue<checkAbort || ForEachTraits<FunctionTy>::NeedsBreak>(tld, wl, false);
        else //No try/catch
          didWork = runQueueSimple(tld);
        //Check for break
        if (ForEachTraits<FunctionTy>::NeedsBreak && broke.data)
          break;
        //Check for abort
        if (checkAbort)
          didWork |= handleAborts(tld);
	didAnyWork |= didWork;

	if ((Distributed::networkHostNum > 1) && (LL::getTID() == 0)) {
	  getSystemLocalDirectory().makeProgress();
	  getSystemRemoteDirectory().makeProgress();
	  Distributed::getSystemNetworkInterface().handleReceives();
	}

      } while (didWork);
      if (ForEachTraits<FunctionTy>::NeedsBreak && broke.data)
        break;

      //update node color and prop token
      term.localTermination(didAnyWork);
    } while ((ForEachTraits<FunctionTy>::NeedsPush 
	     ||ForEachTraits<FunctionTy>::NeedsBreak
	     ||ForEachTraits<FunctionTy>::NeedsAborts)
	     && !term.globalTermination());

    if (ForEachTraits<FunctionTy>::NeedsAborts)
      setThreadContext(0);
  }

public:
  ForEachWork(FunctionTy& f, const char* l): origFunction(f), loopname(l), term(getSystemTermination()), broke(false) { }
  
  template<typename W>
  ForEachWork(W& w, FunctionTy& f, const char* l): wl(w), origFunction(f), loopname(l), term(getSystemTermination()), broke(false) { }

  template<typename RangeTy>
  void AddInitialWork(const RangeTy& range) {
    wl.push_initial(range);
  }

  void initThread(void) {
    term.initializeThread();
  }

  // in the distributed case even with 1 thread there can be aborts
  void operator()() {
    if (LL::isPackageLeaderForSelf(LL::getTID()) &&
	(activeThreads > 1 || Distributed::networkHostNum > 1) && 
	ForEachTraits<FunctionTy>::NeedsAborts)
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
  Distributed::getSystemRemoteObjects().clear();

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


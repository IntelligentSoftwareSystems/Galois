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
#include "Galois/Statistic.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/WorkList/GFifo.h"

#include <algorithm>
#include <functional>
#include <memory>

namespace Galois {
//! Internal Galois functionality - Use at your own risk.
namespace Runtime {

namespace {

template<bool Enabled> 
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
  struct Item { value_type val; int retries; };

  typedef WorkList::GFIFO<Item> AbortedList;
  PerThreadStorage<AbortedList> queues;
  bool useBasicPolicy;
  
  /**
   * Policy: serialize via tree over packages.
   */
  void basicPolicy(const Item& item) {
    unsigned tid = LL::getTID();
    unsigned package = LL::getPackageForSelf(tid);
    queues.getRemote(LL::getLeaderForPackage(package / 2))->push(item);
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
    
    unsigned tid = LL::getTID();
    unsigned package = LL::getPackageForSelf(tid);
    unsigned leader = LL::getLeaderForPackage(package);
    if (tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.getRemote(next)->push(item);
    } else {
      queues.getRemote(LL::getLeaderForPackage(package / 2))->push(item);
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
    
    unsigned tid = LL::getTID();
    unsigned package = LL::getPackageForSelf(tid);
    unsigned leader = LL::getLeaderForPackage(package);
    if (retries < 5 && tid != leader) {
      unsigned next = leader + (tid - leader) / 2;
      queues.getRemote(next)->push(item);
    } else {
      queues.getRemote(LL::getLeaderForPackage(package / 2))->push(item);
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
    useBasicPolicy = LL::getMaxPackages() > 2;
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

template<typename value_type>
class RemoteAbortHandler {
  std::multimap<fatPointer, value_type> waiting_on;
  std::map<value_type, std::set<fatPointer> > contended_set;

  typedef WorkList::GFIFO<value_type> AbortedList;
  AbortedList ready;
  
  LL::SimpleLock lock;
  
  void notify_impl(fatPointer ptr) {
    std::lock_guard<LL::SimpleLock> lg(lock);
    assert(waiting_on.count(ptr)); //push should only set up notify once per pointer
    auto p = waiting_on.equal_range(ptr);
    auto p2 = p;
    while (p.first != p.second) {
      ready.push(p.first->second);
      ++p.first;
    }
    waiting_on.erase(p2.first, p2.second);
  }

public:
  void push(const value_type& val, fatPointer ptr, 
            void (RemoteDirectory::*rfetch) (fatPointer, ResolveFlag),
            void (LocalDirectory::*lfetch) (fatPointer, ResolveFlag)) {
    std::lock_guard<LL::SimpleLock> lg(lock);

    //Set contended and fetch
    if (ptr.isLocal())
      (getLocalDirectory().*(lfetch))(ptr, RW);
    else
      (getRemoteDirectory().*(rfetch))(ptr, RW);
    contended_set[val].insert(ptr);

    //already waiting on this pointer
    if (waiting_on.count(ptr)) {
      auto p = waiting_on.equal_range(ptr);
      if (std::count(p.first, p.second, std::pair<const fatPointer, value_type>(ptr,val)))
        return; //already waiting
      waiting_on.insert(std::make_pair(ptr, val));
      return;
    }

    //Likely currently remote
    bool mustWait;
    auto fnotify = [this] (fatPointer p) { this->notify_impl(p); };
    if (ptr.isLocal())
      mustWait = getLocalDirectory().notify(ptr, RW, fnotify);
    else
      mustWait = getRemoteDirectory().notify(ptr, RW, fnotify);

    //Certainly Remote
    if (mustWait) {
      waiting_on.insert(std::make_pair(ptr, val));
    } else { //Present
      ready.push(val);
    }
  }

  void push(const value_type& val, Lockable* ptr) {
    ready.push(val);
  }

  void commit(const value_type& val) {
    std::lock_guard<LL::SimpleLock> lg(lock);
    auto p = contended_set[val];
    assert(!p.empty());
    for (auto ptr : p)
      if (ptr.isLocal()) {
        getLocalDirectory().clearContended(ptr);
      } else {
        getRemoteDirectory().clearContended(ptr);
      }
    contended_set.erase(val);
  }

  decltype(ready.pop()) pop() {
    return ready.pop();
  }

  bool noHiddenWork() {
    std::lock_guard<LL::SimpleLock> lg(lock);
    return waiting_on.empty();
  }

};

template<class WorkListTy, class T, class FunctionTy>
class ForEachWork {
protected:
  typedef T value_type;
  typedef typename WorkListTy::template retype<value_type>::type WLTy;
  struct ThreadLocalData {
    FunctionTy function;
    UserContextAccess<value_type> facing;
    SimpleRuntimeContext ctx;
    ResolveCache rc;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    ThreadLocalData(const FunctionTy& fn, const char* ln): function(fn), stat(ln) {}

    inline void commitIteration(WLTy& wl) {
      if (ForEachTraits<FunctionTy>::NeedsPush) {
        auto ii = facing.getPushBuffer().begin();
        auto ee = facing.getPushBuffer().end();
        if (ii != ee) {
          wl.push(ii, ee);
          facing.resetPushBuffer();
        }
      }
      if (ForEachTraits<FunctionTy>::NeedsPIA)
        facing.resetAlloc();
      if (ForEachTraits<FunctionTy>::NeedsAborts)
        ctx.commitIteration();
    }

    GALOIS_ATTRIBUTE_NOINLINE
    void abortIteration() {
      assert(ForEachTraits<FunctionTy>::NeedsAborts);
      ctx.cancelIteration();
      stat.inc_conflicts(); //Class specialization handles opt
      //clear push buffer
      if (ForEachTraits<FunctionTy>::NeedsPush)
        facing.resetPushBuffer();
      //reset allocator
      if (ForEachTraits<FunctionTy>::NeedsPIA)
        facing.resetAlloc();
    }
    
    inline void startIteration() {
      stat.inc_iterations();
      if (ForEachTraits<FunctionTy>::NeedsAborts)
        ctx.startIteration();
      rc.reset();
    }
  };

  // NB: Place dynamically growing wl after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage

  RemoteAbortHandler<value_type> aborted; 
  TerminationDetection& term;

  WLTy wl;
  FunctionTy& origFunction;
  const char* loopname;
  bool broke;

  template<int limit, bool inAborted, typename WL>
  bool runQueue(ThreadLocalData& tld, WL& lwl) {
    bool workHappened = false;
    Galois::optional<value_type> p = lwl.pop();
    unsigned num = 0;
    if (p)
      workHappened = true;
    while (p) {
      try {
        tld.startIteration();
        tld.function(*p, tld.facing.data());
        tld.commitIteration(wl);
        if (ForEachTraits<FunctionTy>::NeedsAborts && inAborted)
          aborted.commit(*p);
      } catch (const remote_ex& ex) {
        //      std::cout << "R";
        tld.abortIteration();
        aborted.push(*p, ex.ptr, ex.rfetch, ex.lfetch);
      } catch (const conflict_ex& ex) {
        //      std::cout << "L";
        tld.abortIteration();
        aborted.push(*p, ex.ptr);
      }
      if (limit) {
        ++num;
        if (num == limit)
          break;
      }
      p = lwl.pop();
    }
    return workHappened;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool handleAborts(ThreadLocalData& tld) {
    return runQueue<8, true>(tld, aborted) || !aborted.noHiddenWork();
  }

  void fastPushBack(typename UserContextAccess<value_type>::PushBufferTy& x) {
    wl.push(x.begin(), x.end());
    x.clear();
  }

  template<typename WL>
  bool checkEmpty(WL&, ThreadLocalData&, ...) { return true; }

  template<typename WL>
  auto checkEmpty(WL& wl, ThreadLocalData& tld, int) -> decltype(wl.empty(), bool()) {
    return wl.empty();
  }

  template<bool couldAbort, bool isLeader>
  void go() {
    // Thread-local data goes on the local stack to be NUMA friendly
    ThreadLocalData tld(origFunction, loopname);
    tld.facing.setBreakFlag(&broke);
    if (couldAbort)
      setThreadContext(&tld.ctx);
    if (ForEachTraits<FunctionTy>::NeedsPush && !couldAbort)
      tld.facing.setFastPushBack(
          std::bind(&ForEachWork::fastPushBack, std::ref(*this), std::placeholders::_1));

    bool didWork;
    while (true) {
      do {
        didWork = false;
        //Run some iterations
        if (isLeader)
          didWork = runQueue<64, false>(tld, wl);
        else
          didWork = runQueue<ForEachTraits<FunctionTy>::NeedsBreak ? 64 : 0, false>(tld, wl);
        // Check for abort
        if (couldAbort)
          didWork |= handleAborts(tld);
        if (isLeader)
          doNetworkWork();
        // Update node color and prop token
        term.localTermination(didWork);
        doNetworkWork();
      } while (!term.globalTermination() && (!ForEachTraits<FunctionTy>::NeedsBreak || !broke));

      if (checkEmpty(wl, tld, 0))
        break;
      if (ForEachTraits<FunctionTy>::NeedsBreak && broke)
        break;
      initThread();
      getSystemBarrier().wait();
    }

    if (couldAbort)
      setThreadContext(0);
  }

public:
  ForEachWork(FunctionTy& f, const char* l): term(getSystemTermination()), origFunction(f), loopname(l), broke(false) { }
  
  template<typename W>
  ForEachWork(W& w, FunctionTy& f, const char* l): term(getSystemTermination()), wl(w), origFunction(f), loopname(l), broke(false) { }

  template<typename RangeTy>
  void AddInitialWork(const RangeTy& range) {
    wl.push_initial(range);
  }

  void initThread() {
    term.initializeThread();
  }

  // in the distributed case even with 1 thread there can be aborts
  void operator()() {
    bool isLeader = LL::isPackageLeaderForSelf(LL::getTID());
    bool couldAbort = ForEachTraits<FunctionTy>::NeedsAborts && activeThreads > 1;
    couldAbort = true;
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
constexpr auto has_with_iterator(int) -> decltype(std::declval<typename WLTy::template with_iterator<int*>::type>(), bool()) {
  return true;
}

template<typename>
constexpr bool has_with_iterator(...) {
  return false;
}

template<typename WLTy, typename IterTy, typename Enable = void>
struct reiterator {
  typedef WLTy type;
};

template<typename WLTy, typename IterTy>
struct reiterator<WLTy, IterTy, typename std::enable_if<has_with_iterator<WLTy>(0)>::type> {
  typedef typename WLTy::template with_iterator<IterTy>::type type;
};

template<typename WLTy, typename RangeTy, typename FunctionTy>
void for_each_impl(const RangeTy& range, FunctionTy f, const char* loopname) {
  typedef typename RangeTy::value_type T;
  typedef typename reiterator<WLTy, typename RangeTy::iterator>::type WLNewTy;
  typedef ForEachWork<WLNewTy, T, FunctionTy> WorkTy;

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  WorkTy W(f, loopname);
  getLocalDirectory().resetStats();
  getRemoteDirectory().resetStats();
  trace("Loop start %\n", loopname);
  beginLoopSampling(loopname);
  getSystemThreadPool().run(activeThreads, 
    std::bind(&WorkTy::initThread, std::ref(W)),
    std::bind(&WorkTy::template AddInitialWork<RangeTy>, std::ref(W), range), 
    std::ref(getSystemBarrier()),
    std::ref(W)
  );
  endLoopSampling(loopname);
  trace("Loop end %\n", loopname);
  getLocalDirectory().reportStats(loopname);
  getRemoteDirectory().reportStats(loopname);
  inGaloisForEach = false;
}

} // end namespace anonymous

void preAlloc_impl(int num);

} // end namespace Runtime
} // end namespace Galois

#endif


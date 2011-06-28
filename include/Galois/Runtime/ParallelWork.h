/** Galois scheduler and runtime -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
#include <algorithm>
#include <numeric>
#include <sstream>
#include <math.h>
#include "Galois/TypeTraits.h"
#include "Galois/Executable.h"
#include "Galois/Mem.h"

#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/DebugWorkList.h"
#include "Galois/Runtime/Termination.h"

#ifdef GALOIS_VTUNE
#include "ittnotify.h"
#endif

namespace GaloisRuntime {

//Handle Runtime Conflict Detection
template<bool SRC_ACTIVE>
class SimpleRuntimeContextHandler;

template<>
class SimpleRuntimeContextHandler<true> {
  SimpleRuntimeContext src;
public:
  void start_iteration() {
    src.start_iteration();
  }
  void cancel_iteration() {
    src.cancel_iteration();
  }
  void commit_iteration() {
    src.commit_iteration();
  }
  void start_parallel_region() {
    setThreadContext(&src);
  }
  void end_parallel_region() {
    setThreadContext(0);
  }
};

template<>
class SimpleRuntimeContextHandler<false> {
public:
  void start_iteration() {}
  void cancel_iteration() {}
  void commit_iteration() {}
  void start_parallel_region() {}
  void end_parallel_region() {}
};

//Handle Statistic gathering
template<bool STAT_ACTIVE>
class StatisticHandler;

template<>
class StatisticHandler<true> {
  unsigned long conflicts;
  unsigned long iterations;
public:
  StatisticHandler() :conflicts(0), iterations(0) {}
  void inc_iterations() {
    ++iterations;
  }
  void inc_conflicts() {
    ++conflicts;
  }
  void report_stat() const {
    reportStat("Conflicts", conflicts);
    reportStat("Iterations", iterations);
  }
  void merge_stat(const StatisticHandler& rhs) {
    conflicts += rhs.conflicts;
    iterations += rhs.iterations;
  }
  struct stat_sum {
    std::vector<long> list;
    void add(StatisticHandler& x) {
      list.push_back(x.iterations);
    }
    void done() {
      GaloisRuntime::summarizeList("IterationDistribution", 
				   &list[0], &list[list.size()]);
    }
    int num() {
      return GaloisRuntime::getSystemThreadPool().getActiveThreads();
    }
  };
};

template<>
class StatisticHandler<false> {
public:
  void inc_iterations() {}
  void inc_conflicts() {}
  void report_stat() const {}
  void merge_stat(const StatisticHandler& rhs) {}
  struct stat_sum {
    void add(StatisticHandler& x) {}
    void done() {}
    int num() { return 0; }
  };
};


//Handle Parallel Break
template<bool BREAK_ACTIVE>
class API_Break;
template<bool BREAK_ACTIVE>
class BreakImpl;

template<>
class BreakImpl<true> {
  volatile bool break_;
public:
  BreakImpl() : break_(false) {}

  void handleBreak() {
    break_ = true;
  }

  template<typename WLTy>
  void check(WLTy* wl) {
    typedef typename WLTy::value_type value_type;
    if (break_) {
      while (true) {
        std::pair<bool, value_type> p = wl->pop();
        if (!p.first)
          break;
      }
    }
  }
};

template<>
class BreakImpl<false> {
public:
  template<typename WLTy>
  void check(WLTy*) {}
};

template<>
class API_Break<true> {
  BreakImpl<true>* p_;
protected:
  void init_break(BreakImpl<true>* p) {
    p_ = p;
  }
public:
  void breakLoop() {
    p_->handleBreak();
  }
};
template<>
class API_Break<false> {
protected:
  void init_break(BreakImpl<false>* p) { }
};

//Handle Per Iter Allocator
template<bool PIA_ACTIVE>
class API_PerIter;

template<>
class API_PerIter<true>
{
  Galois::ItAllocBaseTy IterationAllocatorBase;
  Galois::PerIterAllocTy PerIterationAllocator;

protected:
  void __resetAlloc() {
    IterationAllocatorBase.clear();
  }

public:
  API_PerIter()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase)
  {}

  virtual ~API_PerIter() {
    IterationAllocatorBase.clear();
  }

  Galois::PerIterAllocTy& getPerIterAlloc() {
    return PerIterationAllocator;
  }
};

template<>
class API_PerIter<false>
{
protected:
  void __resetAlloc() {}
};


//Handle Parallel Push
template<bool PUSH_ACTIVE, typename WLT>
class API_Push;

template<typename WLT>
class API_Push<true, WLT> {
  typedef typename WLT::value_type value_type;
  WLT* wl;
protected:
  void init_wl(WLT* _wl) {
    wl = _wl;
  }
public:
  void push(const value_type& v) {
    wl->push(v);
  }
};

template<typename WLT>
class API_Push<false, WLT> {
protected:
  void init_wl(WLT* _wl) {}
};


template<typename Function>
struct Configurator {
  enum {
    CollectStats = !Galois::does_not_need_stats<Function>::value,
    NeedsBreak = Galois::needs_parallel_break<Function>::value,
    NeedsPush = !Galois::does_not_need_parallel_push<Function>::value,
    NeedsContext = !Galois::does_not_need_context<Function>::value,
    NeedsPIA = 1
  };
};

template<typename Function, class WorkListTy>
class ParallelThreadContext
  : public SimpleRuntimeContextHandler<Configurator<Function>::NeedsContext>,
    public StatisticHandler<Configurator<Function>::CollectStats>
{
  typedef typename WorkListTy::value_type value_type;
public:
  class UserAPI
    :public API_PerIter<Configurator<Function>::NeedsPIA>,
     public API_Push<Configurator<Function>::NeedsPush, WorkListTy>,
     public API_Break<Configurator<Function>::NeedsBreak>
  {
    friend class ParallelThreadContext;
  };

private:

  UserAPI facing;
  TerminationDetection::tokenHolder* lterm;
  bool leader;

public:
  ParallelThreadContext() {}
  
  virtual ~ParallelThreadContext() {}

  void initialize(TerminationDetection::tokenHolder* t,
		  bool _leader,
		  WorkListTy* wl,
		  BreakImpl<Configurator<Function>::NeedsBreak>* p) {
    lterm = t;
    leader = _leader;
    facing.init_wl(wl);
    facing.init_break(p);
  }

  void workHappened() {
    lterm->workHappened();
  }

  bool is_leader() const {
    return leader;
  }

  UserAPI& userFacing() {
    return facing;
  }

  void resetAlloc() {
    facing.__resetAlloc();
  }

};

template<class WorkListTy, class Function>
class ForEachWork : public Galois::Executable {
  typedef typename WorkListTy::value_type value_type;
  typedef GaloisRuntime::WorkList::MP_SC_FIFO<value_type> AbortedListTy;
  typedef ParallelThreadContext<Function, WorkListTy> PCTy;
  
  WorkListTy global_wl;
  BreakImpl<Configurator<Function>::NeedsBreak> breaker;
  Function& f;

  PerCPU<PCTy> tdata;
  TerminationDetection term;
  AbortedListTy aborted;
  volatile long abort_happened; //hit flag

  bool drainAborted() {
    bool retval = false;
    abort_happened = 0;
    std::pair<bool, value_type> p = aborted.pop();
    while (p.first) {
      retval = true;
      global_wl.push(p.second);
      p = aborted.pop();
    }
    return retval;
  }

  void doAborted(value_type val) {
    aborted.push(val);
    abort_happened = 1;
  }

  void doProcess(value_type val, PCTy& tld) {
    tld.inc_iterations();
    tld.start_iteration();
    try {
      f(val, tld.userFacing());
    } catch (int a) {
      tld.cancel_iteration();
      tld.inc_conflicts();
      doAborted(val);
      return;
    }
    tld.commit_iteration();
    tld.resetAlloc();
  }

public:
  template<typename IterTy>
  ForEachWork(IterTy b, IterTy e, Function& _f)
    :f(_f) {
    global_wl.fill_initial(b, e);
  }
  
  ~ForEachWork() {
    typename PCTy::stat_sum s;
    for (int i = 0; i < s.num(); ++i)
      s.add(tdata.get(i));
    s.done();
    
    for (int i = 1; i < s.num(); ++i)
      tdata.get(0).merge_stat(tdata.get(i));
    tdata.get(0).report_stat();
    assert(global_wl.empty());
  }

  virtual void operator()() {
    PCTy& tld = tdata.get();
    tld.initialize(term.getLocalTokenHolder(), 
		   tdata.myEffectiveID() == 0,
		   &global_wl,
		   &breaker);

    tld.start_parallel_region();
    do {
      std::pair<bool, value_type> p = global_wl.pop();
      if (p.first) {
        tld.workHappened();
        doProcess(p.second, tld);
        do {
          if (tld.is_leader() && abort_happened) {
            drainAborted();
          }
          breaker.check(&global_wl);
          p = global_wl.pop();
          if (p.first) {
            doProcess(p.second, tld);
          } else {
            break;
          }
        } while(true);
      }
      if (tld.is_leader() && drainAborted())
        continue;

      breaker.check(&global_wl);

      term.localTermination();
    } while (!term.globalTermination());

    tld.end_parallel_region();
  }
};

template<typename WLTy, typename IterTy, typename Function>
void for_each_impl(IterTy b, IterTy e, Function f) {
#ifdef GALOIS_VTUNE
  __itt_resume();
#endif

  typedef typename WLTy::template retype<typename std::iterator_traits<IterTy>::value_type>::WL aWLTy;

  ForEachWork<aWLTy, Function> GW(b, e, f);
  ThreadPool& PTP = getSystemThreadPool();

  PTP.run(&GW);

#ifdef GALOIS_VTUNE
  __itt_pause();
#endif
}

}

#endif

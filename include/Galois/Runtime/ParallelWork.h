// simple galois scheduler and runtime -*- C++ -*-
#ifndef __PARALLELWORK_H_
#define __PARALLELWORK_H_

#include "Galois/Executable.h"
#include "Galois/PerIterMem.h"

#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/DistributedWorkList.h"
#include "Galois/Runtime/DebugWorkList.h"
#include "Galois/Runtime/Termination.h"

#ifdef GALOIS_VTUNE
#include "ittnotify.h"
#endif

namespace GaloisRuntime {

template<class WorkListTy>
class ParallelThreadContext
  : public SimpleRuntimeContext, public Galois::PerIterMem
{
  unsigned long conflicts;
  unsigned long iterations;
  unsigned long TotalTime;
  WorkListTy* wl;

  typedef typename WorkListTy::value_type value_type;
  
  using SimpleRuntimeContext::start_iteration;
  using SimpleRuntimeContext::cancel_iteration;
  using SimpleRuntimeContext::commit_iteration;
  using PerIterMem::__resetAlloc;

public:
  ParallelThreadContext()
    :conflicts(0), iterations(0), TotalTime(0), wl(0)
  {}
  
  virtual ~ParallelThreadContext() {}

  void setTotalTime(unsigned long L) { TotalTime = L; }
  void set_wl(WorkListTy* WL) { wl = WL; }

  void report() const {
    reportStat("Conflicts", conflicts);
    reportStat("Iterations ", iterations);
    reportStat("TotalTime ", TotalTime);
  }

  template<typename Function>
  bool doProcess(value_type val, Function& f) {
    ++iterations;
    start_iteration();
    try {
      f(val, *this);
    } catch (int a) {
      cancel_iteration();
      ++conflicts;
      wl->aborted(val);
      return false;
    }
    commit_iteration();
    __resetAlloc();
    return true;
  }


  void push(value_type val) {
    wl->push(val);
  }

  void finish() { }
  
  void suspendWith(Galois::Executable* E) {
    abort();
  }

  static void merge(ParallelThreadContext& lhs, ParallelThreadContext& rhs) {
    lhs.conflicts += rhs.conflicts;
    lhs.iterations += rhs.iterations;
    lhs.TotalTime += rhs.TotalTime;
  }
  
};

template<class WorkListTy, class Function>
class ParallelWork : public Galois::Executable {
  typedef typename WorkListTy::value_type value_type;
  typedef ParallelThreadContext<WorkListTy> PCTy;

  WorkListTy& global_wl;
  Function& f;

  PerCPU<PCTy> tdata;
  TerminationDetection term;

public:
  ParallelWork(WorkListTy& _wl, Function& _f)
    :global_wl(_wl), f(_f) {}
  
  ~ParallelWork() {
    for (int i = 1; i < tdata.size(); ++i)
      PCTy::merge(tdata.get(0), tdata.get(i));
    tdata.get().report();
    assert(global_wl.empty());
  }

  virtual void preRun(int tmax) {
  }

  virtual void postRun() {  }

  virtual void operator()() {
    PCTy& tld = tdata.get();
    setThreadContext(&tld);
    tld.set_wl(&global_wl);
    Timer T;
    T.start();
    do {
      do {
	do {
	  std::pair<bool, value_type> p = global_wl.pop();
	  if (p.first) {
	    term.workHappened();
	    tld.doProcess(p.second, f);
	  } else {
	    break;
	  }
	} while(true);
	//break to here to do more expensive empty check
      } while (!global_wl.empty());
      term.localTermination();
    } while (!term.globalTermination());
    T.stop();
    tld.setTotalTime(T.get());
    setThreadContext(0);
  }
};

template<class Function, class GWLTy>
void for_each_parallel(GWLTy& GWL, Function& f) {
#ifdef GALOIS_VTUNE
  __itt_resume();
#endif
  ParallelWork<GWLTy, Function> GW(GWL, f);
  ThreadPool& PTP = getSystemThreadPool();
  PTP.run(&GW);
#ifdef GALOIS_VTUNE
  __itt_pause();
#endif
}

}

#endif

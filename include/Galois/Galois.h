// simple galois scheduler and runtime -*- C++ -*-

#include "Galois/Scheduling.h"
#include "Galois/Context.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/Termination.h"

#ifdef WITH_VTUNE
#include "/opt/intel/vtune_amplifier_xe_2011/include/ittnotify.h"
#endif

namespace GaloisRuntime {

template<class WorkListTy, class Function>
class GaloisWork : public Galois::Executable {
  typedef typename WorkListTy::value_type value_type;

  WorkListTy& global_wl;
  Function f;
  int do_halt;

  TerminationDetection term;

  class ThreadLD : public Galois::Context<value_type> {
  public:
    WorkListTy* wl;
    unsigned int conflicts;
    unsigned int iterations;
    unsigned long TotalTime;
    GaloisRuntime::SimpleRuntimeContext cnx;

  public:
    ThreadLD()
      :wl(0), conflicts(0), iterations(0), TotalTime(0)
    {}
    virtual ~ThreadLD() {}

    virtual void push(value_type val) {
      wl->push(val);
    }

    virtual void finish() {
    }

    virtual void suspendWith(Executable* E) {
      abort();
    }
    
    virtual SimpleRuntimeContext* getRuntimeContext() {
      return &cnx;
    }

    static void merge(ThreadLD& lhs, ThreadLD& rhs) {
      lhs.conflicts += rhs.conflicts;
      lhs.iterations += rhs.iterations;
      lhs.TotalTime += rhs.TotalTime;
    }

    void resetAlloc_() {
      Galois::Context<value_type>::resetAlloc();
    }

  };

  CPUSpaced<ThreadLD> tdata;

public:
  GaloisWork(WorkListTy& _wl, Function _f)
    :global_wl(_wl), f(_f), do_halt(0), tdata(ThreadLD::merge) {}

  ~GaloisWork() {

    std::cout << "STAT: Conflicts " << tdata.get().conflicts << "\n";
    std::cout << "STAT: Iterations " << tdata.get().iterations << "\n";
    std::cout << "STAT: TotalTime " << tdata.get().TotalTime << "\n";
    assert(global_wl.empty());
  }

  void doProcess(value_type val, ThreadLD& tld) __attribute__((noinline)) {
    ++tld.iterations;
    tld.cnx.start();
    try {
      f(val, tld);
    } catch (int a) {
      tld.cnx.cancel();
      ++tld.conflicts;
      global_wl.aborted(val);
      return;
    }
    tld.cnx.commit();
    tld.resetAlloc_();
    return;
  }

  virtual void preRun(int tmax) {
  }

  virtual void postRun() {
  }

  virtual void operator()() {
    ThreadLD& tld = tdata.get();
    setThreadContext(&tld.cnx);
    tld.wl = &global_wl;
    term.initialize(&tld.iterations);
    Timer T;
    T.start();
    //    do {
      do {
	do {
	  std::pair<bool, value_type> p = global_wl.pop();
	  if (p.first)
	    doProcess(p.second, tld);
	  else
	    break;
	} while(true);
	//break to here to do more expensive empty check
      } while (!global_wl.empty());
      //term.locallyDone();
      //    } while (!term.areWeThereYet());
    T.stop();
    tld.TotalTime = T.get();
    setThreadContext(0);
  }
};

template<class Function, class GWLTy>
void for_each_simple(GWLTy& GWL, Function f) {
#ifdef WITH_VTUNE
  __itt_resume();
#endif
  GaloisWork<GWLTy, Function> GW(GWL, f);
  ThreadPool& PTP = getSystemThreadPool();
  PTP.run(&GW);
#ifdef WITH_VTUNE
  __itt_pause();
#endif
}

}

//The user interface
namespace Galois {

static __attribute__((unused)) void setMaxThreads(int T) {
  GaloisRuntime::getSystemThreadPool().resize(T);
}

template<typename Function, typename GWLTy>
void for_each(GWLTy& P, Function f) {
  GaloisRuntime::for_each_simple(P, f);
}

template<typename IterTy, typename Function>
void for_each(IterTy b, IterTy e, Function f) {
  typedef GaloisRuntime::WorkList::ChunkedFIFO<typename IterTy::value_type, 256> GWLTy;
  GWLTy GWL;
  GWL.fill_initial(b, e);
  for_each(GWL, f);
}



}

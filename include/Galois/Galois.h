// simple galois scheduler and runtime -*- C++ -*-

#include "Galois/Scheduling.h"
#include "Galois/Context.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"

namespace GaloisRuntime {

template<class WorkListTy, class Function>
class GaloisWork : public Galois::Executable {
  typedef typename WorkListTy::value_type value_type;
  typedef typename WorkListTy::LocalTy localWLTy;
  //typedef GaloisRuntime::WorkList::Local::GWL_LIFO_SB<value_type> localWLTy;
  //typedef GaloisRuntime::WorkList::Local::GWL_ChaseLev_Dyn<value_type> localWLTy;
  //typedef GaloisRuntime::WorkList::Local::GWL_Idempotent_FIFO_SB<value_type> localWLTy;
  //typedef GaloisRuntime::WorkList::Local::GWL_PQueue<value_type, std::greater<value_type> > localWLTy;

  WorkListTy& global_wl;
  Function f;
  int threadmax;
  int threadsWorking;

  class ThreadLD : public Galois::Context<value_type> {
  public:
    localWLTy* wl;
    int conflicts;
    int iterations;
    unsigned long TotalTime;
    GaloisRuntime::TimeAccumulator ProcessTime;
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
      lhs.ProcessTime += rhs.ProcessTime;
      lhs.TotalTime += rhs.TotalTime;
    }

    void resetAlloc_() {
      Galois::Context<value_type>::resetAlloc();
    }

  };

  CPUSpaced<ThreadLD> tdata;

public:
  GaloisWork(WorkListTy& _wl, Function _f)
    :global_wl(_wl), f(_f), threadmax(0), threadsWorking(0), tdata(ThreadLD::merge) {}

  ~GaloisWork() {

    std::cout << "STAT: Conflicts " << tdata.get().conflicts << "\n";
    std::cout << "STAT: Iterations " << tdata.get().iterations << "\n";
    std::cout << "STAT: TotalTime " << tdata.get().TotalTime << "\n";
    std::cout << "STAT: ProcessTime " << tdata.get().ProcessTime.get() << "\n";
    assert(global_wl.empty());
  }

  void doProcess(value_type val, ThreadLD& tld) __attribute__((noinline)) {
    ++tld.iterations;
    tld.cnx.start();
    tld.ProcessTime.start();
    try {
      f(val, tld);
    } catch (int a) {
      tld.ProcessTime.stop();
      tld.cnx.cancel();
      ++tld.conflicts;
      global_wl.push_aborted(val, tld.wl);
      return;
    }
    tld.ProcessTime.stop();
    tld.cnx.commit();
    tld.resetAlloc_();
    return;
  }

  virtual void preRun(int tmax) {
    threadmax = tmax;
  }

  virtual void postRun() {
  }

  bool trySteal(ThreadLD& tld) {
#if 0
    if (localWLTy::MAYSTEAL) {
      //Try to steal work
      int num = (ThreadPool::getMyID()) % (threadmax + 1);
      bool foundone = false;
      value_type val;
      if (tdata[num].wl) //Unresolvable Data race
	tdata[num].wl->steal(foundone);
      //Don't push it on the queue before we can execute it
      if (foundone) {
	doProcess(val, tld);
	//One item is enough
	return true;
      }
    }
#endif
    return false;
  }

  void runLocalQueue(ThreadLD& tld) __attribute__((noinline)) {
    if (tld.wl) {
      while (!tld.wl->empty()) {
	bool suc = false;
	value_type val = tld.wl->pop(suc);
	//      bool psuc = false;
	//      value_type pval = tld.wl->peek(psuc);
	//      if (psuc)
	//	pval.prefetch_all();
	if (suc)
	  doProcess(val, tld);
      }
      delete tld.wl;
      tld.wl = 0;
    }
  }

  virtual void operator()() {
    ThreadLD& tld = tdata.get();
    setThreadContext(&tld.cnx);

    Timer T;
    T.start();
    //    do {
    //      __sync_fetch_and_add(&threadsWorking, +1);
      do {
	do {
	  do {
	    runLocalQueue(tld);
	  } while ((tld.wl = global_wl.getNext()));
	} while (trySteal(tld));
      } while ((tld.wl && !tld.wl->empty()) || !global_wl.empty());
      //      __sync_fetch_and_sub(&threadsWorking, 1);
      //      usleep(50);
      //    } while (threadsWorking > 0);
    T.stop();
    tld.TotalTime = T.get();
    setThreadContext(0);
  }
};

template<class WorkListTy, class Function>
void for_each_simple(WorkListTy& wl, Function f) {
  //wl.sort();
  typedef GaloisRuntime::WorkList::Global::ChunkedFIFO<typename WorkListTy::value_type, 256> GWLTy;
  GWLTy GWL;
  GWL.fill_initial(wl.begin(), wl.end());
  GaloisWork<GWLTy, Function> GW(GWL, f);
  ThreadPool& PTP = getSystemThreadPool();
  PTP.run(&GW);
}

}

//The user interface
namespace Galois {

static __attribute__((unused)) void setMaxThreads(int T) {
  GaloisRuntime::getSystemThreadPool().resize(T);
}

template<typename WorkListTy, typename Function, typename Pri>
void for_each(WorkListTy& wl, Function f, Pri P) {
  GaloisRuntime::for_each_simple(wl, f);
}

template<typename WorkListTy, typename Function>
void for_each(WorkListTy& wl, Function f) {
  for_each(wl, f, Galois::Scheduling::Priority.defaultOrder());
}

}

// simple galois scheduler and runtime -*- C++ -*-

#include <cmath>

#include <stack>
#include <vector>

#include <iostream>

#include "Galois/Scheduling.h"
#include "Galois/Context.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"

namespace GaloisRuntime {

template<class WorkListTy, class Function>
class GaloisWork : public Galois::Executable {
  typedef typename WorkListTy::value_type value_type;
  typedef GaloisRuntime::WorkList::Local::GWL_LIFO_SB<value_type> localWLTy;
  //typedef GaloisRuntime::WorkList::Local::GWL_ChaseLev_Dyn<value_type> localWLTy;
  //typedef GaloisRuntime::WorkList::Local::GWL_Idempotent_FIFO_SB<value_type> localWLTy;
  //typedef GaloisRuntime::WorkList::Local::GWL_PQueue<value_type, std::greater<value_type> > localWLTy;

  WorkListTy& global_wl;
  Function f;
  int threadmax;
  int threadsWorking;

  class ThreadLD : public Galois::Context<value_type> {
  public:
    localWLTy wl;
    int conflicts;
    int iterations;
    GaloisRuntime::TimeAccumulator ProcessTime;
    GaloisRuntime::Timer TotalTime;
    GaloisRuntime::SimpleRuntimeContext cnx;
  public:
    ThreadLD()
      :conflicts(0), iterations(0)
    {}

    void setThreadID(int ID) {
      Galois::Context<value_type>::threadID = ID;
    }

    virtual void push(value_type T) {
      wl.push(T);
    }

    virtual void finish() {
    }

    virtual void suspendWith(Executable* E) {
      abort();
    }
  };

  CPUSpaced<ThreadLD> tdata;

public:
  GaloisWork(WorkListTy& _wl, Function _f)
    :global_wl(_wl), f(_f), threadmax(0), threadsWorking(0) {}

  ~GaloisWork() {
    int conflicts = 0;
    int iterations = 0;
    unsigned long tTime = 0;
    unsigned long pTime = 0;

    for (int i = 0; i < tdata.size(); ++i) {
      conflicts += tdata[i].conflicts;
      iterations += tdata[i].iterations;
      tTime += tdata[i].TotalTime.get();
      pTime += tdata[i].ProcessTime.get();
      assert(tdata[i].wl.empty());
    }

    std::cout << "STAT: Conflicts " << conflicts << "\n";
    std::cout << "STAT: Iterations " << iterations << "\n";
    std::cout << "STAT: TotalTime " << tTime << "\n";
    std::cout << "STAT: ProcessTime " << pTime << "\n";
    assert(global_wl.empty());
  }

  void doProcess(value_type val, ThreadLD& tld) {
    ++tld.iterations;
    setThreadContext(&tld.cnx);
    tld.cnx.start();
    tld.ProcessTime.start();
    try {
      f(val, tld);
    } catch (int a) {
      tld.ProcessTime.stop();
      tld.cnx.cancel();
      ++tld.conflicts;
      global_wl.push(val);
      return;
    }
    tld.ProcessTime.stop();
    tld.cnx.commit();
    return;
  }

  virtual void preRun(int tmax) {
    threadmax = tmax;
    tdata.late_initialize(tmax);
  }

  virtual void postRun() {
  }

  bool trySteal(ThreadLD& tld) {
    if (localWLTy::MAYSTEAL) {
      //Try to steal work
      int num = (1 + tld.getThreadID()) % threadmax;
      bool foundone = false;
      value_type val = tdata[num].wl.steal(foundone);
      //Don't push it on the queue before we can execute it
      if (foundone) {
	doProcess(val, tld);
	//One item is enough
	return true;
      }
    }
    return false;
  }

  void runLocalQueue(ThreadLD& tld) {
    while (!tld.wl.empty()) {
      bool suc = false;
      value_type val = tld.wl.pop(suc);
      //      bool psuc = false;
      //      value_type pval = tld.wl.peek(psuc);
      //      if (psuc)
      //	pval.prefetch_all();
      if (suc)
	doProcess(val, tld);
    }
  }

  virtual void operator()(int ID, int tmax) {
    ThreadLD& tld = tdata[ID];
    tld.setThreadID(ID);
    setThreadContext(&tld.cnx);

    tld.TotalTime.start();
    //    do {
    //      __sync_fetch_and_add(&threadsWorking, +1);
      do {
	do {
	  do {
	    runLocalQueue(tld);
	  } while (global_wl.moveTo(tld.wl, 512));
	} while (trySteal(tld));
      } while (!tld.wl.empty() || !global_wl.empty());
      //      __sync_fetch_and_sub(&threadsWorking, 1);
      //      usleep(50);
      //    } while (threadsWorking > 0);
    tld.TotalTime.stop();
  }
};

template<class WorkListTy, class Function>
void for_each_simple(WorkListTy& wl, Function f) {
  //wl.sort();
  GaloisWork<WorkListTy, Function> GW(wl, f);
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

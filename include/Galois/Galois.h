// simple galois scheduler and runtime -*- C++ -*-

#include <cmath>

#include <stack>
#include <vector>
#include <ext/malloc_allocator.h>

#include <iostream>

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"

#include <boost/utility.hpp>

namespace GaloisRuntime {

template<class WorkListTy, class Function>
class GaloisWork : public Executable {
  typedef typename WorkListTy::value_type value_type;
  //typedef GWL_LIFO_SB<value_type> localWLTy;
  typedef GWL_ChaseLev_Dyn<value_type> localWLTy;
  //  typedef GWL_Idempotent_FIFO_SB<value_type> localWLTy;

  WorkListTy& global_wl;
  Function f;
  int threadmax;
  int threadsWorking;

  struct ThreadLD {
    localWLTy wl;
    int conflicts;
    GaloisRuntime::TimeAccumulator ProcessTime;
    GaloisRuntime::Timer TotalTime;
    int ID;
    bool pausing;
    ThreadLD() :conflicts(0), ID(-1), pausing(false) {}
  };

  CPUSpaced<ThreadLD> tdata;

public:
  GaloisWork(WorkListTy& _wl, Function _f)
    :global_wl(_wl), f(_f), threadmax(0), threadsWorking(0) {}

  ~GaloisWork() {
    int conflicts = 0;
    unsigned long tTime = 0;
    unsigned long pTime = 0;

    for (int i = 0; i < tdata.size(); ++i) {
      conflicts += tdata[i].conflicts;
      tTime += tdata[i].TotalTime.get();
      pTime += tdata[i].ProcessTime.get();
      assert(tdata[i].wl.empty());
    }

    std::cout << "STAT: Conflicts " << conflicts << "\n";
    std::cout << "STAT: TotalTime " << tTime << "\n";
    std::cout << "STAT: ProcessTime " << pTime << "\n";
    assert(global_wl.empty());
  }

  void doProcess(value_type val, ThreadLD& tld) {
    tld.ProcessTime.start();
    SimpleRuntimeContext cnx;
    setThreadContext(&cnx);
    try {
      f(val, tld.wl);
    } catch (int a) {
      tld.ProcessTime.stop();
      ++tld.conflicts;
      global_wl.push(val);
      return;
    }
    setThreadContext(0);
    tld.ProcessTime.stop();
    return;
  }

  virtual void preRun(int tmax) {
    threadmax = tmax;
    tdata.late_initialize(tmax);
  }

  virtual void postRun() {
  }

  bool trySteal(ThreadLD& tld) {
    //Try to steal work
    int num = (1 + tld.ID) % threadmax;
    bool foundone = false;
    value_type val = tdata[num].wl.steal(foundone);
    //Don't push it on the queue before we can execute it
    if (foundone) {
      doProcess(val, tld);
      //One item is enough
      return true;
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
    tld.ID = ID;

    tld.TotalTime.start();
    //    do {
    //      __sync_fetch_and_add(&threadsWorking, +1);
      do {
	do {
	  do {
	    runLocalQueue(tld);
	  } while (global_wl.moveTo(tld.wl, 4));
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

template<typename T>
class WorkList : boost::noncopyable {
public:
  virtual void push(T) = 0;
};

static __attribute__((unused)) void setMaxThreads(int T) {
  GaloisRuntime::getSystemThreadPool().resize(T);
}

template<typename WorkListTy, typename Function>
void for_each(WorkListTy& wl, Function f) {
  GaloisRuntime::for_each_simple(wl, f);
}
}

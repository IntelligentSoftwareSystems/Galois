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

namespace GaloisRuntime {
  
  template<class WorkListTy, class Function> 
  class GaloisWork : public Executable {
    typedef typename WorkListTy::value_type value_type;
    typedef GWL_LIFO<value_type> localWLTy;

    WorkListTy& global_wl;
    Function f;
    int conflicts;
    unsigned long tTime;
    unsigned long pTime;
    CPUSpaced<localWLTy> local_wl;

  public:
    GaloisWork(WorkListTy& _wl, Function _f)
      :global_wl(_wl), f(_f), conflicts(0), tTime(0), pTime(0)
    {}
    
    ~GaloisWork() {
      std::cout << "STAT: Conflicts " << conflicts << "\n";
      std::cout << "STAT: TotalTime " << tTime << "\n";
      std::cout << "STAT: ProcessTime " << pTime << "\n";
    }

    bool doProcess(value_type val, localWLTy& wlLocal) {
      SimpleRuntimeContext cnx;
      setThreadContext(&cnx);
      try {
	f(val, wlLocal);
      } catch (int a) {
	return false;
      }	
      setThreadContext(0);
      return true;
    }

    virtual void preRun(int tmax) {
      local_wl.resize(tmax);
    }

    virtual void postRun() {
    }
    
    virtual void operator() (int ID, int tmax) {
      localWLTy& wlLocal = local_wl[ID];

      int lconflicts = 0;
      GaloisRuntime::Timer TotalTime;
      GaloisRuntime::TimeAccumulator ProcessTime;

      TotalTime.start();

      int failed = 0;

      do {
	do {
	  //move some items out of the global list
	  global_wl.moveTo(wlLocal, 256);

	  while (!wlLocal.empty()) {
	    bool suc;
	    value_type val = wlLocal.pop(suc);
	    if (suc) {
	      ProcessTime.start();
	      bool result = doProcess(val, wlLocal);
	      ProcessTime.stop();
	      if (!result) {
		++lconflicts;
		//wlDelay.push_back(val);
		global_wl.push(val);
	      }
	    }     
	  }
	} while (!global_wl.empty());

	//Try to steal work
	for (int i = 0; i < tmax; ++i) {
	  bool suc;
	  value_type val = local_wl[(i + ID) % tmax].steal(suc);
	  //Don't push it on the queue before we can execute it
	  if (suc) {
	    failed = 0;
	    ProcessTime.start();
	    bool result = doProcess(val, wlLocal);
	    ProcessTime.stop();
	    if (!result) {
	      ++lconflicts;
	      //wlDelay.push_back(val);
	      global_wl.push(val);
	    }
	    //One item is enough
	    break;
	  }
	}
      } while (!wlLocal.empty());
    
      TotalTime.stop();
      __sync_fetch_and_add(&tTime, TotalTime.get());
      __sync_fetch_and_add(&pTime, ProcessTime.get());
      __sync_fetch_and_add(&conflicts, lconflicts);
    }
  };

  template<class WorkListTy, class Function>
  void for_each_simple (WorkListTy& wl, Function f)
  {
    GaloisWork<WorkListTy, Function> GW(wl, f);
    ThreadPool& PTP = getSystemThreadPool();
    PTP.run(&GW);
  }
  

}

//The user interface
namespace Galois {

  template<typename T>
  class WorkList {
  public:
    virtual void push(T) = 0;
  };

  static __attribute__((unused)) void setMaxThreads(int T) {
    GaloisRuntime::getSystemThreadPool().resize(T);
  }

  template<typename WorkListTy, typename Function>
  void for_each (WorkListTy& wl, Function f) {
    GaloisRuntime::for_each_simple(wl, f);
  }

}

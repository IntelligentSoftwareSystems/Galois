// simple galois scheduler and runtime -*- C++ -*-

#include <set>
#include <stack>
#include <iostream>
#include <pthread.h>
#include "Galois/Runtime/Context.h"

namespace GaloisRuntime {
  
  extern int numThreads;

  template<class WorkListTy, class Function> 
  class GaloisWork {
    WorkListTy& wl;
    Function f;
    int conflicts;
  public:
    GaloisWork(WorkListTy& _wl, Function _f)
      :wl(_wl), f(_f), conflicts(0)
    {}
    
    ~GaloisWork() {
      std::cerr << "Conflicts: " << conflicts << "\n";
    }

    static void* threadLaunch(void* _GW) {
      GaloisWork* GW = (GaloisWork*)_GW;
      GW->perThreadLaunch();
      return 0;
    }
    void perThreadLaunch() {
      std::stack<typename WorkListTy::value_type> wlLocal;
      while (!wl.empty()) {
	//move some items out of the global list
	wl.moveTo(wlLocal, 256);

	while (!wlLocal.empty()) {
	  SimpleRuntimeContext cnx;
	  setThreadContext(&cnx);
	  typename WorkListTy::value_type val = wlLocal.top();
	  wlLocal.pop();
	  try {
	    val.getData(); //acquire lock
	    f(val, wlLocal);
	  } catch (int a) {
	    wl.push(val); // put conflicting work onto the global wl
	    __sync_fetch_and_add(&conflicts, 1);
	  }
	  setThreadContext(0);
	}
      }
    }
  };

  class pThreadPool {
    pthread_t threadpool[64]; // FIXME: be dynamic
    int num;
  public:
    pThreadPool(int i) {
      num = i;
    }

    template<class _GaloisWork>
    void launch(_GaloisWork* GW) {
      for (int i = 0; i < (num - 1); ++i)
	pthread_create(&threadpool[i], 0, _GaloisWork::threadLaunch, (void*)GW);
      //We use this thread for the last thread to avoid serial overhead for now
      _GaloisWork::threadLaunch(GW);
    }
 
    void wait() {
      for (int i = 0; i < (num - 1); ++i)
	pthread_join(threadpool[i], 0);
    }
  };

  template<class WorkListTy, class Function>
  void for_each_simple (WorkListTy& wl, Function f)
  {
    GaloisWork<WorkListTy, Function> GW(wl, f);
    pThreadPool PTP(numThreads);
    PTP.launch(&GW);
    PTP.wait();
  }
  

}

//The user interface

namespace Galois {

  extern void setMaxThreads(int T);

  template<typename WorkListTy, typename Function>
  void for_each (WorkListTy& wl, Function f)
  {
    GaloisRuntime::for_each_simple(wl, f);
  }

}

// simple galois scheduler and runtime -*- C++ -*-

#include <set>
#include <pthread.h>
#include "Galois/Runtime/Context.h"


namespace GaloisRuntime {
  
  extern int numThreads;

  template<class WorkListTy, class Function> 
  class GaloisWork {
    WorkListTy& wl;
    Function f;
  public:
    GaloisWork(WorkListTy& _wl, Function _f)
      :wl(_wl), f(_f)
    {}
    
    static void* threadLaunch(void* _GW) {
      GaloisWork* GW = (GaloisWork*)_GW;
      GW->perThreadLaunch();
      return 0;
    }
    void perThreadLaunch() {
      WorkListTy wlLocal;
      while (!wl.empty()) {
	bool gotOne = false;
	typename WorkListTy::value_type val = wl.pop(gotOne);
	if (gotOne) {
	  wlLocal.push(val);
	  while (!wlLocal.empty()) {
	    SimpleRuntimeContext cnx;
	    setThreadContext(&cnx);
	    val = wlLocal.pop(gotOne);
	    try {
	      val.getData(); //acquire lock
	      f(val, wlLocal);
	    } catch (int a) {
	      wl.push(val); // put conflicting work on the global wl
	    }
	    setThreadContext(0);
	  }
	}
      }
    }
  };

  class pThreadPool {
    pthread_t threadpool[16]; // FIXME
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

  template<class WorkListTy, class Function> 
  void for_each (WorkListTy& wl, Function f)
  {
    GaloisRuntime::for_each_simple(wl, f);
  }
  
}

// simple galois scheduler and runtime -*- C++ -*-

#include <cmath>

#include <stack>
#include <vector>
#include <ext/malloc_allocator.h>

#include <iostream>

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/ThreadPool.h"

namespace GaloisRuntime {
  
  template<typename T>
  struct LocalWorkListTy {
    typedef std::stack<T, std::deque<T, __gnu_cxx::malloc_allocator<T> > > type;
  };

  template<class WorkListTy, class Function> 
  class GaloisWork : public Executable {
    WorkListTy& wl;
    Function f;
    int conflicts;
    unsigned long tTime;
    unsigned long pTime;
  public:
    GaloisWork(WorkListTy& _wl, Function _f)
      :wl(_wl), f(_f), conflicts(0), tTime(0), pTime(0)
    {}
    
    ~GaloisWork() {
      std::cerr << "Conflicts: " << conflicts << "\n";
      std::cerr << "Total Time: " << tTime << "\n";
      std::cerr << "Process Time: " << pTime << "\n";
      std::cerr << "Scheduling Overhead: " << (double)(tTime - pTime) / tTime << "\n";

    }

    static void* threadLaunch(void* _GW) {
      GaloisWork* GW = (GaloisWork*)_GW;
      GW->perThreadLaunch();
      return 0;
    }

    bool doProcess(typename WorkListTy::value_type val, typename LocalWorkListTy<typename WorkListTy::value_type>::type& wlLocal) {
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
    
    virtual void operator() (int ID, int tmax) {
      typename LocalWorkListTy<typename WorkListTy::value_type>::type wlLocal;

      //std::vector<typename WorkListTy::value_type> wlDelay;
      int lconflicts = 0;
      GaloisRuntime::Timer TotalTime;
      GaloisRuntime::TimeAccumulator ProcessTime;

      TotalTime.start();

      while (!wl.empty()) {
	//move some items out of the global list
	wl.moveTo(wlLocal, 256);

	while (!wlLocal.empty()) {
	  typename WorkListTy::value_type val = wlLocal.top();
	  wlLocal.pop();
	  ProcessTime.start();
	  bool result = doProcess(val, wlLocal);
	  ProcessTime.stop();
	  if (!result) {
	    ++lconflicts;
	    //wlDelay.push_back(val);
	    wl.push(val);
	  }
	}
	//Wait to the end to push conflicts globally.  Intended to 
	//help caching
	//wl.insert(wlDelay.begin(), wlDelay.end());
	//wlDelay.clear();
      }
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

  static void setMaxThreads(int T) {
    GaloisRuntime::getSystemThreadPool().resize(T);
  }

  template<typename WorkListTy, typename Function>
  void for_each (WorkListTy& wl, Function f)
  {
    GaloisRuntime::for_each_simple(wl, f);
  }

}

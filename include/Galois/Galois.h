// simple galois scheduler and runtime -*- C++ -*-

#include <set>
#include <list>
#include <map>
#include <pthread.h>

class Lockable {
  threadsafe::simpleLock L;
public:
  bool try_lock() {
    return L.try_write_lock();
  }
  void unlock() {
    L.write_unlock();
  }
};

namespace GaloisRuntime {
  
  extern int numThreads;

  class GaloisWorkContextCausious {
    std::set<Lockable*> locks;
    
    void rollback() {
      throw -1;
    }

  public:

    ~GaloisWorkContextCausious() {
      for (std::set<Lockable*>::iterator ii = locks.begin(), ee = locks.end(); ii != ee; ++ii)
	(*ii)->unlock();
    }
    void acquire(Lockable* C) {
      if (!locks.count(C)) {
	bool suc = C->try_lock();
	if (suc) {
	  locks.insert(C);
	} else {
	  rollback();
	}
      }
    }

    void release(Lockable* C) {
      C->unlock();
      locks.erase(C);
    }
  };

  extern __thread GaloisWorkContextCausious* thread_cnx;

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
      while (!wl.empty()) {
	GaloisWorkContextCausious cnx;
	thread_cnx = &cnx;
	typename WorkListTy::value_type val = wl.pop_or_value(0);
	if (val) {
	  try {
	    cnx.acquire(val);
	    f(val, wl);
	  } catch (int a) {
	    wl.push(val);
	  }
	}
	thread_cnx = 0;
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
      for (int i = 0; i < num; ++i)
	pthread_create(&threadpool[i], 0, _GaloisWork::threadLaunch, (void*)GW);
    }
 
    void wait() {
      for (int i = 0; i < num; ++i)
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

/*! \file 
 *  \brief cray thread pool implementation
 */

#include "galois_config.h"
#ifdef WITH_CRAY_POOL

#include "Galois/Runtime/ThreadPool.h"

using namespace GaloisRuntime;

namespace {

  class ThreadPool_cray : public ThreadPool {
    
    int tmax;
    int num;
    Executable* work;

    int mkID() {
      return int_fetch_add(&tmax, 1);
    }

    void launch(void) {
      int myID = mkID();
      (*work)(myID, num);
    }

  public:
    ThreadPool_cray() 
      :tmax(0), num(1)
    {}

    virtual void run(Executable* E) {
      work = E;
      work->preRun(numThreads());
      ThreadPool_cray* p = this;
      future int children[num];
      future children$(p) {
	p->launch();
	return 0;
      }
      for (int i = 0; i < num; ++i)
	touch(&children$[i])
      work->postRun();
    }

    virtual void resize(int num) {
      this->num = num;
    }

    virtual int size() {
      return num;
    }
  };
}


//! Implement the global threadpool
static ThreadPool_cray pool;

ThreadPool& GaloisRuntime::getSystemThreadPool() {
  return pool;
}

#endif

/*! \file 
 *  \brief cray thread pool implementation
 */

#ifdef GALOIS_CRAY

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
      work->preRun(num);
#pragma mta assert parallel
      for (int i = 0; i < num; ++i) {
	launch();
      }
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

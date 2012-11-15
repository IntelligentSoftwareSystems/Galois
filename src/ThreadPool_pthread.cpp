/** pthread thread pool implementation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/ll/EnvCheck.h"
#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Runtime/ll/TID.h"

#include "boost/utility.hpp"

#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <cassert>
#include <vector>

#include <semaphore.h>
#include <pthread.h>

//Forward declare this to avoid including PerThreadStorage
//We avoid this to stress that the thread Pool MUST NOT depend on PTS
namespace GaloisRuntime {
extern void initPTS();
}

using namespace GaloisRuntime;

//! Generic check for pthread functions
static void checkResults(int val) {
  if (val) {
    perror("PTHREAD: ");
    assert(0 && "PThread check");
    abort();
  }
}
 
namespace {

class Semaphore: private boost::noncopyable {
  sem_t sem;
public:
  explicit Semaphore(int val = 0) {
    int rc = sem_init(&sem, 0, val);
    checkResults(rc);
  }

  ~Semaphore() {
    int rc = sem_destroy(&sem);
    checkResults(rc);
  }

  void release(int n = 1) {
    while (n) {
      --n;
      int rc = sem_post(&sem);
      checkResults(rc);
    }
  }

  void acquire(int n = 1) {
    while (n) {
      --n;
      int rc;
      while (((rc = sem_wait(&sem)) < 0) && (errno == EINTR)) { }
      checkResults(rc);
    }
  }
};

class ThinBarrier: private boost::noncopyable {
  volatile int started;
  int val;
public:
  ThinBarrier(int v): val(v) { }
  void release(int n = 1) {
    __sync_fetch_and_add(&started, 1);
  }
  void acquire(int n = 1) {
    while (started < n) { }
  }
};


class ThreadPool_pthread : public ThreadPool {
  pthread_t* threads; // set of threads
  Semaphore* starts;  // signal to release threads to run
  ThinBarrier started;
  unsigned maxThreads;
  volatile bool shutdown; // Set and start threads to have them exit
  volatile unsigned starting; //Each run call uses this to control #threads
  volatile RunCommand* workBegin; //Begin iterator for work commands
  volatile RunCommand* workEnd; //End iterator for work commands

  void initThread() {
    //initialize TID
    GaloisRuntime::LL::initTID();
    unsigned id = GaloisRuntime::LL::getTID();
    GaloisRuntime::initPTS();
    if (id != 0 || !LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD"))
      GaloisRuntime::LL::bindThreadToProcessor(id);
    //we use a simple pthread or atomic to avoid depending on Galois
    //stuff too early in the initialization process
    started.release();
  }

  void cascade(int tid) {
    const unsigned multiple = 2;
    for (unsigned i = 1; i <= multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < starting)
        starts[n].release();
    }
  }

  void doWork(unsigned LocalThreadID) {
    cascade(LocalThreadID);
    RunCommand* workPtr = (RunCommand*)workBegin;
    RunCommand* workEndL = (RunCommand*)workEnd;
    while (workPtr != workEndL) {
      (*workPtr)();
      ++workPtr;
    }
  }

  void launch(void) {
    unsigned LocalThreadID = GaloisRuntime::LL::getTID();
    while (!shutdown) {
      starts[LocalThreadID].acquire();  
      doWork(LocalThreadID);
    }
  }

  static void* slaunch(void* V) {
    ThreadPool_pthread* TP = (ThreadPool_pthread*)V;
    TP->initThread();
    TP->launch();
    return 0;
  }
  
public:
  ThreadPool_pthread(): started(0), shutdown(false), workBegin(0), workEnd(0)
  {
    maxThreads = GaloisRuntime::LL::getMaxThreads();
    initThread();

    starts = new Semaphore[maxThreads];
    threads = new pthread_t[maxThreads];

    for (unsigned i = 1; i < maxThreads; ++i) {
      int rc = pthread_create(&threads[i], 0, &slaunch, this);
      checkResults(rc);
    }
    started.acquire(maxThreads);
  }

  virtual ~ThreadPool_pthread() {
    shutdown = true;
    workBegin = workEnd = 0;
    __sync_synchronize();
    for (unsigned i = 1; i < maxThreads; ++i)
      starts[i].release();
    for (unsigned i = 1; i < maxThreads; ++i) {
      int rc = pthread_join(threads[i], NULL);
      checkResults(rc);
    }
    delete [] starts;
    delete [] threads;
  }

  virtual void run(RunCommand* begin, RunCommand* end, unsigned num) {
    //sanatize num
    num = std::min(num, maxThreads);
    num = std::max(num, 1U);
    starting = num;
    //setup work
    workBegin = begin;
    workEnd = end;
    //ensure stores happen before children are spawned
    __sync_synchronize();
    //Do master thread work
    doWork(0);
    //clean up
    workBegin = workEnd = 0;
  }

  virtual unsigned getMaxThreads() const {
    return maxThreads;
  }
};

} // end namespace

//! Implement the global threadpool
ThreadPool& GaloisRuntime::getSystemThreadPool() {
  static ThreadPool_pthread pool;
  return pool;
}

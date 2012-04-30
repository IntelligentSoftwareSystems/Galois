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

#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Runtime/ll/TID.h"

#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <cassert>
#include <list>
#include <limits>
#include <vector>

#include <semaphore.h>
#include <pthread.h>

#ifdef GALOIS_VTUNE
#include "ittnotify.h"
#endif

using namespace GaloisRuntime;

//! Generic check for pthread functions
static void checkResults(int val) {
  if (val) {
    perror("PTHREADS: ");
    assert(0 && "PThread check");
    abort();
  }
}
 
namespace {

class Semaphore {
  sem_t sem;
public:
  explicit Semaphore(int val) {
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
      while ((rc = sem_wait(&sem)) == EINTR) { }
      checkResults(rc);
    }
  }
};

#ifdef GALOIS_VTUNE
class SampleProfiler {
  bool IsOn;
public:
  SampleProfiler() :IsOn(false) {}
  void startIf(int TID, bool ON) {
    if (IsOn != ON && TID == 0) {
      if (ON)
	__itt_resume();
      else
	__itt_pause();
      IsOn = ON;
    }
  }
};
#else
class SampleProfiler {
public:
  void startIf(int TID, bool ON) {}
};
#endif

class ThreadPool_pthread : public ThreadPool {
  std::vector<Semaphore> starts; // signal to release threads to run
  GBarrier* finish; // want on to block on running threads
  std::list<pthread_t> threads; // Set of threads
  unsigned int maxThreads;
  volatile unsigned int started;
  volatile bool shutdown; // Set and start threads to have them exit
  volatile RunCommand* workBegin; //Begin iterator for work commands
  volatile RunCommand* workEnd; //End iterator for work commands

  void initThread() {
    //initialize TID
    GaloisRuntime::LL::initTID();
    int id = GaloisRuntime::LL::getTID();
    GaloisRuntime::initPTS();
    GaloisRuntime::LL::bindThreadToProcessor(id);
    __sync_fetch_and_add(&started, 1);
  }

  void cascade(int tid) {
    unsigned multiple = 2;
    for (unsigned i = 0; i < multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < activeThreads && n != 0)
        starts[n].release();
    }
  }

  void doWork(void) {
    int LocalThreadID = GaloisRuntime::LL::getTID();
    
    if (LocalThreadID != 0)
      starts[LocalThreadID].acquire();

    if (LocalThreadID >= (int) activeThreads)
      return;

    cascade(LocalThreadID);
    
    RunCommand* workPtr = (RunCommand*)workBegin;
    RunCommand* workEndL = (RunCommand*)workEnd;
    SampleProfiler VT;
    while (workPtr != workEndL) {
      VT.startIf(LocalThreadID, workPtr->profile);
      if (workPtr->isParallel)
        workPtr->work();
      else if (LocalThreadID == 0)
        workPtr->work();

      if (workPtr->barrierAfter)
        finish->wait();
      ++workPtr;
    }
    VT.startIf(LocalThreadID, false);
  }

  void launch(void) {
    while (!shutdown) doWork();
  }

  static void* slaunch(void* V) {
    ThreadPool_pthread* TP = (ThreadPool_pthread*)V;
    TP->initThread();
    TP->launch();
    return 0;
  }
  
public:
  ThreadPool_pthread()
    :maxThreads(0), started(0), shutdown(false), workBegin(0), workEnd(0)
  {
    initThread();
    ThreadPool::activeThreads = 1;
    unsigned int num = GaloisRuntime::LL::getMaxThreads();
    maxThreads = num;

    starts.reserve(num);
    for (unsigned i = 0; i < num; ++i)
      starts.push_back(Semaphore(0));

    for (unsigned i = 1; i < num; ++i) {
      pthread_t t;
      int rc = pthread_create(&t, 0, &slaunch, this);
      checkResults(rc);
      threads.push_front(t);
    }
    while (started != maxThreads) {}
  }

  ~ThreadPool_pthread() {
    shutdown = true;
    workBegin = workEnd = 0;
    __sync_synchronize();
    for (unsigned i = 1; i < starts.size(); ++i)
      starts[i].release();
    while (!threads.empty()) {
      pthread_t t = threads.front();
      threads.pop_front();
      int rc = pthread_join(t, NULL);
      checkResults(rc);
    }
    delete finish;
  }

  virtual void run(RunCommand* begin, RunCommand* end) {
    workBegin = begin;
    workEnd = end;
    if (!finish) {
      finish = new GBarrier();
      finish->reinit(activeThreads);
    }
    __sync_synchronize();
    //Do master thread work
    doWork();
    //clean up
    workBegin = workEnd = 0;
  }

  virtual unsigned int setActiveThreads(unsigned int num) {
    if (num == 0) {
      activeThreads = 1;
    } else {
      activeThreads = std::min(num, maxThreads);
    }
    assert(activeThreads <= maxThreads);
    assert(activeThreads - 1 <= threads.size());

    if (!finish)
      finish = new GBarrier();
    finish->reinit(activeThreads);

    return activeThreads;
  }
};
}

//! Implement the global threadpool
ThreadPool& GaloisRuntime::getSystemThreadPool() {
  static ThreadPool_pthread pool;
  return pool;
}

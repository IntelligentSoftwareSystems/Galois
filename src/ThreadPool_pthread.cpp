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

#ifdef GALOIS_DMP
#include "dmp.h"
#endif

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

class SemSemaphore {
  sem_t sem;
public:
  explicit SemSemaphore(int val) {
    int rc = sem_init(&sem, 0, val);
    checkResults(rc);
  }

  ~SemSemaphore() {
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

class PthreadSemaphore {
  pthread_mutex_t lock;
  pthread_cond_t cond;
  int val;
public:
  explicit PthreadSemaphore(int v): val(v) {
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&cond, NULL);
  }
  
  ~PthreadSemaphore() {
    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&cond);
  }

  void release(int n = 1) {
    pthread_mutex_lock(&lock);
    val += n;
    if (val > 0)
      pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);
  }

  void acquire(int n = 1) {
    pthread_mutex_lock(&lock);
    while (val < n) {
      pthread_cond_wait(&cond, &lock);
    }
    val -= n;
    pthread_mutex_unlock(&lock);
  }
};

#ifdef GALOIS_VTUNE
class VtuneSampleProfiler {
  bool IsOn;
public:
  VtuneSampleProfiler() :IsOn(false) {}
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
#endif

class DummySampleProfiler {
public:
  void startIf(int TID, bool ON) {}
};

class AtomicThinBarrier {
  volatile int started;
  int val;
public:
  AtomicThinBarrier(int v): val(v) { }
  void release(int n = 1) {
    __sync_fetch_and_add(&started, 1);
  }
  void acquire(int n = 1) {
    while (started < n) { }
  }
};


class ThreadPool_pthread : public ThreadPool {
  // Instantiate pthread;
#if defined(GALOIS_VTUNE) && !defined(GALOIS_DRF)
  typedef VtuneSampleProfiler SampleProfiler;
#else
  typedef DummySampleProfiler SampleProfiler;
#endif
#ifndef GALOIS_DRF
  typedef SemSemaphore Semaphore;
  typedef GBarrier Barrier;
  typedef AtomicThinBarrier ThinBarrier;
#else
  typedef PthreadSemaphore Semaphore;
  typedef PthreadBarrier Barrier;
  typedef PthreadSemaphore ThinBarrier;
#endif

  std::vector<Semaphore> starts; // signal to release threads to run
  Barrier* finish; // want on to block on running threads
  std::list<pthread_t> threads; // Set of threads
  ThinBarrier started;
  unsigned int maxThreads;
  volatile bool shutdown; // Set and start threads to have them exit
  volatile RunCommand* workBegin; //Begin iterator for work commands
  volatile RunCommand* workEnd; //End iterator for work commands

  void initThread() {
    //initialize TID
    GaloisRuntime::LL::initTID();
    int id = GaloisRuntime::LL::getTID();
    GaloisRuntime::initPTS();
    GaloisRuntime::LL::bindThreadToProcessor(id);
    started.release();
  }

  void cascade(int tid) {
    unsigned multiple = 2;
    for (unsigned i = 1; i <= multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < activeThreads)
        starts[n].release();
    }
  }

  void doWork(void) {
    int LocalThreadID = GaloisRuntime::LL::getTID();
    
    if (LocalThreadID != 0)
      starts[LocalThreadID].acquire();

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
  ThreadPool_pthread():
    maxThreads(GaloisRuntime::LL::getMaxThreads()), 
    started(0), shutdown(false), workBegin(0), workEnd(0)
  {
    initThread();

    starts.reserve(maxThreads);
    for (unsigned i = 0; i < maxThreads; ++i)
      starts.push_back(Semaphore(0));

    for (unsigned i = 1; i < maxThreads; ++i) {
      pthread_t t;
      int rc = pthread_create(&t, 0, &slaunch, this);
      checkResults(rc);
      threads.push_front(t);
    }
    started.acquire(maxThreads);
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
      finish = new Barrier();
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
      finish = new Barrier();
    finish->reinit(activeThreads);

    return activeThreads;
  }
};

} // end namespace

//! Implement the global threadpool
ThreadPool& GaloisRuntime::getSystemThreadPool() {
  static ThreadPool_pthread pool;
  return pool;
}

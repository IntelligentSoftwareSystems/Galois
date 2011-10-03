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
#ifdef GALOIS_PTHREAD

#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/SimpleLock.h"

#include <semaphore.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cerrno>

//#include <iostream>
//#include <sstream>
#include <list>
#include <cassert>
#include <limits>

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
      while ((rc = sem_wait(&sem)) == EINTR) {}
      checkResults(rc);
    }
  }
};

class Barrier {
  pthread_barrier_t bar;
public:
  Barrier() {
    //uninitialized barriers block a lot of threads to help with debugging
    int rc = pthread_barrier_init(&bar, 0, std::numeric_limits<int>::max());
    checkResults(rc);
  }

  ~Barrier() {
    int rc = pthread_barrier_destroy(&bar);
    checkResults(rc);
  }

  void reinit(int val) {
   int rc = pthread_barrier_destroy(&bar);
   checkResults(rc);
   rc = pthread_barrier_init(&bar, 0, val);
   checkResults(rc);
   
  }

  void wait() {
    int rc = pthread_barrier_wait(&bar);
    if (rc && rc != PTHREAD_BARRIER_SERIAL_THREAD)
      checkResults(rc);
  }
};

class ThreadPool_pthread : public ThreadPool {
  Semaphore start; // Signal to release threads to run
  Barrier finish; // want on to block on running threads
  std::list<pthread_t> threads; // Set of threads
  unsigned int maxThreads;
  unsigned int nextThread;
  volatile bool shutdown; // Set and start threads to have them exit
  volatile runCMD* workBegin; //Begin iterator for work commands
  volatile runCMD* workEnd; //End iterator for work commands

  void initThread() {
    LocalThreadID = __sync_fetch_and_add(&nextThread, 1);
    GaloisRuntime::getSystemThreadPolicy().bindThreadToProcessor();
  }

  void doWork(void) {
    start.acquire();
    runCMD* workPtr = (runCMD*)workBegin;
    runCMD* workEndL = (runCMD*)workEnd;
    while (workPtr != workEndL) {
      if (LocalThreadID < activeThreads) {
	if (workPtr->isParallel)
	  workPtr->work();
	else if (LocalThreadID == 0)
	  workPtr->work();
      }
      if (workPtr->barrierAfter)
	finish.wait();
      ++workPtr;
    }
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
    :start(0), maxThreads(0), nextThread(0), shutdown(false), workBegin(0), workEnd(0)
  {
    initThread();
    ThreadPool::activeThreads = 1;
    unsigned int num = GaloisRuntime::getSystemThreadPolicy().getNumThreads();
    finish.reinit(num);
    maxThreads = num;
    for (unsigned i = 1; i < num; ++i) {
      pthread_t t;
      int rc = pthread_create(&t, 0, &slaunch, this);
      checkResults(rc);
      threads.push_front(t);
    }
  }

  ~ThreadPool_pthread() {
    shutdown = true;
    workBegin = workEnd = 0;
    __sync_synchronize();
    start.release(threads.size());
    while(!threads.empty()) {
      pthread_t t = threads.front();
      threads.pop_front();
      int rc = pthread_join(t, NULL);
      checkResults(rc);
    }
  }

  virtual void run(runCMD* begin, runCMD* end) {
#ifdef GALOIS_VTUNE
    __itt_resume();
#endif
    
    workBegin = begin;
    workEnd = end;
    __sync_synchronize();
    //start all threads
    start.release(maxThreads);
    //Do master thread work
    doWork();
    //clean up
    workBegin = workEnd = 0;

#ifdef GALOIS_VTUNE
    __itt_pause();
#endif
    
  }

  virtual unsigned int setActiveThreads(unsigned int num) {
    if (num == 0) {
      activeThreads = 1;
    } else {
      activeThreads = std::min(num, maxThreads);
    }
    assert(activeThreads <= maxThreads);
    assert(activeThreads <= threads.size());

    //std::ostringstream out;
    //out << "Threads set to " << num << " using " << activeThreads << " of " << maxThreads;
    //reportInfo("ThreadPool", out.str().c_str());
    return activeThreads;
  }
};
}

//! Implement the global threadpool
//static ThreadPool_pthread pool;
ThreadPool& GaloisRuntime::getSystemThreadPool() {
  static ThreadPool_pthread pool;
  return pool;
}

//FIXME: May need to factor this out to initialize all objects in the correct order
struct Init {
  Init() { 
    getSystemThreadPool();
  }
};

static Init iii;

#endif

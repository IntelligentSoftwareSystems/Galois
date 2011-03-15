/*! \file 
 *  \brief pthread thread pool implementation
 */

#ifdef GALOIS_PTHREAD

#include "Galois/Executable.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/Support.h"

#include "Galois/Runtime/SimpleLock.h"

#include <semaphore.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <iostream>
#include <list>
#include <cassert>
#include <limits>

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
      int rc = sem_wait(&sem);
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
  Galois::Executable* work; // Thing to execute
  volatile bool shutdown; // Set and start threads to have them exit
  std::list<pthread_t> threads; // Set of threads
  unsigned int maxThreads;

  // Return the number of processors on this hardware
  // This is the maximum number of threads that can be started
//   unsigned int numProcessors() {
// #ifdef __linux__
//     return sysconf(_SC_NPROCESSORS_CONF);
// #endif
//     reportWarning("Unknown number of processors (assuming 64)");
//     return 64;
//   }
  void bindToProcessor(int proc) {
#if 0
    int id = proc;
    int carry = 0;
    if (id > 23) {
      id -= 24;
      carry = 24;
    }
    proc = carry + ((id % 6) * 4) + (id / 6);
#endif
#ifdef __linux__
    cpu_set_t mask;
    /* CPU_ZERO initializes all the bits in the mask to zero. */
    CPU_ZERO( &mask );
    
    /* CPU_SET sets only the bit corresponding to cpu. */
    // void to cancel unused result warning
    (void)CPU_SET( proc, &mask );
    
    /* sched_setaffinity returns 0 in success */
    if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 )
      reportWarning("Could not set CPU Affinity for thread");
    
    return;
#endif      
    reportWarning("Don't know how to bind thread to cpu on this platform");
  }

  void launch(void) {
    GaloisRuntime::getSystemThreadPolicy().bindThreadToProcessor();
    unsigned int id = ThreadPool::getMyID();
    bindToProcessor(id - 1);

    while (true) {
      start.acquire();
      //std::cerr << "starting " << id << "\n";
      if (work && id <= activeThreads) {
	//std::cerr << "using " << id << "\n";
	(*work)();
      }
      if(shutdown)
	break;
      finish.wait();
    }
  }

  static void* slaunch(void* V) {
    ((ThreadPool_pthread*)V)->launch();
    return 0;
  }

public:
  ThreadPool_pthread() 
    :start(0), work(0), shutdown(false), maxThreads(0)
  {
    ThreadPool::activeThreads = 1;
    unsigned int num = GaloisRuntime::getSystemThreadPolicy().getNumThreads();
    finish.reinit(num + 1);
    while (num) {
      ++maxThreads;
      --num;
      pthread_t t;
      int rc = pthread_create(&t, 0, &slaunch, this);
      checkResults(rc);
      threads.push_front(t);
    }
  }

  ~ThreadPool_pthread() {
    shutdown = true;
    work = 0;
    __sync_synchronize();
    start.release(threads.size());
    while(!threads.empty()) {
      pthread_t t = threads.front();
      threads.pop_front();
      int rc = pthread_join(t, NULL);
      checkResults(rc);
    }
  }

  virtual void run(Galois::Executable* E) {
    work = E;
    __sync_synchronize();
    ThreadPool::NotifyAware(true);
    work->preRun(activeThreads);
    __sync_synchronize();
    start.release(maxThreads);
    finish.wait();
    work->postRun();
    work = 0;
    ThreadPool::NotifyAware(false);
  }

  virtual unsigned int setActiveThreads(unsigned int num) {
    if (num == 0) {
      activeThreads = 1;
    } else if (num <= maxThreads) {
      activeThreads = num;
    } else {
      activeThreads = maxThreads;
    }
    assert(activeThreads <= maxThreads);
    assert(activeThreads <= threads.size());
    std::cerr << "Threads set to " << num << " using " << activeThreads << " of " << maxThreads << "\n";
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



#endif

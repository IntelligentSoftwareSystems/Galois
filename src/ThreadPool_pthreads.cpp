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
#include <unistd.h>
#include <iostream>
#include <list>
#include <cassert>

using namespace GaloisRuntime;

//! Generic check for pthread functions
static void checkResults(int val, char* errmsg = 0, bool fail = true) {
  if (val) {
    std::cerr << "FAULT " << val;
    if (errmsg)
      std::cerr << ": " << errmsg;
    std::cerr << "\n";
    if (fail)
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

class ThreadPool_pthread : public ThreadPool {
  Semaphore start; // Signal to release threads to run
  Semaphore finish; // want on to block on running threads
  Galois::Executable* work; // Thing to execute
  volatile bool shutdown; // Set and start threads to have them exit
  std::list<pthread_t> threads; // Set of threads
  unsigned int startNum; // Number to release in parallel region

  // Return the number of processors on this hardware
  // This is the maximum number of threads that can be started
  unsigned int numProcessors() {
#ifdef __linux__
    return sysconf(_SC_NPROCESSORS_CONF);
#endif
    reportWarning("Unknown number of processors (assuming 64)");
    return 64;
  }

  void bindToProcessor(int proc) {
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
    unsigned int id = ThreadPool::getMyID();
    bindToProcessor(id - 1);

    while (true) {
      start.acquire();
      if (work)
	(*work)();
      if(shutdown)
	break;
      finish.release();
    }
    finish.release();
  }

  static void* slaunch(void* V) {
    ((ThreadPool_pthread*)V)->launch();
    return 0;
  }

public:
  ThreadPool_pthread() 
    :start(0), finish(0), work(0), shutdown(false), startNum(1)
  {
    unsigned int num = numProcessors();
    while (num) {
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
    start.release(threads.size());
    finish.acquire(threads.size());
    while(!threads.empty()) {
      pthread_t t = threads.front();
      threads.pop_front();
      int rc = pthread_join(t, NULL);
      checkResults(rc);
    }
  }

  virtual void run(Galois::Executable* E) {
    work = E;
    ThreadPool::NotifyAware(true);
    work->preRun(startNum);
    start.release(startNum);
    finish.acquire(startNum);
    work->postRun();
    work = 0;
    ThreadPool::NotifyAware(false);
  }

  virtual unsigned int setMaxThreads(unsigned int num) {
    if (num == 0) {
      startNum = 1;
    } else if (num < threads.size()) {
      startNum = num;
    } else {
      startNum = threads.size();
    }
    return startNum;
  }

  virtual unsigned int size() {
    return threads.size();
  }
};
}

//! Implement the global threadpool
static ThreadPool_pthread pool;

ThreadPool& GaloisRuntime::getSystemThreadPool() {
  return pool;
}



#endif

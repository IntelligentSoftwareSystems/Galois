/*! \file 
 *  \brief pthread thread pool implementation
 */

#include "galois_config.h"
#ifdef WITH_PTHREAD_POOL

#include "Galois/Executable.h"
#include "Galois/Runtime/ThreadPool.h"

#include <semaphore.h>
#include <pthread.h>
#include <stdlib.h>
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
      sem_init(&sem, 0, val);
    }
    ~Semaphore() {
      sem_destroy(&sem);
    }

    void release(int n = 1) {
      while (n) {
	--n;
	sem_post(&sem);
      }
    }

    void acquire(int n = 1) {
      while (n) {
	--n;
	sem_wait(&sem);
      }
    }
    
  };

  class ThreadPool_pthread : public ThreadPool {
    
    Semaphore start;
    Semaphore finish;
    Galois::Executable* work;
    volatile bool shutdown;
    std::list<pthread_t> threads;
    int tmax;

    int numThreads() {
      return threads.size();
    }

    int mkID() {
      return __sync_fetch_and_add(&tmax, 1);
    }

    void launch(void) {
      int myID = mkID();
      while (!shutdown) {
	start.acquire();
	if (!shutdown)
	  (*work)(myID, numThreads());
	finish.release();
      }
    }

    static void* slaunch(void* V) {
      ((ThreadPool_pthread*)V)->launch();
      return 0;
    }

  public:
    ThreadPool_pthread() 
      :start(0), finish(0), work(0), shutdown(false),tmax(0)
    {
      resize(1);
    }

    ~ThreadPool_pthread() {
      resize(0);
    }

    virtual void run(Galois::Executable* E) {
      work = E;
      work->preRun(numThreads());
      start.release(numThreads());
      finish.acquire(numThreads());
      work->postRun();
    }

    virtual void resize(int num) {
      //To make this easy, we just kill everything and try again
      shutdown = true;
      start.release(numThreads());
      finish.acquire(numThreads());
      while(!threads.empty()) {
	pthread_t t = threads.front();
	threads.pop_front();
	int rc = pthread_join(t, NULL);
	checkResults(rc);
      }
      tmax = 0;
      shutdown = false;
      while (num) {
	--num;
	pthread_t t;
	pthread_create(&t, 0, &slaunch, this);
	threads.push_front(t);
      }
    }

    virtual int size() {
      return numThreads();
    }
  };
}


//! Implement the global threadpool
static ThreadPool_pthread pool;

ThreadPool& GaloisRuntime::getSystemThreadPool() {
  return pool;
}

#endif

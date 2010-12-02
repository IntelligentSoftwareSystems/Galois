/*! \file 
 *  \brief pthread thread pool implementation
 */

#ifdef GALOIS_PTHREAD

#include "Galois/Executable.h"
#include "Galois/Runtime/Threads.h"

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

    int numThreads() {
      return threads.size();
    }

    void launch(void) {
#ifdef __linux__
      int id = ThreadPool::getMyID();
      cpu_set_t mask;
      /* CPU_ZERO initializes all the bits in the mask to zero. */
      CPU_ZERO( &mask );
      
      /* CPU_SET sets only the bit corresponding to cpu. */
      CPU_SET( id - 1, &mask );
      
      /* sched_setaffinity returns 0 in success */
      if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 ) {
	std::cerr << "WARNING: Could not set CPU Affinity for thread " << id << ", continuing...\n";
      }
#endif      
      while (!shutdown) {
	start.acquire();
	if (!shutdown)
	  (*work)();
	finish.release();
      }
    }

    static void* slaunch(void* V) {
      ((ThreadPool_pthread*)V)->launch();
      return 0;
    }

  public:
    ThreadPool_pthread() 
      :start(0), finish(0), work(0), shutdown(false)
    {
      resize(1);
    }

    ~ThreadPool_pthread() {
      resize(0);
    }

    virtual void run(Galois::Executable* E) {
      work = E;
      ThreadPool::NotifyAware(numThreads());
      work->preRun(numThreads());
      start.release(numThreads());
      finish.acquire(numThreads());
      work->postRun();
      ThreadPool::NotifyAware(0);
    }

    virtual void resize(int num) {
#ifdef __linux__
      int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
      if (num > NUM_PROCS) {
	num = NUM_PROCS;
	std::cerr << "Capping threads to number of processors (" << num << ")\n";
      }
#endif

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
      ResetThreadNumbers();
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

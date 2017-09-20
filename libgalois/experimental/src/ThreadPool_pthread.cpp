/** pthread thread pool implementation -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "galois/runtime/Sampling.h"
#include "galois/Substrate/ThreadPool.h"
#include "galois/Substrate/EnvCheck.h"
#include "galois/Substrate/HWTopo.h"
#include "galois/Substrate/TID.h"
#include "galois/gIO.h"

#include "boost/utility.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <cassert>

#include <semaphore.h>
#include <pthread.h>

// Forward declare this to avoid including PerThreadStorage.
// We avoid this to stress that the thread Pool MUST NOT depend on PTS.
namespace galois {
namespace runtime {
extern void initPTS();
}
}

using namespace galois::runtime;

namespace {

class Semaphore: private boost::noncopyable {
  sem_t sem;
public:
  explicit Semaphore(int val = 0) {
    if (sem_init(&sem, 0, val))
      GALOIS_DIE("PTHREAD");
  }

  ~Semaphore() {
    if (sem_destroy(&sem))
      GALOIS_DIE("PTHREAD");
  }

  void release(int n = 1) {
    while (n) {
      --n;
      if (sem_post(&sem))
        GALOIS_DIE("PTHREAD");
    }
  }

  void acquire(int n = 1) {
    while (n) {
      --n;
      int rc;
      while (((rc = sem_wait(&sem)) < 0) && (errno == EINTR)) { }
      if (rc)
        GALOIS_DIE("PTHREAD");
    }
  }
};

class ThreadPool_pthread : public ThreadPool {
  pthread_t* threads; // Set of threads
  Semaphore* starts;  // Signal to release threads to run

  virtual void threadWakeup(unsigned n) {
    starts[n].release();
  }

  virtual void threadWait(unsigned tid) {
    starts[tid].acquire();
  }

  static void* slaunch(void* V) {
    ThreadPool_pthread* TP = (ThreadPool_pthread*)V;
    static unsigned next = 0;
    unsigned tid = __sync_add_and_fetch(&next, 1);
    TP->threadLoop(tid);
    return 0;
  }
  
public:
  ThreadPool_pthread():
    ThreadPool(galois::runtime::LL::getMaxThreads())
  {
    starts = new Semaphore[maxThreads];
    threads = new pthread_t[maxThreads];

    for (unsigned i = 1; i < maxThreads; ++i) {
      if (pthread_create(&threads[i], 0, &slaunch, this))
        GALOIS_DIE("PTHREAD");
    }
    decascade(0);
  }

  virtual ~ThreadPool_pthread() {
    destroyCommon();
    for (unsigned i = 1; i < maxThreads; ++i) {
      if (pthread_join(threads[i], NULL))
        GALOIS_DIE("PTHREAD");
    }
    delete [] starts;
    delete [] threads;
  }
};

} // end namespace

//! Implement the global threadpool
ThreadPool& galois::runtime::getThreadPool() {
  static ThreadPool_pthread pool;
  return pool;
}

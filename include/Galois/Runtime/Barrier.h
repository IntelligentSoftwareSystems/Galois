/** Barriers -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_BARRIER_H
#define GALOIS_RUNTIME_BARRIER_H

#include "PerCPU.h"

#include <limits>
#include <cstdio>

#include <stdlib.h>
#include <pthread.h>

namespace GaloisRuntime {

class PthreadBarrier {
  pthread_barrier_t bar;

  void checkResults(int val) {
    if (val) {
      perror("PTHREADS: ");
      assert(0 && "PThread check");
      abort();
    }
  }
public:
  PthreadBarrier() {
    //uninitialized barriers block a lot of threads to help with debugging
    int rc = pthread_barrier_init(&bar, 0, std::numeric_limits<int>::max());
    checkResults(rc);
  }

  PthreadBarrier(unsigned int val) {
    int rc = pthread_barrier_init(&bar, 0, val);
    checkResults(rc);
  }

  ~PthreadBarrier() {
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

//! Simple busy waiting barrier, not cyclic
class SimpleBarrier {
  struct PLD {
    int count;
    int total;
    PLD(): count(0) { }
  };

  struct TLD {
    volatile int flag;
    TLD(): flag(0) { }
  };

  volatile int globalTotal;
  PerCPU<TLD> tlds;
  PerLevel<PLD> plds;
  int size;

  void cascade(int tid);

public:
  SimpleBarrier(): globalTotal(0), size(-1) { }

  //! Not thread-safe and should only be called when no threads are in wait()/increment()
  void reinit(int val, int init);

  void increment();
  void wait();
  void barrier();
};

//! Busy waiting barrier biased towards getting master thread out as soon
//! as possible. Cyclic.
class FastBarrier {
  SimpleBarrier in;
  SimpleBarrier out;
  int size;

public:
  FastBarrier(): size(-1) { }
  FastBarrier(unsigned int val): size(-1) { reinit(val); }

  void reinit(int val);
  void wait(); 
};

}

#endif

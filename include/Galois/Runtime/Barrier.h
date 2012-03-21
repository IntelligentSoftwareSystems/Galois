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

  void pause() {
#if defined(__i386__) || defined(__amd64__)
    asm volatile ( "pause");
#endif
  }

  void compilerBarrier() {
    asm volatile ("":::"memory");
  }

  void cascade(int tid) {
    int multiple = 2;
    for (int i = 0; i < multiple; ++i) {
      int n = tid * multiple + i;
      if (n < size && n != 0)
        tlds.get(n).flag = 1;
    }
  }

public:
  SimpleBarrier(): globalTotal(0), size(-1) { }

  //! Not thread-safe and should only be called when no threads are in wait()/increment()
  void reinit(int val, int init) {
    assert(val > 0);

    if (val != size) {
      for (unsigned i = 0; i < plds.size(); ++i)
        plds.get(i).total = 0;
      for (int i = 0; i < val; ++i) {
        int j = LL::getPackageForThreadInternal(i);
        ++plds.get(j).total;
      }

      size = val;
    }

    globalTotal = init;
  }

  void increment() {
    PLD& pld = plds.get();
    int total = pld.total;

    if (__sync_add_and_fetch(&pld.count, 1) == total) {
      pld.count = 0;
      compilerBarrier();
      __sync_add_and_fetch(&globalTotal, total);
    }
  }

  void wait() {
    int tid = (int) LL::getTID();
    TLD& tld = tlds.get(tid);
    if (tid == 0) {
      while (globalTotal < size) {
        pause();
      }
    } else {
      while (!tld.flag) {
        pause();
      }
    }
  }

  void barrier() {
    assert(size > 0);

    int tid = (int) LL::getTID();
    TLD& tld = tlds.get(tid);

    if (tid == 0) {
      while (globalTotal < size) {
        pause();
      }

      globalTotal = 0;
      tld.flag = 0;
      compilerBarrier();
      cascade(tid);
    } else {
      while (!tld.flag) {
        pause();
      }

      tld.flag = 0;
      compilerBarrier();
      cascade(tid);
    }
  }
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

  void reinit(int val) {
    if (val != size) {
      if (size != -1)
        out.wait();

      in.reinit(val, 0);
      out.reinit(val, val);
      val = size;
    }
  }

  void wait() {
    out.barrier();
    in.increment();
    in.barrier();
    out.increment();
  }
};

}

#endif

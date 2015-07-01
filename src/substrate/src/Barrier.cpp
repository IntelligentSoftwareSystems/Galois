/** Galois barrier -*- C++ -*-
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
 * @section Description
 *
 * Barriers
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/ll/gio.h"
#include <pthread.h>

#include <cstdlib>
#include <cstdio>

class OneWayBarrier: public Galois::Runtime::Barrier {
  pthread_mutex_t lock;
  pthread_cond_t cond;
  unsigned count; 
  unsigned total;

public:
  OneWayBarrier(unsigned p = Galois::Runtime::activeThreads) {
    if (pthread_mutex_init(&lock, NULL))
      GALOIS_DIE("PTHREAD");
    if (pthread_cond_init(&cond, NULL))
      GALOIS_DIE("PTHREAD");
    reinit(p);
  }
  
  virtual ~OneWayBarrier() {
    if (pthread_mutex_destroy(&lock))
      GALOIS_DIE("PTHREAD");
    if (pthread_cond_destroy(&cond))
      GALOIS_DIE("PTHREAD");
  }

  virtual void reinit(unsigned val) {
    count = 0;
    total = val;
  }

  virtual void wait() {
    if (pthread_mutex_lock(&lock))
      GALOIS_DIE("PTHREAD");
    count += 1;
    while (count < total) {
      if (pthread_cond_wait(&cond, &lock))
        GALOIS_DIE("PTHREAD");
    }
    if (pthread_cond_broadcast(&cond))
        GALOIS_DIE("PTHREAD");

    if (pthread_mutex_unlock(&lock))
      GALOIS_DIE("PTHREAD");
  }

  virtual const char* name() const { return "OneWayBarrier"; }
};

class SimpleBarrier: public Galois::Runtime::Barrier {
  OneWayBarrier barrier1;
  OneWayBarrier barrier2;
  unsigned total;
public:
  SimpleBarrier(unsigned p = Galois::Runtime::activeThreads): barrier1(p), barrier2(p), total(p) { }

  virtual ~SimpleBarrier() { }

  virtual void reinit(unsigned val) {
    total = val;
    barrier1.reinit(val);
    barrier2.reinit(val);
  }

  virtual void wait() {
    barrier1.wait();
    if (Galois::Runtime::LL::getTID() == 0)
      barrier1.reinit(total);
    barrier2.wait();
    if (Galois::Runtime::LL::getTID() == 0)
      barrier2.reinit(total);
  }

  virtual const char* name() const { return "SimpleBarrier"; }

};


Galois::Runtime::Barrier::~Barrier() {}


Galois::Runtime::Barrier* Galois::Runtime::createSimpleBarrier() {
  //FIXME: allow pthread barrier again
// #if _POSIX_BARRIERS > 1
//   return new PthreadBarrier();
// #else
  return new SimpleBarrier();
// #endif
}

Galois::Runtime::Barrier& Galois::Runtime::getSystemBarrier() {
  return benchmarking::getTopoBarrier();
}


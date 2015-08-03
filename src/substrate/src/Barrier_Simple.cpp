/** Galois barrier -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/CompilerSpecific.h"
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

class SimpleBarrier: public Galois::Substrate::Barrier {
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

std::unique_ptr<Galois::Substrate::Barrier> Galois::Substrate::createSimpleBarrier() {
  //FIXME: allow pthread barrier again
// #if _POSIX_BARRIERS > 1
//   return new PthreadBarrier();
// #else
  return new SimpleBarrier();
// #endif
}

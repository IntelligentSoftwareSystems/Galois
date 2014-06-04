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
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ActiveThreads.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/ll/gio.h"
#include <pthread.h>

#if defined(_POSIX_BARRIERS)

namespace {

class PthreadBarrier: public Galois::Runtime::Barrier {
  pthread_barrier_t bar;

public:
  PthreadBarrier() {
    if (pthread_barrier_init(&bar, 0, ~0))
      GALOIS_DIE("PTHREAD");
  }
  
  PthreadBarrier(unsigned int val) {
    if (pthread_barrier_init(&bar, 0, val))
      GALOIS_DIE("PTHREAD");
  }

  virtual ~PthreadBarrier() {
    if (pthread_barrier_destroy(&bar))
      GALOIS_DIE("PTHREAD");
  }

  virtual void reinit(unsigned val) {
    if (pthread_barrier_destroy(&bar))
      GALOIS_DIE("PTHREAD");
    if (pthread_barrier_init(&bar, 0, val))
      GALOIS_DIE("PTHREAD");
  }

  virtual void wait() {
    int rc = pthread_barrier_wait(&bar);
    if (rc && rc != PTHREAD_BARRIER_SERIAL_THREAD)
      GALOIS_DIE("PTHREAD");
  }

  virtual const char* name() const { return "PthreadBarrier"; }
};

}

Galois::Runtime::Barrier& Galois::Runtime::benchmarking::getPthreadBarrier() {
  static PthreadBarrier b;
  static unsigned num = ~0;
  if (activeThreads != num) {
    num = activeThreads;
    b.reinit(num);
  }
  return b;
}

#endif

/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/substrate/Barrier.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/gIO.h"

#if defined(GALOIS_HAVE_PTHREAD)

#include <unistd.h>
#include <pthread.h>

#endif

#if defined(GALOIS_HAVE_PTHREAD) && defined(_POSIX_BARRIERS) &&                \
    (_POSIX_BARRIERS > 0)

namespace {

class PthreadBarrier : public galois::substrate::Barrier {
  pthread_barrier_t bar;

public:
  PthreadBarrier() {
    if (pthread_barrier_init(&bar, 0, ~0))
      GALOIS_DIE("PTHREAD");
  }

  PthreadBarrier(unsigned int v) {
    if (pthread_barrier_init(&bar, 0, v))
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

} // namespace

std::unique_ptr<galois::substrate::Barrier>
galois::substrate::createPthreadBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new PthreadBarrier(activeThreads));
}

#else

std::unique_ptr<galois::substrate::Barrier>
galois::substrate::createPthreadBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(nullptr);
}

#endif

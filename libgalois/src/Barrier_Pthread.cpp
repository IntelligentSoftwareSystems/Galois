/** Galois barrier -*- C++ -*-
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
 * @section Description
 *
 * Pthread based Barriers
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Galois/Substrate/BarrierImpl.h"
#include "Galois/Substrate/CompilerSpecific.h"
#include "Galois/gIO.h"

#if defined(GALOIS_HAVE_PTHREAD)

#include <unistd.h>
#include <pthread.h>

#endif


#if defined(GALOIS_HAVE_PTHREAD) && defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS > 0)

namespace {

class PthreadBarrier: public Galois::Substrate::Barrier {
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

}

std::unique_ptr<Galois::Substrate::Barrier> Galois::Substrate::createPthreadBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new PthreadBarrier(activeThreads));
}

#else

std::unique_ptr<Galois::Substrate::Barrier> Galois::Substrate::createPthreadBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(nullptr);
}

#endif


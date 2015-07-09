/** Barriers -*- C++ -*-
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
 * @section Description
 *
 * Public API for interacting with barriers
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_SUBSTRATE_BARRIER_H
#define GALOIS_SUBSTRATE_BARRIER_H

#include <memory>

namespace Galois {
namespace Substrate {

class Barrier {
public:
  virtual ~Barrier();

  //not safe if any thread is in wait
  virtual void reinit(unsigned val) = 0;

  //Wait at this barrier
  virtual void wait() = 0;

  //wait at this barrier
  void operator()(void) { wait(); }

  //barrier type.
  virtual const char* name() const = 0;
};

/**
 * Have a pre-instantiated barrier available for use.
 * This is initialized to the current activeThreads. This barrier
 * is designed to be fast and should be used in the common
 * case. 
 *
 * However, there is a race if the number of active threads
 * is modified after using this barrier: some threads may still
 * be in the barrier while the main thread reinitializes this
 * barrier to the new number of active threads. If that may
 * happen, use {@link createSimpleBarrier()} instead. 
 */
Barrier& getSystemBarrier(unsigned activeThreads);

/**
 * Create specific types of barriers.  For benchmarking only.  Use
 * getSystemBarrier() for all production code
 */
namespace benchmarking {
Barrier& getPthreadBarrier(unsigned);
Barrier& getMCSBarrier(unsigned);
Barrier& getTopoBarrier(unsigned);
Barrier& getCountingBarrier(unsigned);
Barrier& getDisseminationBarrier(unsigned);
}

/**
 * Creates a new simple barrier. This barrier is not designed to be fast but
 * does gaurantee that all threads have left the barrier before returning
 * control. Useful when the number of active threads is modified to avoid a
 * race in {@link getSystemBarrier()}.  Client is reponsible for deallocating
 * returned barrier.
 */
std::unique_ptr<Barrier> createSimpleBarrier();

} // end namespace Substrate
} // end namespace Galois

#endif

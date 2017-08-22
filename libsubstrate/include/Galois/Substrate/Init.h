/** Initialization -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * API for Initialization of Substrate Library components
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_SUBSTRATE_INIT_H
#define GALOIS_SUBSTRATE_INIT_H

#include "Galois/gIO.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/BarrierImpl.h"


namespace Galois {
namespace Substrate {


/**
 * Initializes the Substrate library components
 */
void init(void);

/**
 * Destroys the Substrate library components
 */
void kill(void);

/**
 * return a reference to system thread pool
 */
ThreadPool& getThreadPool();

/**
 * return a reference to system barrier 
 */
Barrier& getBarrier(unsigned numT);


namespace internal {

void initBarrier();

void killBarrier();

struct BarrierInstance {
  unsigned m_num_threads;
  std::unique_ptr<Barrier> m_barrier;

  BarrierInstance(void) {
    m_num_threads = getThreadPool().getMaxThreads();
    m_barrier = createTopoBarrier(m_num_threads);
  }

  Barrier& get(unsigned numT) {
    GALOIS_ASSERT(numT > 0, "Substrate::getBarrier() number of threads must be > 0");

    numT = std::min(numT, getThreadPool().getMaxUsableThreads());
    numT = std::max(numT, 1u);

    if (numT != m_num_threads) {
      m_num_threads = numT;
      m_barrier->reinit(numT);
    }

    return *m_barrier;
  }

};
} // end namespace internal


}
}

#endif // GALOIS_SUBSTRATE_INIT_H


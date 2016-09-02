/** Barriers -*- C++ -*-
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
 * Public API for interacting with barriers
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_SUBSTRATE_BARRIERIMPL_H
#define GALOIS_SUBSTRATE_BARRIERIMPL_H

#include <memory>

#include "Galois/Substrate/Barrier.h"

namespace Galois {
namespace Substrate {

/**
 * Create specific types of barriers.  For benchmarking only.  Use
 * getSystemBarrier() for all production code
 */
std::unique_ptr<Barrier> createPthreadBarrier(unsigned);
std::unique_ptr<Barrier> createMCSBarrier(unsigned);
std::unique_ptr<Barrier> createTopoBarrier(unsigned);
std::unique_ptr<Barrier> createCountingBarrier(unsigned);
std::unique_ptr<Barrier> createDisseminationBarrier(unsigned);

/**
 * Creates a new simple barrier. This barrier is not designed to be fast but
 * does gaurantee that all threads have left the barrier before returning
 * control. Useful when the number of active threads is modified to avoid a
 * race in {@link getSystemBarrier()}.  Client is reponsible for deallocating
 * returned barrier.
 */
std::unique_ptr<Barrier> createSimpleBarrier(unsigned);

} // end namespace Substrate
} // end namespace Galois

#endif

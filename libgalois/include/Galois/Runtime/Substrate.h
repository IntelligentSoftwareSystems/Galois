/** Stuff from libSubstrate which we only want to instantiate once -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Things which are imported from libsubstrate.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_SUBSTRATE_H
#define GALOIS_RUNTIME_SUBSTRATE_H

#include "Galois/Substrate/Barrier.h"

namespace galois {
namespace runtime {

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
Substrate::Barrier& getBarrier(unsigned activeThreads);


} // end namespace runtime
} // end namespace galois

#endif

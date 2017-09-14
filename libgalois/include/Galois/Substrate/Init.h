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
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/Termination.h"

namespace Galois {
namespace Substrate {

class SharedMemSubstrate {

  // Order is critical here
  ThreadPool m_tpool;

  internal::LocalTerminationDetection<>*  m_termPtr;
  internal::BarrierInstance<>*  m_biPtr;

public:

  /**
   * Initializes the Substrate library components
   */
  SharedMemSubstrate();

  /**
   * Destroys the Substrate library components
   */
  ~SharedMemSubstrate();

};


}
}

#endif // GALOIS_SUBSTRATE_INIT_H


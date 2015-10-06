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
#ifndef GALOIS_SUBSTRATE_BARRIER_H
#define GALOIS_SUBSTRATE_BARRIER_H

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

} // end namespace Substrate
} // end namespace Galois

#endif

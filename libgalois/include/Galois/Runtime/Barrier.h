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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Public API for interacting with barriers
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_BARRIER_H
#define GALOIS_RUNTIME_BARRIER_H

#include "Galois/Runtime/PerThreadStorage.h"

#include <memory>
#include <atomic>

namespace Galois {
namespace Runtime {

class Barrier {
  struct treenode {
    //vpid is Galois::Runtime::LL::getTID()

    //package binary tree
    treenode* parentpointer; //null of vpid == 0
    treenode* childpointers[2];

    //waiting values:
    unsigned havechild;
    std::atomic<unsigned> childnotready;

    //signal values
    std::atomic<unsigned> parentsense;

  };

  PerPackageStorage<treenode> nodes;
  PerThreadStorage<unsigned> sense;

  void _reinit(unsigned);

public:
  Barrier(unsigned);
  ~Barrier();

  //not safe if any thread is in wait
  void reinit(unsigned val);

  //Wait at this barrier
  void wait();

  //wait at this barrier
  void operator()(void) { wait(); }
};

} // end namespace Runtime
} // end namespace Galois

#endif

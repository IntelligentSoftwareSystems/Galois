/** Galois barrier -*- C++ -*-
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
 * MCS Barriers
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */


#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Substrate/Barrier.h"

#include <atomic>

namespace {

class MCSBarrier: public Galois::Substrate::Barrier {
  struct treenode {
    //vpid is Galois::Runtime::LL::getTID()
    std::atomic<bool>* parentpointer; //null for vpid == 0
    std::atomic<bool>* childpointers[2];
    bool havechild[4];

    std::atomic<bool> childnotready[4];
    std::atomic<bool> parentsense;
    bool sense;
    treenode() {}
  };

  Galois::Runtime::PerThreadStorage<treenode> nodes;
  
  void _reinit(unsigned P) {
    for (unsigned i = 0; i < nodes.size(); ++i) {
      treenode& n = *nodes.getRemote(i);
      n.sense = true;
      n.parentsense = false;
      for (int j = 0; j < 4; ++j)
	n.childnotready[j] = n.havechild[j] = ((4*i+j+1) < P);
      n.parentpointer = (i == 0) ? 0 :
	&nodes.getRemote((i-1)/4)->childnotready[(i-1)%4];
      n.childpointers[0] = ((2*i + 1) >= P) ? 0 :
	&nodes.getRemote(2*i+1)->parentsense;
      n.childpointers[1] = ((2*i + 2) >= P) ? 0 :
	&nodes.getRemote(2*i+2)->parentsense;
    }
  }

public:
  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
    treenode& n = *nodes.getLocal();
    while (n.childnotready[0] || n.childnotready[1] || 
	   n.childnotready[2] || n.childnotready[3]) {
      Galois::Substrate::asmPause();
    }
    for (int i = 0; i < 4; ++i)
      n.childnotready[i] = n.havechild[i];
    if (n.parentpointer) {
      //FIXME: make sure the compiler doesn't do a RMW because of the as-if rule
      *n.parentpointer = false;
      while(n.parentsense != n.sense) {
	Galois::Substrate::asmPause();
      }
    }
    //signal children in wakeup tree
    if (n.childpointers[0])
      *n.childpointers[0] = n.sense;
    if (n.childpointers[1])
      *n.childpointers[1] = n.sense;
    n.sense = !n.sense;
  }

  virtual const char* name() const { return "MCSBarrier"; }
};

}

Galois::Substrate::Barrier& Galois::Substrate::benchmarking::getMCSBarrier() {
  static MCSBarrier b;
  static unsigned num = ~0;
  if (Runtime::activeThreads != num) {
    num = Runtime::activeThreads;
    b.reinit(num);
  }
  return b;
}

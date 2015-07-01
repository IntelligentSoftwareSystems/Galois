/** Galois barrier -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * MCS Barriers
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */


#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"

namespace {

class MCSBarrier: public Galois::Runtime::Barrier {
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
  MCSBarrier(unsigned P = Galois::Runtime::activeThreads) {
    _reinit(P);
  }

  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
    treenode& n = *nodes.getLocal();
    while (n.childnotready[0] || n.childnotready[1] || 
	   n.childnotready[2] || n.childnotready[3]) {
      Galois::Runtime::LL::asmPause();
    }
    for (int i = 0; i < 4; ++i)
      n.childnotready[i] = n.havechild[i];
    if (n.parentpointer) {
      //FIXME: make sure the compiler doesn't do a RMW because of the as-if rule
      *n.parentpointer = false;
      while(n.parentsense != n.sense) {
	Galois::Runtime::LL::asmPause();
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

Galois::Runtime::Barrier& Galois::Runtime::benchmarking::getMCSBarrier() {
  static MCSBarrier b;
  static unsigned num = ~0;
  if (activeThreads != num) {
    num = activeThreads;
    b.reinit(num);
  }
  return b;
}

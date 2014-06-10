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
 * Topologoy Hybrid Shared-MCS Barriers
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ActiveThreads.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

namespace {

class TopoBarrier : public Galois::Runtime::Barrier {
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

  Galois::Runtime::PerPackageStorage<treenode> nodes;
  Galois::Runtime::PerThreadStorage<unsigned> sense;

  void _reinit(unsigned P) {
    unsigned pkgs = Galois::Runtime::LL::getMaxPackageForThread(P-1) + 1;
    for (unsigned i = 0; i < pkgs; ++i) {
      treenode& n = *nodes.getRemoteByPkg(i);
      n.childnotready = 0;
      n.havechild = 0;
      for (int j = 0; j < 4; ++j) {
	if ((4*i+j+1) < pkgs) {
	  ++n.childnotready;
          ++n.havechild;
	}
      }
      for (unsigned j = 0; j < P; ++j) {
	if (Galois::Runtime::LL::getPackageForThread(j) == i && !Galois::Runtime::LL::isPackageLeader(j)) {
          ++n.childnotready;
          ++n.havechild;
        }
      }
      n.parentpointer = (i == 0) ? 0 : nodes.getRemoteByPkg((i-1)/4);
      n.childpointers[0] = ((2*i + 1) >= pkgs) ? 0 : nodes.getRemoteByPkg(2*i+1);
      n.childpointers[1] = ((2*i + 2) >= pkgs) ? 0 : nodes.getRemoteByPkg(2*i+2);
      n.parentsense = 0;
    }
    for (unsigned i = 0; i < P; ++i)
      *sense.getRemote(i) = 1;
#if 0
    for (unsigned i = 0; i < pkgs; ++i) {
      treenode& n = *nodes.getRemoteByPkg(i);
      Galois::Runtime::LL::gPrint(i, 
          " this ", &n,
          " parent ", n.parentpointer, 
          " child[0] ", n.childpointers[0],
          " child[1] ", n.childpointers[1],
          " havechild ", n.havechild,
          "\n");
    }
#endif
  }

public:
  TopoBarrier(unsigned val = Galois::Runtime::activeThreads) {
    _reinit(val);
  }

  //not safe if any thread is in wait
  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
    unsigned id = Galois::Runtime::LL::getTID();
    treenode& n = *nodes.getLocal();
    unsigned& s = *sense.getLocal();
    bool leader = Galois::Runtime::LL::isPackageLeaderForSelf(id);
    //completion tree
    if (leader) {
      while (n.childnotready) { Galois::Runtime::LL::asmPause(); }
      n.childnotready = n.havechild;
      if (n.parentpointer) {
	--n.parentpointer->childnotready;
      }
    } else {
      --n.childnotready;
    }
    
    //wait for signal
    if (id != 0) {
      while (n.parentsense != s) {
	Galois::Runtime::LL::asmPause();
      }
    }
    
    //signal children in wakeup tree
    if (leader) {
      if (n.childpointers[0])
	n.childpointers[0]->parentsense = s;
      if (n.childpointers[1])
	n.childpointers[1]->parentsense = s;
      if (id == 0)
	n.parentsense = s;
    }
    ++s;
  }

  virtual const char* name() const { return "TopoBarrier"; }

};

}

Galois::Runtime::Barrier& Galois::Runtime::benchmarking::getTopoBarrier() {
  static TopoBarrier b;
  static unsigned num = ~0;
  if (activeThreads != num) {
    num = activeThreads;
    b.reinit(num);
  }
  return b;
}


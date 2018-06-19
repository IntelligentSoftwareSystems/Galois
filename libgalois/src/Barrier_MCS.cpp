/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/CompilerSpecific.h"

#include <atomic>

namespace {

class MCSBarrier : public galois::substrate::Barrier {
  struct treenode {
    // vpid is galois::runtime::LL::getTID()
    std::atomic<bool>* parentpointer; // null for vpid == 0
    std::atomic<bool>* childpointers[2];
    bool havechild[4];

    std::atomic<bool> childnotready[4];
    std::atomic<bool> parentsense;
    bool sense;
    treenode() {}
    treenode(const treenode& rhs)
        : parentpointer(rhs.parentpointer), sense(rhs.sense) {
      childpointers[0] = rhs.childpointers[0];
      childpointers[1] = rhs.childpointers[1];
      for (int i = 0; i < 4; ++i) {
        havechild[i]     = rhs.havechild[i];
        childnotready[i] = rhs.childnotready[i].load();
      }
      parentsense = rhs.parentsense.load();
    }
  };

  std::vector<galois::substrate::CacheLineStorage<treenode>> nodes;

  void _reinit(unsigned P) {
    nodes.resize(P);
    for (unsigned i = 0; i < P; ++i) {
      treenode& n   = nodes.at(i).get();
      n.sense       = true;
      n.parentsense = false;
      for (int j = 0; j < 4; ++j)
        n.childnotready[j] = n.havechild[j] = ((4 * i + j + 1) < P);
      n.parentpointer =
          (i == 0) ? 0
                   : &nodes.at((i - 1) / 4).get().childnotready[(i - 1) % 4];
      n.childpointers[0] =
          ((2 * i + 1) >= P) ? 0 : &nodes.at(2 * i + 1).get().parentsense;
      n.childpointers[1] =
          ((2 * i + 2) >= P) ? 0 : &nodes.at(2 * i + 2).get().parentsense;
    }
  }

public:
  MCSBarrier(unsigned v) { _reinit(v); }

  virtual void reinit(unsigned val) { _reinit(val); }

  virtual void wait() {
    treenode& n = nodes.at(galois::substrate::ThreadPool::getTID()).get();
    while (n.childnotready[0] || n.childnotready[1] || n.childnotready[2] ||
           n.childnotready[3]) {
      galois::substrate::asmPause();
    }
    for (int i = 0; i < 4; ++i)
      n.childnotready[i] = n.havechild[i];
    if (n.parentpointer) {
      // FIXME: make sure the compiler doesn't do a RMW because of the as-if
      // rule
      *n.parentpointer = false;
      while (n.parentsense != n.sense) {
        galois::substrate::asmPause();
      }
    }
    // signal children in wakeup tree
    if (n.childpointers[0])
      *n.childpointers[0] = n.sense;
    if (n.childpointers[1])
      *n.childpointers[1] = n.sense;
    n.sense = !n.sense;
  }

  virtual const char* name() const { return "MCSBarrier"; }
};

} // namespace

std::unique_ptr<galois::substrate::Barrier>
galois::substrate::createMCSBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new MCSBarrier(activeThreads));
}

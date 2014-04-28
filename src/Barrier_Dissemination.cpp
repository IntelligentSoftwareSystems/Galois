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
 * Dissemination Barriers
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */


#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"


namespace {

#define FAST_LOG2(x) (sizeof(unsigned long)*8 - 1 - __builtin_clzl((unsigned long)(x)))
#define FAST_LOG2_UP(x) (((x) - (1 << FAST_LOG2(x))) ? FAST_LOG2(x) + 1 : FAST_LOG2(x))


class DisseminationBarrier: public Galois::Runtime::Barrier {

  struct node {
    bool flag[2];
    node* partner;
  };

  struct LocalData {
    int parity;
    bool sense;
    std::array<node,32> myflags;
  };

  Galois::Runtime::PerThreadStorage<LocalData> nodes;
  unsigned LogP;

  void _reinit(unsigned P) {
    LogP = FAST_LOG2_UP(P);
    for (unsigned i = 0; i < P; ++i) {
      auto& lhs = *nodes.getRemote(i);
      lhs.parity = 0;
      lhs.sense = true;
      for (auto& n : lhs.myflags)
        n.flag[0] = n.flag[1] = false;
      int d = 1;
      for (unsigned j = 0; j < LogP; ++j) {
        auto& rhs = *nodes.getRemote((i+d) % P);
        lhs.myflags[j].partner = &rhs.myflags[j];
        d *= 2;
      }
    }
  }

public:
  DisseminationBarrier(unsigned P = Galois::Runtime::activeThreads) {
    _reinit(P);
  }

  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
    auto& ld = *nodes.getLocal();
    auto& sense = ld.sense;
    auto& parity = ld.parity;
    for (unsigned r = 0; r < LogP; ++r) {
      ld.myflags[r].partner->flag[parity] = sense;
      while (ld.myflags[r].flag[parity] != sense) { Galois::Runtime::LL::asmPause(); }
    }
    if (parity == 1)
      sense = !ld.sense;
    parity = 1 - parity;
  }

  virtual const char* name() const { return "DisseminationBarrier"; }
};

}

Galois::Runtime::Barrier& Galois::Runtime::benchmarking::getDisseminationBarrier() {
  static DisseminationBarrier b;
  static unsigned num = ~0;
  if (activeThreads != num) {
    num = activeThreads;
    b.reinit(num);
  }
  return b;
}

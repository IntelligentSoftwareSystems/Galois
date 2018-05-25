/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#define FAST_LOG2(x) (sizeof(unsigned long)*8 - 1 - __builtin_clzl((unsigned long)(x)))
#define FAST_LOG2_UP(x) (((x) - (1 << FAST_LOG2(x))) ? FAST_LOG2(x) + 1 : FAST_LOG2(x))


class DisseminationBarrier: public galois::substrate::Barrier {

  struct node {
    std::atomic<int> flag[2];
    node* partner;
    node() :partner(nullptr) {}
    node(const node& rhs) :partner(rhs.partner) {
      flag[0] = rhs.flag[0].load();
      flag[1] = rhs.flag[1].load();
    }
  };

  struct LocalData {
    int parity;
    int sense;
    node myflags[32];
    //std::array<node, 32> myflags;
  };

  std::vector<galois::substrate::CacheLineStorage<LocalData> > nodes;
  unsigned LogP;


  void _reinit(unsigned P) {
    LogP = FAST_LOG2_UP(P);
    nodes.resize(P);
    for (unsigned i = 0; i < P; ++i) {
      LocalData& lhs = nodes.at(i).get();
      lhs.parity = 0;
      lhs.sense = 1;
      for (unsigned j = 0; j < sizeof(lhs.myflags)/sizeof(*lhs.myflags); ++j)
        lhs.myflags[j].flag[0] = lhs.myflags[j].flag[1] = 0;

      int d = 1;
      for (unsigned j = 0; j < LogP; ++j) {
        LocalData& rhs = nodes.at((i+d) % P).get();
        lhs.myflags[j].partner = &rhs.myflags[j];
        d *= 2;
      }
    }
  }

public:

  DisseminationBarrier(unsigned v) {
    _reinit(v);
  }

  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
    auto& ld = nodes.at(galois::substrate::ThreadPool::getTID()).get();
    auto& sense = ld.sense;
    auto& parity = ld.parity;
    for (unsigned r = 0; r < LogP; ++r) {
      ld.myflags[r].partner->flag[parity] = sense;
      while (ld.myflags[r].flag[parity] != sense) { galois::substrate::asmPause(); }
    }
    if (parity == 1)
      sense = 1 - ld.sense;
    parity = 1 - parity;
  }

  virtual const char* name() const { return "DisseminationBarrier"; }
};

}

std::unique_ptr<galois::substrate::Barrier> galois::substrate::createDisseminationBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new DisseminationBarrier(activeThreads));
}

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
 * Dissemination Barriers
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */


#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Substrate/Barrier.h"

#include <atomic>

namespace {

#define FAST_LOG2(x) (sizeof(unsigned long)*8 - 1 - __builtin_clzl((unsigned long)(x)))
#define FAST_LOG2_UP(x) (((x) - (1 << FAST_LOG2(x))) ? FAST_LOG2(x) + 1 : FAST_LOG2(x))


class DisseminationBarrier: public Galois::Substrate::Barrier {

  struct node {
    std::atomic<int> flag[2];
    node* partner;
    node() :partner(nullptr) {}
  };

  struct LocalData {
    int parity;
    int sense;
    node myflags[32];
    //std::array<node, 32> myflags;
  };

  Galois::Runtime::PerThreadStorage<LocalData> nodes;
  unsigned LogP;

  void _reinit(unsigned P) {
    LogP = FAST_LOG2_UP(P);
    for (unsigned i = 0; i < P; ++i) {
      LocalData& lhs = *nodes.getRemote(i);
      lhs.parity = 0;
      lhs.sense = 1;
      for (unsigned j = 0; j < sizeof(lhs.myflags)/sizeof(*lhs.myflags); ++j)
        lhs.myflags[j].flag[0] = lhs.myflags[j].flag[1] = 0;

      int d = 1;
      for (unsigned j = 0; j < LogP; ++j) {
        LocalData& rhs = *nodes.getRemote((i+d) % P);
        lhs.myflags[j].partner = &rhs.myflags[j];
        d *= 2;
      }
    }
  }

public:

  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
    auto& ld = *nodes.getLocal();
    auto& sense = ld.sense;
    auto& parity = ld.parity;
    for (unsigned r = 0; r < LogP; ++r) {
      ld.myflags[r].partner->flag[parity] = sense;
      while (ld.myflags[r].flag[parity] != sense) { Galois::Substrate::asmPause(); }
    }
    if (parity == 1)
      sense = 1 - ld.sense;
    parity = 1 - parity;
  }

  virtual const char* name() const { return "DisseminationBarrier"; }
};

}

Galois::Substrate::Barrier& Galois::Substrate::benchmarking::getDisseminationBarrier() {
  static DisseminationBarrier b;
  static unsigned num = ~0;
  if (Runtime::activeThreads != num) {
    num = Runtime::activeThreads;
    b.reinit(num);
  }
  return b;
}

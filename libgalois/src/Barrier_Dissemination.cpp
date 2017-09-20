/** Galois barrier -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * Dissemination Barriers
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */


#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/CompilerSpecific.h"

#include <atomic>

namespace {

#define FAST_LOG2(x) (sizeof(unsigned long)*8 - 1 - __builtin_clzl((unsigned long)(x)))
#define FAST_LOG2_UP(x) (((x) - (1 << FAST_LOG2(x))) ? FAST_LOG2(x) + 1 : FAST_LOG2(x))


class DisseminationBarrier: public galois::Substrate::Barrier {

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

  std::vector<galois::Substrate::CacheLineStorage<LocalData> > nodes;
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
    auto& ld = nodes.at(galois::Substrate::ThreadPool::getTID()).get();
    auto& sense = ld.sense;
    auto& parity = ld.parity;
    for (unsigned r = 0; r < LogP; ++r) {
      ld.myflags[r].partner->flag[parity] = sense;
      while (ld.myflags[r].flag[parity] != sense) { galois::Substrate::asmPause(); }
    }
    if (parity == 1)
      sense = 1 - ld.sense;
    parity = 1 - parity;
  }

  virtual const char* name() const { return "DisseminationBarrier"; }
};

}

std::unique_ptr<galois::Substrate::Barrier> galois::Substrate::createDisseminationBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new DisseminationBarrier(activeThreads));
}

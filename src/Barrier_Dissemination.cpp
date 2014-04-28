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

  struct flags {
    std::array<int,32> myflags[2];
    std::array<int*,32> partnerflags[2];
  };

  struct LocalData {
    int parity;
    bool sense;
    flags* localflags;
  };

  Galois::Runtime::PerThreadStorage<flags> allnodes;
  Galois::Runtime::PerThreadStorage<LocalData> local;
  int LogP;


  void _reinit(unsigned P) {
    LogP = FAST_LOG2_UP(P);
    for (int i = 0; i < P; ++i) {
      auto& rd = *allnodes.getRemote(i);
      local.getRemote(i)->parity = 0;
      local.getRemote(i)->sense = true;
      local.getRemote(i)->localflags = &rd;
      rd.myflags[0].fill(0); rd.myflags[1].fill(0);
      for (int k = 0; k < 32; ++k) {
        int j = (i + (1 << k)) % P;
        for (int r = 0; r < 2; ++r)
          rd.partnerflags[r][k] = &(allnodes.getRemote(i)->myflags[r][k]);
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
    LocalData& ld = *local.getLocal();
    for (int instance = 0; instance < LogP-1; ++instance) {
      *(ld.localflags->partnerflags[ld.parity][instance]) = ld.sense;
      while (ld.localflags->myflags[ld.parity][instance] != ld.sense) { Galois::Runtime::LL::asmPause(); }
    }
    if (ld.parity == 1)
      ld.sense = !ld.sense;
    ld.parity = 1 - ld.parity;
  }

  virtual const char* name() const { return "DiseminationBarrier"; }
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

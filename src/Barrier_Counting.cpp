/** Galois barrier -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * Simple Counting Barrier
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ActiveThreads.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

namespace {

class CountingBarrier: public Galois::Runtime::Barrier {
  std::atomic<unsigned> count;
  std::atomic<bool> sense;
  Galois::Runtime::PerThreadStorage<bool> local_sense;
  unsigned num;

  void _reinit(unsigned val) {
    count = num = val;
    sense = false;
    for (unsigned i = 0; i < local_sense.size(); ++i)
      *local_sense.getRemote(i) = false;
  }

public:
  CountingBarrier() { _reinit(0); }
  CountingBarrier(unsigned int val) { _reinit(val); }

  virtual ~CountingBarrier() {}

  virtual void reinit(unsigned val) { _reinit(val); }

  virtual void wait() {
    bool& lsense = *local_sense.getLocal();
    lsense = !lsense;
    if (--count == 0) {
      count = num;
      sense = lsense;
    } else {
      while (sense != lsense) { Galois::Runtime::LL::asmPause(); }
    }
  }

  virtual const char* name() const { return "CountingBarrier"; }
};

}

Galois::Runtime::Barrier& Galois::Runtime::benchmarking::getCountingBarrier() {
  static CountingBarrier b;
  static unsigned num = ~0;
  if (activeThreads != num) {
    num = activeThreads;
    b.reinit(num);
  }
  return b;
}


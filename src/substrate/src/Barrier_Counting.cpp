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
 * Simple Counting Barrier
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/CompilerSpecific.h"

namespace {

class CountingBarrier: public Galois::Substrate::Barrier {
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
  virtual ~CountingBarrier() {}

  virtual void reinit(unsigned val) { _reinit(val); }

  virtual void wait() {
    bool& lsense = *local_sense.getLocal();
    lsense = !lsense;
    if (--count == 0) {
      count = num;
      sense = lsense;
    } else {
      while (sense != lsense) { Galois::Substrate::asmPause(); }
    }
  }

  virtual const char* name() const { return "CountingBarrier"; }
};

}

Galois::Substrate::Barrier& Galois::Substrate::benchmarking::getCountingBarrier() {
  static CountingBarrier b;
  static unsigned num = ~0;
  if (Runtime::activeThreads != num) {
    num = Runtime::activeThreads;
    b.reinit(num);
  }
  return b;
}


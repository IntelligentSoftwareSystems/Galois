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
 * Simple Counting Barrier
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/CompilerSpecific.h"

namespace {

class CountingBarrier: public galois::substrate::Barrier {
  std::atomic<unsigned> count;
  std::atomic<bool> sense;
  unsigned num;
  std::vector<galois::substrate::CacheLineStorage<bool> > local_sense;

  void _reinit(unsigned val) {
    count = num = val;
    sense = false;
    local_sense.resize(val);
    for (unsigned i = 0; i < val; ++i)
      local_sense.at(i).get() = false;
  }

public:
  CountingBarrier(unsigned int activeT) {
    _reinit(activeT);
  }

  virtual ~CountingBarrier() {}

  virtual void reinit(unsigned val) { _reinit(val); }

  virtual void wait() {
    bool& lsense = local_sense.at(galois::substrate::ThreadPool::getTID()).get();
    lsense = !lsense;
    if (--count == 0) {
      count = num;
      sense = lsense;
    } else {
      while (sense != lsense) { galois::substrate::asmPause(); }
    }
  }

  virtual const char* name() const { return "CountingBarrier"; }
};

}

std::unique_ptr<galois::substrate::Barrier> galois::substrate::createCountingBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new CountingBarrier(activeThreads));
}


/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

namespace {

class CountingBarrier : public galois::substrate::Barrier {
  std::atomic<unsigned> count;
  std::atomic<bool> sense;
  unsigned num;
  std::vector<galois::substrate::CacheLineStorage<bool>> local_sense;

  void _reinit(unsigned val) {
    count = num = val;
    sense       = false;
    local_sense.resize(val);
    for (unsigned i = 0; i < val; ++i)
      local_sense.at(i).get() = false;
  }

public:
  CountingBarrier(unsigned int activeT) { _reinit(activeT); }

  virtual ~CountingBarrier() {}

  virtual void reinit(unsigned val) { _reinit(val); }

  virtual void wait() {
    bool& lsense =
        local_sense.at(galois::substrate::ThreadPool::getTID()).get();
    lsense = !lsense;
    if (--count == 0) {
      count = num;
      sense = lsense;
    } else {
      while (sense != lsense) {
        galois::substrate::asmPause();
      }
    }
  }

  virtual const char* name() const { return "CountingBarrier"; }
};

} // namespace

std::unique_ptr<galois::substrate::Barrier>
galois::substrate::createCountingBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new CountingBarrier(activeThreads));
}

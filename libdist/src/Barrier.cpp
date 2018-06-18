/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

/**
 * @file libdist/src/Barrier.cpp
 *
 * Contains implementation of HostFence and HostBarrier as well as functions
 * that get static singletons of the 2.
 *
 * A fence flushes out and receives all messages in the network while a barrier
 * simply acts as a barrier in the code for all hosts.
 */

#include "galois/substrate/PerThreadStorage.h"
#include "galois/runtime/Substrate.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/runtime/Network.h"
#include "galois/runtime/LWCI.h"

#include <cstdlib>
#include <cstdio>
#include <limits>

#include <iostream>
#include "galois/runtime/BareMPI.h"

namespace {
class HostFence : public galois::substrate::Barrier {
public:
  virtual const char* name() const { return "HostFence"; }

  virtual void reinit(unsigned val) {}

  //! control-flow barrier across distributed hosts
  //! acts as a distributed-memory fence as well (flushes send and receives)
  virtual void wait() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    for (unsigned h = 0; h < net.Num; ++h) {
      if (h == net.ID)
        continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, net.ID + 1); // non-zero message
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush(); // flush all sends

    unsigned received = 1; // self
    while (received < net.Num) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives(); // flush all receives from net.sendMsg() or
                              // net.sendSimple()
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      assert(p->first != net.ID);
      // ignore received data
      ++received;
    }
    ++galois::runtime::evilPhase;
    if (galois::runtime::evilPhase >=
        std::numeric_limits<int16_t>::max()) { // limit defined by MPI or LCI
      galois::runtime::evilPhase = 1;
    }
  }
};

class HostBarrier : public galois::substrate::Barrier {
public:
  virtual const char* name() const { return "HostBarrier"; }

  virtual void reinit(unsigned val) {}

  //! Control-flow barrier across distributed hosts
  virtual void wait() {
#ifdef GALOIS_USE_LWCI
    lc_barrier(mv);
#else
    MPI_Barrier(MPI_COMM_WORLD); // assumes MPI_THREAD_MULTIPLE
#endif
  }
};

} // end anonymous namespace

galois::substrate::Barrier& galois::runtime::getHostBarrier() {
  static HostBarrier b;
  return b;
}

galois::substrate::Barrier& galois::runtime::getHostFence() {
  static HostFence b;
  return b;
}

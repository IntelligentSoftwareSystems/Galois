#include "galois/substrate/PerThreadStorage.h"
#include "galois/runtime/Substrate.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/runtime/Network.h"
//#include "galois/runtime/Directory.h"
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

  virtual void reinit(unsigned val) { }

  // control-flow barrier across distributed hosts
  // acts as a distributed-memory fence as well (flushes send and receives)
  virtual void wait() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    for (unsigned h = 0; h < net.Num; ++h) {
      if (h == net.ID) continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, net.ID+1); // non-zero message
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush(); // flush all sends

    unsigned received = 1; // self
    while (received < net.Num) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives(); // flush all receives from net.sendMsg() or net.sendSimple()
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      assert(p->first != net.ID);
      // ignore received data
      ++received;
    }
    ++galois::runtime::evilPhase;
    if (galois::runtime::evilPhase >= std::numeric_limits<int16_t>::max()) { // limit defined by MPI or LCI
      galois::runtime::evilPhase = 1;
    }
  }
};

class HostBarrier : public galois::substrate::Barrier {

public:
  virtual const char* name() const { return "HostBarrier"; }

  virtual void reinit(unsigned val) { }

  // control-flow barrier across distributed hosts
  virtual void wait() {
#ifdef GALOIS_USE_LWCI
    lc_barrier(mv);
#else
    MPI_Barrier(MPI_COMM_WORLD); // assumes MPI_THREAD_MULTIPLE
#endif
  }
};

} // end namespace ""

galois::substrate::Barrier& galois::runtime::getHostBarrier() {
  static HostBarrier b;
  return b;
}

galois::substrate::Barrier& galois::runtime::getHostFence() {
  static HostFence b;
  return b;
}


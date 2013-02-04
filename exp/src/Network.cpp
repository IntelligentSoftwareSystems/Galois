/** Galois Network Layer Generic Support -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Network.h"

#include <cassert>

namespace Galois {
namespace Runtime {
namespace Distributed {

uint32_t networkHostID = 0;
uint32_t networkHostNum = 1;
}
}
}

static bool ourexit = false;

//!landing pad for worker hosts
static void networkExit(Galois::Runtime::Distributed::RecvBuffer& buf) {
  assert(Galois::Runtime::Distributed::networkHostNum > 1);
  assert(Galois::Runtime::Distributed::networkHostID > 0);
  ourexit = true;
}

void Galois::Runtime::Distributed::networkStart() {
  NetworkInterface& net = getSystemNetworkInterface();
  if (networkHostID != 0) {
    while (!ourexit) {
      net.handleReceives();
    }
    exit(0);
  }
}

void Galois::Runtime::Distributed::networkTerminate() {
  assert(networkHostNum > 1);
  assert(networkHostID == 0);
  NetworkInterface& net = getSystemNetworkInterface();
  SendBuffer buf;
  net.broadcastMessage (&networkExit, buf);
  return;
}

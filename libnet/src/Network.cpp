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

#include "Galois/Runtime/Tracer.h"
//#include "Galois/Runtime/Barrier.h"
//#include "Galois/Runtime/Directory.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/NetworkIO.h"
#include "Galois/Runtime/NetworkBackend.h"

#include <type_traits>
#include <cassert>
#include <iostream>
#include <mutex>

using namespace Galois::Runtime;

uint32_t Galois::Runtime::NetworkInterface::ID = 0;
uint32_t Galois::Runtime::NetworkInterface::Num = 1;

uint32_t Galois::Runtime::getHostID() { return NetworkInterface::ID; }

Galois::Runtime::NetworkIO::~NetworkIO() {}

void NetworkInterface::reportStats() {
  Galois::Runtime::reportStat("NULL", "RecvNum", statRecvNum, 0);
  Galois::Runtime::reportStat("NULL", "RecvBytes", statRecvBytes, 0);
  Galois::Runtime::reportStat("NULL", "SendNum", statSendNum, 0);
  Galois::Runtime::reportStat("NULL", "SendBytes", statSendBytes, 0);
#if 0
  statRecvNum.report();
  statRecvBytes.report();
  statSendNum.report();
  statSendBytes.report();
#endif

}

#if 0
void NetworkInterface::dumpStats() const {
  std::cout << ID
            << ": RN " << statRecvNum.getVal()
            << " RB " <<statRecvBytes.getVal()
            << " SN " <<statSendNum.getVal()
            << " SB " << statSendBytes.getVal() << "\n";
}
#endif
unsigned long NetworkInterface::reportSendBytes() {
  //  return statSendBytes.getVal();
}

//anchor vtable
NetworkInterface::~NetworkInterface() {}

//forward decl
static void bcastLandingPad(::RecvBuffer& buf);

//recieve broadcast message over the network
static void bcastLandingPad(RecvBuffer& buf) {
  unsigned source;
  recvFuncTy recv;
  gDeserialize(buf, source, recv);
  trace("NetworkInterface::bcastLandingPad % %\n", source, (void*)recv);
  recv(buf);
}

void NetworkInterface::broadcast(recvFuncTy recv, SendBuffer& buf, bool self) {
  unsigned source = NetworkInterface::ID;
  trace("NetworkInterface::broadcast % %\n", source, (void*)recv);
  for (unsigned x = 0; x < Num; ++x) {
    if (x != ID) {
      SendBuffer b;
      gSerialize(b, source, recv, buf);
      send(x, &bcastLandingPad, b);
    } else if (self) {
      RecvBuffer rb(buf.begin(), buf.end());
      recv(rb);
    }
  }
}

void NetworkInterface::flush() { }

NetworkBackend::SendBlock* NetworkBackend::allocSendBlock() {
  //FIXME: review for TBAA rules
  std::lock_guard<Substrate::SimpleLock> lg(flLock);
  SendBlock* retval = nullptr;
  if (freelist.empty()) {
    unsigned char* data = (unsigned char*)malloc(sizeof(SendBlock) + size());
    retval = new (data) SendBlock(data + sizeof(SendBlock));
  } else {
    retval = &freelist.front();
    freelist.pop_front();
    retval->size = 0;
    retval->dest = ~0;
  }
  return retval;
}

void NetworkBackend::freeSendBlock(SendBlock* sb) {
  std::lock_guard<Substrate::SimpleLock> lg(flLock);
  freelist.push_front(*sb);
}

NetworkBackend::~NetworkBackend() {
  while (!freelist.empty()) {
    SendBlock* sb = &freelist.front();
    freelist.pop_front();
    sb->~SendBlock();
    free(sb);
  }
}

NetworkBackend::NetworkBackend(unsigned size) :sz(size),_ID(0),_Num(0) {}

NetworkInterface& Galois::Runtime::getSystemNetworkInterface() {
  //return makeNetworkRouted();
  return makeNetworkBuffered();
}

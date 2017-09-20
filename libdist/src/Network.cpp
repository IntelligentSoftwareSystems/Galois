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

using namespace galois::Runtime;

uint32_t galois::runtime::evilPhase = 1;

uint32_t galois::runtime::NetworkInterface::ID = 0;
uint32_t galois::runtime::NetworkInterface::Num = 1;

uint32_t galois::runtime::getHostID() { return NetworkInterface::ID; }

galois::runtime::NetworkIO::~NetworkIO() {}

//anchor vtable
NetworkInterface::~NetworkInterface() {}

//forward decl
static void bcastLandingPad(uint32_t src, ::RecvBuffer& buf);

//receive broadcast message over the network
static void bcastLandingPad(uint32_t src, RecvBuffer& buf) {
  uintptr_t fp;
  gDeserialize(buf, fp);
  auto recv = (void (*)(uint32_t, RecvBuffer&))fp;
  trace("NetworkInterface::bcastLandingPad", (void*)recv);
  recv(src, buf);
}

void NetworkInterface::sendMsg(uint32_t dest, void (*recv)(uint32_t, RecvBuffer&), SendBuffer& buf) {
  gSerialize(buf, recv);
  sendTagged(dest, 0, buf);
}

void NetworkInterface::broadcast(void (*recv)(uint32_t, RecvBuffer&), SendBuffer& buf, bool self) {
  trace("NetworkInterface::broadcast", (void*)recv);
  auto fp = (uintptr_t)recv;
  for (unsigned x = 0; x < Num; ++x) {
    if (x != ID) {
      SendBuffer b;
      gSerialize(b, fp, buf, (uintptr_t)&bcastLandingPad);
      sendTagged(x, 0, b);
    } else if (self) {
      RecvBuffer rb(buf.begin(), buf.end());
      recv(ID, rb);
    }
  }
}

void NetworkInterface::handleReceives() {
  std::unique_lock<substrate::SimpleLock> lg;
  auto opt = recieveTagged(0, &lg);
  while (opt) {
    uint32_t src = std::get<0>(*opt);
    RecvBuffer& buf = std::get<1>(*opt);
    uintptr_t fp  = 0;
    gDeserializeRaw(buf.r_linearData() + buf.r_size() - sizeof(uintptr_t), fp);
    buf.pop_back(sizeof(uintptr_t));
    assert(fp);
    auto f = (void (*)(uint32_t, RecvBuffer&))fp;
    f(src, buf);
    opt = recieveTagged(0, &lg);
  }
}

NetworkBackend::SendBlock* NetworkBackend::allocSendBlock() {
  //FIXME: review for TBAA rules
  std::lock_guard<substrate::SimpleLock> lg(flLock);
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
  std::lock_guard<substrate::SimpleLock> lg(flLock);
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

NetworkInterface& galois::runtime::getSystemNetworkInterface() {
  //return makeNetworkRouted();
  return makeNetworkBuffered();
}

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
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Directory.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/NetworkBackend.h"

#include <cassert>

using namespace Galois::Runtime;

uint32_t NetworkInterface::ID = 0;
uint32_t NetworkInterface::Num = 1;

static bool ourexit = false;
static std::deque<std::pair<recvFuncTy, RecvBuffer>> loopwork;

//!landing pad for worker hosts
static void networkExit() {
  assert(NetworkInterface::Num > 1);
  assert(NetworkInterface::ID > 0);
  ourexit = true;
}

static void loop_pad(::RecvBuffer& b) {
  uintptr_t f;
  gDeserialize(b, f);
  loopwork.push_back(std::make_pair((recvFuncTy)f, b));
}

void NetworkInterface::start() {
  getSystemBarrier(); // initialize barrier before anyone might be at it
  getSystemNetworkInterface();
  getSystemDirectory();
  if (NetworkInterface::ID != 0) {
    while (!ourexit) {
      doNetworkWork();
      if (!loopwork.empty()) {
        auto p = loopwork.front();
        loopwork.pop_front();
        p.first(p.second);
      }
    }
    exit(0);
  }
}

void NetworkInterface::terminate() {
  //return if just one host is running
  if (NetworkInterface::Num == 1)
    return;
  assert(NetworkInterface::ID == 0);
  NetworkInterface& net = getSystemNetworkInterface();
  net.broadcastAlt(&networkExit);
  doNetworkWork();
  return;
}

void NetworkInterface::sendLoop(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
  buf.serialize_header((void*)recv);
  getSystemNetworkInterface().send(dest, &loop_pad, buf);
}

//anchor vtable
NetworkInterface::~NetworkInterface() {}

//RealID -> effective ID for the broadcast tree
static unsigned getEID(unsigned realID, unsigned srcID) {
  return (realID + NetworkInterface::Num - srcID) % NetworkInterface::Num;
}

//Effective id in the broadcast tree -> realID
static unsigned getRID(unsigned eID, unsigned srcID) {
  return (eID + srcID) % NetworkInterface::Num;
}

//forward decl
static void bcastLandingPad(::RecvBuffer& buf);

//forward message along tree
static void bcastForward(NetworkInterface& net, unsigned source, ::RecvBuffer& buf) {
  static const int width = 2;

  unsigned eid = getEID(NetworkInterface::ID, source);
  
  for (int i = 0; i < width; ++i) {
    unsigned ndst = eid * width + i + 1;
    if (ndst < NetworkInterface::Num) {
      SendBuffer sbuf;
      gSerialize(sbuf, source, buf);
      net.send(getRID(ndst, source), &bcastLandingPad, sbuf);
    }
  }
}

//recieve broadcast message over the network
static void bcastLandingPad(RecvBuffer& buf) {
  unsigned source;
  gDeserialize(buf, source);
  trace_bcast_recv(source);
  bcastForward(getSystemNetworkInterface(), source, buf);
  //deliver locally
  recvFuncTy recv;
  gDeserialize(buf, recv);
  recv(buf);
}

void NetworkInterface::broadcast(recvFuncTy recv, SendBuffer& buf, bool self) {
  unsigned source = NetworkInterface::ID;
  trace_bcast_recv(source);
  buf.serialize_header((void*)recv);
  RecvBuffer rbuf(std::move(buf));
  bcastForward(*this, source, rbuf);
  if (self)
    recv(rbuf);
}

void NetworkInterface::flush() {
}

NetworkBackend::SendBlock* NetworkBackend::allocSendBlock() {
  //FIXME: review for TBAA rules
  unsigned char* data = (unsigned char*)malloc(sizeof(SendBlock) + size());
  SendBlock* retval = (SendBlock*)data;
  retval->size = 0;
  retval->dest = ~0;
  retval->next = nullptr;
  retval->data = data + sizeof(SendBlock);
  return retval;
}

void NetworkBackend::freeSendBlock(SendBlock* sb) {
  free(sb);
}

NetworkBackend::~NetworkBackend() {}

NetworkBackend::NetworkBackend(unsigned size) :sz(size) {}

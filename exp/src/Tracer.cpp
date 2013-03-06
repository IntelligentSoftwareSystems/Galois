/** Galois Distributed Object Tracer -*- C++ -*-
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
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/ll/gio.h"

using Galois::Runtime::LL::gDebug;
using Galois::Runtime::Distributed::getSystemNetworkInterface;
using Galois::Runtime::Distributed::RecvBuffer;
using Galois::Runtime::Distributed::gSerialize;
using Galois::Runtime::Distributed::gDeserialize;
using Galois::Runtime::Distributed::networkHostID;

namespace {

int count;

void trace_obj_send_do(uint32_t src, uint32_t owner, void* ptr, uint32_t remote) {
  int v = __sync_add_and_fetch(&count, 1);
  gDebug("SEND ", src, " -> ", remote, " [", owner, ",", ptr, "] (", v, ")");
}

void trace_obj_recv_do(uint32_t src, uint32_t owner, void* ptr) {
  int v = __sync_sub_and_fetch(&count, 1);
  gDebug("RECV * -> ", src, " [", owner, ",", ptr, "] (", v, ")");
}

void trace_obj_recv_pad(RecvBuffer &buf) {
  uint32_t src;
  uint32_t owner;
  void* ptr;
  gDeserialize(buf, src, owner, ptr);
  trace_obj_recv_do(src, owner, ptr);
}

void trace_obj_send_pad(RecvBuffer &buf) {
  uint32_t src;
  uint32_t owner;
  void* ptr;
  uint32_t remote;
  gDeserialize(buf, src, owner, ptr, remote);
  trace_obj_send_do(src, owner, ptr, remote);
}

}

void Galois::Runtime::Distributed::trace_obj_send(uint32_t owner, void* ptr, uint32_t remote) {
#ifdef NDEBUG
  return;
#endif

  if (networkHostID == 0) {
    trace_obj_send_do(networkHostID, owner, ptr, remote);
  } else {
    SendBuffer sbuf;
    gSerialize(sbuf, networkHostID, owner, ptr, remote);
    getSystemNetworkInterface().sendMessage(0, &trace_obj_send_pad, sbuf);
  }
}

void Galois::Runtime::Distributed::trace_obj_recv(uint32_t owner, void* ptr) {
#ifdef NDEBUG
  return;
#endif
  if (networkHostID == 0) {
    trace_obj_recv_do(networkHostID, owner, ptr);
  } else {
    SendBuffer sbuf;
    gSerialize(sbuf, networkHostID, owner, ptr);
    getSystemNetworkInterface().sendMessage(0, &trace_obj_recv_pad, sbuf);
  }
}


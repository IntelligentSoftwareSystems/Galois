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
#include "Galois/Runtime/ll/EnvCheck.h"

using Galois::Runtime::LL::gDebug;
using Galois::Runtime::getSystemNetworkInterface;
using Galois::Runtime::RecvBuffer;
using Galois::Runtime::gSerialize;
using Galois::Runtime::gDeserialize;
using Galois::Runtime::networkHostID;

namespace {

static int count;

void trace_obj_send_do(uint32_t src, uint32_t owner, void* ptr, uint32_t remote) {
  int v = __sync_add_and_fetch(&count, 1);
  gDebug("SEND ", src, " -> ", remote, " [", owner, ",", ptr, "] (", v, ")");
}

void trace_obj_recv_do(uint32_t src, uint32_t owner, void* ptr) {
  int v = __sync_sub_and_fetch(&count, 1);
  gDebug("RECV * -> ", src, " [", owner, ",", ptr, "] (", v, ")");
}

void trace_bcast_recv_do(uint32_t host, uint32_t source) {
  gDebug("BCast at ", host, " from ", source);
}

}

static const bool traceLocal = Galois::Runtime::LL::EnvCheck("GALOIS_TRACE_LOCAL");

void Galois::Runtime::trace_obj_send_impl(uint32_t owner, void* ptr, uint32_t remote) {
  if (networkHostID == 0 || traceLocal) {
    trace_obj_send_do(networkHostID, owner, ptr, remote);
  } else {
    getSystemNetworkInterface().sendAlt(0, &trace_obj_send_do, networkHostID, owner, ptr, remote);
  }
}

void Galois::Runtime::trace_obj_recv_impl(uint32_t owner, void* ptr) {
  if (networkHostID == 0 || traceLocal) {
    trace_obj_recv_do(networkHostID, owner, ptr);
  } else {
    getSystemNetworkInterface().sendAlt(0, &trace_obj_recv_do, networkHostID, owner, ptr);
  }
}

void Galois::Runtime::trace_bcast_recv_impl(uint32_t source) {
  if (networkHostID == 0 || traceLocal) {
    trace_bcast_recv_do(networkHostID, source);
  } else {
    getSystemNetworkInterface().sendAlt(0, &trace_bcast_recv_do, networkHostID, source);
  }
}

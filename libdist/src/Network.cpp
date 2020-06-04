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

/**
 * @file Network.cpp
 *
 * Contains implementations for basic NetworkInterface functions and
 * initializations of some NetworkInterface variables.
 */

#include "galois/runtime/Tracer.h"
#include "galois/runtime/Network.h"
#include "galois/runtime/NetworkIO.h"

#include <iostream>
#include <mutex>

using namespace galois::runtime;

uint32_t galois::runtime::evilPhase = 1;

uint32_t galois::runtime::NetworkInterface::ID  = 0;
uint32_t galois::runtime::NetworkInterface::Num = 1;

uint32_t galois::runtime::getHostID() { return NetworkInterface::ID; }

galois::runtime::NetworkIO::~NetworkIO() {}

void NetworkInterface::initializeMPI() {
  int supportProvided;
  int initSuccess =
      MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &supportProvided);
  if (initSuccess != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, initSuccess);
  }

  if (supportProvided != MPI_THREAD_MULTIPLE) {
    GALOIS_DIE("MPI_THREAD_MULTIPLE not supported.");
  }
}

void NetworkInterface::finalizeMPI() {
  int finalizeSuccess = MPI_Finalize();

  if (finalizeSuccess != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, finalizeSuccess);
  }

  galois::gDebug("[", NetworkInterface::ID, "] MPI finalized");
}

NetworkInterface::NetworkInterface() {}

NetworkInterface::~NetworkInterface() {}

void NetworkInterface::reportMemUsage() const {
  std::string str("CommunicationMemUsage");
  galois::runtime::reportStat_Tmin("dGraph", str + "Min",
                                   memUsageTracker.getMaxMemUsage());
  galois::runtime::reportStat_Tmax("dGraph", str + "Max",
                                   memUsageTracker.getMaxMemUsage());
}

// forward decl
//! Receive broadcasted messages over the network
static void bcastLandingPad(uint32_t src, ::RecvBuffer& buf);

static void bcastLandingPad(uint32_t src, RecvBuffer& buf) {
  uintptr_t fp;
  gDeserialize(buf, fp);
  auto recv = (void (*)(uint32_t, RecvBuffer&))fp;
  trace("NetworkInterface::bcastLandingPad", (void*)recv);
  recv(src, buf);
}

void NetworkInterface::sendMsg(uint32_t dest,
                               void (*recv)(uint32_t, RecvBuffer&),
                               SendBuffer& buf) {
  gSerialize(buf, recv);
  sendTagged(dest, 0, buf);
}

void NetworkInterface::broadcast(void (*recv)(uint32_t, RecvBuffer&),
                                 SendBuffer& buf, bool self) {
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
    uint32_t src    = std::get<0>(*opt);
    RecvBuffer& buf = std::get<1>(*opt);
    uintptr_t fp    = 0;
    gDeserializeRaw(buf.r_linearData() + buf.r_size() - sizeof(uintptr_t), fp);
    buf.pop_back(sizeof(uintptr_t));
    assert(fp);
    auto f = (void (*)(uint32_t, RecvBuffer&))fp;
    f(src, buf);
    opt = recieveTagged(0, &lg);
  }
}

NetworkInterface& galois::runtime::getSystemNetworkInterface() {
#ifndef GALOIS_USE_LCI
  return makeNetworkBuffered();
#else
  return makeNetworkLCI();
#endif
}

void galois::runtime::internal::destroySystemNetworkInterface() {
  // get net interface, then delete it
  NetworkInterface& netInterface = getSystemNetworkInterface();
  delete &netInterface;
}

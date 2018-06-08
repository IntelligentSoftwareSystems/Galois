/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#include "galois/runtime/ll/HWTopo.h"

#include <vector>

#include <unistd.h>
#include <stdio.h>
#include <thread.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>

using namespace galois::runtime::LL;

namespace {

static bool sunBindToProcessor(int proc) {
  if (processor_bind(P_LWPID,  thr_self(), proc, 0) == -1) {
    gWarn("Could not set CPU affinity for thread ", proc, "(", strerror(errno), ")");
    return false;
  }
  return true;
}

//Flat machine with the correct number of threads and binding
struct Policy {
  std::vector<int> procmap; //Galois id -> solaris id

  unsigned numThreads, numCores, numSockets;

  Policy() {
    processorid_t i, cpuid_max;
    cpuid_max = sysconf(_SC_CPUID_MAX);
    for (i = 0; i <= cpuid_max; i++) {
      if (p_online(i, P_STATUS) != -1) {
	procmap.push_back(i);
	//printf("processor %d present\n", i);
      }
    }

    numThreads = procmap.size();
    numCores = procmap.size();
    numSockets = 1;
  }
};

static Policy& getPolicy() {
  static Policy A;
  return A;
}

} //namespace

bool galois::runtime::LL::bindThreadToProcessor(int id) {
  assert(size_t(id) < procmap.size ());
  return sunBindToProcessor(getPolicy().procmap[id]);
}

unsigned galois::runtime::LL::getProcessorForThread(int id) {
  assert(size_t(id) < procmap.size ());
  return getPolicy().procmap[id];
}

unsigned galois::runtime::LL::getMaxThreads() {
  return getPolicy().numThreads;
}

unsigned galois::runtime::LL::getMaxCores() {
  return getPolicy().numCores;
}

unsigned galois::runtime::LL::getMaxSockets() {
  return getPolicy().numSockets;
}

unsigned galois::runtime::LL::getMaxSocketForThread(int id) {
  return getPolicy().numSockets - 1;
}

unsigned galois::runtime::LL::getSocketForThread(int id) {
  return 0;
}

bool galois::runtime::LL::isSocketLeader(int id) {
  return id == 0;
}

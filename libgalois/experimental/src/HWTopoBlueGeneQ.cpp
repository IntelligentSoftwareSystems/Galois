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

#include "galois/substrate/HWTopo.h"
#include "galois/gIO.h"

#include <vector>
#include <sched.h>

using namespace galois::substrate;

namespace {

static bool bindToProcessor(int proc) {
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO(&mask);
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET(proc, &mask);
  
  /* sched_setaffinity returns 0 in success */
  if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    gWarn("Could not set CPU affinity for thread ", proc, "(", strerror(errno), ")");
    return false;
  }
  return true;
}

//! Flat machine with the correct number of threads and binding
struct Policy {
  std::vector<int> procmap; //Galois id -> cpu id

  unsigned numThreads, numCores, numSockets;

  Policy() {
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 4; ++j) {
        procmap.push_back(j*16 + i);
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
  return bindToProcessor(getPolicy().procmap[id]);
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

unsigned galois::runtime::LL::getLeaderForThread(int id) {
  return 0;
}

unsigned galois::runtime::LL::getLeaderForSocket(int id) {
  return 0;
}

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

#ifndef GALOIS_SUBSTRATE_HWTOPO_H
#define GALOIS_SUBSTRATE_HWTOPO_H

#include <string>
#include <vector>

#include "galois/config.h"

namespace galois::substrate {

struct ThreadTopoInfo {
  unsigned tid;                 // this thread (galois id)
  unsigned socketLeader;        // first thread id in tid's socket
  unsigned socket;              // socket (L3 normally) of thread
  unsigned numaNode;            // memory bank.  may be different than socket.
  unsigned cumulativeMaxSocket; // max socket id seen from [0, tid]
  unsigned osContext;           // OS ID to use for thread binding
  unsigned osNumaNode;          // OS ID for numa node
};

struct MachineTopoInfo {
  unsigned maxThreads;
  unsigned maxCores;
  unsigned maxSockets;
  unsigned maxNumaNodes;
};

struct HWTopoInfo {
  MachineTopoInfo machineTopoInfo;
  std::vector<ThreadTopoInfo> threadTopoInfo;
};

/**
 * getHWTopo determines the machine topology from the process information
 * exposed in /proc and /dev filesystems.
 */
HWTopoInfo getHWTopo();

/**
 * parseCPUList parses cpuset information in "List format" as described in
 * cpuset(7) and available under /proc/self/status
 */
std::vector<int> parseCPUList(const std::string& in);

/**
 * bindThreadSelf binds a thread to an osContext as returned by getHWTopo.
 */
bool bindThreadSelf(unsigned osContext);

} // namespace galois::substrate

#endif

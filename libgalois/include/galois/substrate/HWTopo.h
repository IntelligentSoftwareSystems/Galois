/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include <vector>

namespace galois {
namespace substrate {

struct threadTopoInfo {
  unsigned tid;                 // this thread (galois id)
  unsigned socketLeader;        // first thread id in tid's socket
  unsigned socket;              // socket (L3 normally) of thread
  unsigned numaNode;            // memory bank.  may be different than socket.
  unsigned cumulativeMaxSocket; // max socket id seen from [0, tid]
  unsigned osContext;           // OS ID to use for thread binding
  unsigned osNumaNode;          // OS ID for numa node
};

struct machineTopoInfo {
  unsigned maxThreads;
  unsigned maxCores;
  unsigned maxSockets;
  unsigned maxNumaNodes;
};

// parse machine topology
std::pair<machineTopoInfo, std::vector<threadTopoInfo>> getHWTopo();
// bind a thread to a hwContext (returned by getHWTopo)
bool bindThreadSelf(unsigned osContext);

} // end namespace substrate
} // end namespace galois

#endif //_HWTOPO_H

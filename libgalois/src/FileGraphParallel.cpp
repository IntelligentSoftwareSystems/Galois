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

#include "galois/graphs/FileGraph.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/HWTopo.h"

#include <mutex>
#include <condition_variable>

namespace galois {
namespace graphs {

void FileGraph::fromFileInterleaved(const std::string& filename,
                                    size_t sizeofEdgeData) {
  fromFile(filename);

  std::mutex lock;
  std::condition_variable cond;
  auto& tp            = substrate::getThreadPool();
  unsigned maxSockets = tp.getMaxSockets();
  unsigned count      = maxSockets;

  // Interleave across all NUMA nodes
  tp.run(tp.getMaxThreads(), [&]() {
    std::unique_lock<std::mutex> lk(lock);
    if (substrate::ThreadPool::isLeader()) {
      pageInByNode(substrate::ThreadPool::getSocket(), maxSockets,
                   sizeofEdgeData);
      if (--count == 0)
        cond.notify_all();
    } else {
      cond.wait(lk, [&]() { return count == 0; });
    }
  });
}

} // namespace graphs
} // namespace galois

/** Parallel implementations for FileGraph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Graph/FileGraph.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Runtime/ll/TID.h"

#include <mutex>
#include <condition_variable>

namespace Galois {
namespace Graph {

void FileGraph::fromFileInterleaved(const std::string& filename, size_t sizeofEdgeData) {
  fromFile(filename);

  std::mutex lock;
  std::condition_variable cond;
  unsigned maxPackages = Runtime::LL::getMaxPackages();
  unsigned count = maxPackages;

  // Interleave across all NUMA nodes
  Galois::Runtime::getSystemThreadPool().run(std::numeric_limits<unsigned int>::max(), [&]() {
    unsigned tid = Galois::Runtime::LL::getTID();
    std::unique_lock<std::mutex> lk(lock);
    if (Galois::Runtime::LL::isPackageLeaderForSelf(tid)) {
      pageInByNode(Galois::Runtime::LL::getPackageForThread(tid), maxPackages, sizeofEdgeData);
      if (--count == 0)
        cond.notify_all();
    } else {
      cond.wait(lk, [&](){ return count == 0; });
    }
  });
}

}
}

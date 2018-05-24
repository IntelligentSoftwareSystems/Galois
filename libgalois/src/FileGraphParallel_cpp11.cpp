/** Parallel implementations for FileGraph -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "galois/graphs/FileGraph.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/HWTopo.h"

#include <mutex>
#include <condition_variable>

namespace galois {
namespace graphs {

void FileGraph::fromFileInterleaved(const std::string& filename, size_t sizeofEdgeData) {
  fromFile(filename);

  std::mutex lock;
  std::condition_variable cond;
  auto& tp = substrate::getThreadPool();
  unsigned maxPackages = tp.getMaxPackages();
  unsigned count = maxPackages;

  // Interleave across all NUMA nodes
  tp.run(tp.getMaxThreads(), [&]() {
      std::unique_lock<std::mutex> lk(lock);
      if (substrate::ThreadPool::isLeader()) {
        pageInByNode(substrate::ThreadPool::getPackage(), maxPackages, sizeofEdgeData);
        if (--count == 0)
          cond.notify_all();
      } else {
        cond.wait(lk, [&](){ return count == 0; });
      }
    });
}

}
}

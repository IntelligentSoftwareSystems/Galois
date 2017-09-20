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
#include "galois/Graph/FileGraph.h"
#include "galois/Runtime/ThreadPool.h"
#include "galois/Runtime/ll/HWTopo.h"
#include "galois/Runtime/ll/TID.h"

#include <pthread.h>

namespace galois {
namespace graphs {

void FileGraph::fromFileInterleaved(const std::string& filename, size_t sizeofEdgeData) {
  fromFile(filename);

  pthread_mutex_t lock;
  pthread_cond_t cond;
  
  if (pthread_mutex_init(&lock, NULL))
    GALOIS_DIE("PTHREAD");
  if (pthread_cond_init(&cond, NULL))
    GALOIS_DIE("PTHREAD");

  unsigned maxPackages = runtime::LL::getMaxPackages();
  unsigned count = maxPackages;

  // Interleave across all NUMA nodes
  // FileGraphAllocator fn { lock, cond, this, sizeofEdgeData, maxPackages, count };
  galois::runtime::getThreadPool().run(std::numeric_limits<unsigned int>::max(), [&]() {
    unsigned tid = galois::runtime::LL::getTID();
    if (pthread_mutex_lock(&lock))
      GALOIS_DIE("PTHREAD");

    if (galois::runtime::LL::isPackageLeaderForSelf(tid)) {
      pageInByNode(galois::runtime::LL::getPackageForThread(tid), maxPackages, sizeofEdgeData);
      if (--count == 0) {
        if (pthread_cond_broadcast(&cond))
          GALOIS_DIE("PTHREAD");
      }
    } else {
      while (count != 0) {
        pthread_cond_wait(&cond, &lock);
      }
    }
    if (pthread_mutex_unlock(&lock))
      GALOIS_DIE("PTHREAD");
  });

  if (pthread_mutex_destroy(&lock))
    GALOIS_DIE("PTHREAD");
  if (pthread_cond_destroy(&cond))
    GALOIS_DIE("PTHREAD");
}

}
}

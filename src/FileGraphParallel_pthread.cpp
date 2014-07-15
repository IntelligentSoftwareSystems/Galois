/** Parallel implementations for FileGraph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Graph/FileGraph.h"

#include <pthread.h>

namespace Galois {
namespace Graph {

void FileGraph::fromFileInterleaved(const std::string& filename, size_t sizeofEdgeData) {
  fromFile(filename, false);

  pthread_mutex_t lock;
  pthread_cond_t cond;
  
  if (pthread_mutex_init(&lock, NULL))
    GALOIS_DIE("PTHREAD");
  if (pthread_cond_init(&cond, NULL))
    GALOIS_DIE("PTHREAD");

  unsigned maxPackages = Runtime::LL::getMaxPackages();
  unsigned count = maxPackages;

  // Interleave across all NUMA nodes
  // FileGraphAllocator fn { lock, cond, this, sizeofEdgeData, maxPackages, count };
  Galois::Runtime::getSystemThreadPool().run(std::numeric_limits<unsigned int>::max(), [&]() {
    unsigned tid = Galois::Runtime::LL::getTID();
    if (pthread_mutex_lock(&lock))
      GALOIS_DIE("PTHREAD");

    if (Galois::Runtime::LL::isPackageLeaderForSelf(tid)) {
      pageIn(Galois::Runtime::LL::getPackageForThread(tid), maxPackages, sizeofEdgeData);
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

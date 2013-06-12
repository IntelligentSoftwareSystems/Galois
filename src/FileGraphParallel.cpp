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
#include "Galois/Galois.h"
#include "Galois/Graph/FileGraph.h"

#include <pthread.h>

namespace Galois {
namespace Graph {

struct FileGraphAllocator {
  pthread_mutex_t& lock;
  pthread_cond_t& cond;
  FileGraph* self;
  size_t sizeofEdgeData;
  unsigned maxPackages;
  volatile unsigned& count;

  FileGraphAllocator(pthread_mutex_t& l, pthread_cond_t& c, FileGraph* s, size_t ss, unsigned m, volatile unsigned& cc): 
    lock(l), cond(c), self(s), sizeofEdgeData(ss), maxPackages(m), count(cc) { }

  void operator()(unsigned tid, unsigned total) {
    int pret_t;
    if ((pret_t = pthread_mutex_lock(&lock)))
      GALOIS_DIE("pthread error: ", pret_t);

    if (Galois::Runtime::LL::isPackageLeaderForSelf(tid)) {
      auto r = self->divideBy(
        sizeof(uint64_t),
        sizeofEdgeData + sizeof(uint32_t),
        Galois::Runtime::LL::getPackageForThread(tid), maxPackages);
      
      size_t edge_begin = *self->edge_begin(*r.first);
      size_t edge_end = edge_begin;
      if (r.first != r.second)
        edge_end = *self->edge_end(*r.second - 1);
      Galois::Runtime::MM::pageIn(self->outIdx + *r.first, std::distance(r.first, r.second) * sizeof(*self->outIdx));
      Galois::Runtime::MM::pageIn(self->outs + edge_begin, (edge_end - edge_begin) * sizeof(*self->outs));
      Galois::Runtime::MM::pageIn(self->edgeData + edge_begin * sizeofEdgeData, (edge_end - edge_begin) * sizeofEdgeData);
      if (--count == 0) {
        if ((pret_t = pthread_cond_broadcast(&cond)))
          GALOIS_DIE("pthread error: ", pret_t);
      }
    } else {
      while (count > 0) {
        if ((pret_t = pthread_cond_wait(&cond, &lock)))
          GALOIS_DIE("pthread error: ", pret_t);
      }
    }

    if ((pret_t = pthread_mutex_unlock(&lock)))
      GALOIS_DIE("pthread error: ", pret_t);
  }
};

void FileGraph::structureFromFileInterleaved(const std::string& filename, size_t sizeofEdgeData) {
  structureFromFile(filename, false);

  // Interleave across all NUMA nodes
  unsigned oldActive = getActiveThreads();
  setActiveThreads(std::numeric_limits<unsigned int>::max());

  // Manually coarsen synchronization granularity, otherwise it would be at page granularity
  int pret;
  unsigned maxPackages = Runtime::LL::getMaxPackages();
  volatile unsigned count = maxPackages;
  pthread_mutex_t lock;
  if ((pret = pthread_mutex_init(&lock, NULL)))
    GALOIS_DIE("pthread error: ", pret);
  pthread_cond_t cond;
  if ((pret = pthread_cond_init(&cond, NULL)))
    GALOIS_DIE("pthread error: ", pret);

  // NB(ddn): Use on_each_simple_impl because we are fiddling with the
  // number of active threads after this loop. Otherwise, the main
  // thread might change the number of active threads while some threads
  // are still in on_each_impl.
  Galois::Runtime::on_each_simple_impl(FileGraphAllocator(lock, cond, this, sizeofEdgeData, maxPackages, count));

  if ((pret = pthread_mutex_destroy(&lock)))
    GALOIS_DIE("pthread error: ", pret);
  if ((pret = pthread_cond_destroy(&cond)))
    GALOIS_DIE("pthread error: ", pret);

  setActiveThreads(oldActive);
}

}
}

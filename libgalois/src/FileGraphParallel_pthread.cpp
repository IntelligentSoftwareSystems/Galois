#include "galois/Graph/FileGraph.h"
#include "galois/runtime/ThreadPool.h"
#include "galois/runtime/ll/HWTopo.h"
#include "galois/runtime/ll/TID.h"

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

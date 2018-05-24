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

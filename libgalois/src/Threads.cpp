#include "galois/substrate/ThreadPool.h"
#include "galois/Threads.h"

#include <algorithm>
namespace galois {
namespace runtime {
unsigned int activeThreads = 1;
}
}

unsigned int galois::setActiveThreads(unsigned int num) noexcept {
  num = std::min(num, galois::substrate::getThreadPool().getMaxUsableThreads());
  num = std::max(num, 1U);
  galois::runtime::activeThreads = num;
  return num;
}

unsigned int galois::getActiveThreads() noexcept {
  return galois::runtime::activeThreads;
}

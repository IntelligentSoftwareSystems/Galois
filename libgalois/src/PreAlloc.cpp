#include "galois/runtime/Executor_OnEach.h"
#include "galois/runtime/Mem.h"
#include "galois/runtime/PagePool.h"

void galois::runtime::preAlloc_impl(unsigned num) {
  unsigned pagesPerThread = (num + activeThreads - 1) / activeThreads;
  substrate::getThreadPool().run(activeThreads, std::bind(pagePoolPreAlloc, pagesPerThread));
}

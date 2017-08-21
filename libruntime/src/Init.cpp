#include "Galois/Runtime/Init.h"
#include "Galois/Runtime/PagePool.h"
#include "Galois/Runtime/StatCollector.h"

void Galois::Runtime::init(void) {
  using namespace Galois::Runtime;
  internal::initPagePool();
  internal::initStatManager();
}


void Galois::Runtime::finish(void) {
  using namespace Galois::Runtime;
  internal::deleteStatManager();
  internal::destroyPagePool();
}


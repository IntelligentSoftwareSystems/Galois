#include "Galois/Runtime/Init.h"

void Galois::Runtime::init(Galois::Runtime::StatCollector* sc) {
  using namespace Galois::Runtime;
  internal::initPagePool();
  internal::setStatCollector(sc);
}


void Galois::Runtime::kill(void) {
  using namespace Galois::Runtime;
  internal::killPagePool();
  internal::setStatCollector(nullptr);
}


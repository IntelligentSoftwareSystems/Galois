#include "Galois/Runtime/Init.h"

void Galois::Runtime::init(Galois::Runtime::StatManager* sm) {
  using namespace Galois::Runtime;
  internal::initPagePool();
  internal::setSysStatManager(sm);
}


void Galois::Runtime::kill(void) {
  using namespace Galois::Runtime;
  internal::killPagePool();
  internal::setSysStatManager(nullptr);
}


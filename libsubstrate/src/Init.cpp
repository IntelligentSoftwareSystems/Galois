#include "Galois/gIO.h"
#include "Galois/Substrate/Init.h"
#include "Galois/Substrate/BarrierImpl.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/Termination.h"


#include <memory>



void Galois::Substrate::init(void) {
  internal::initThreadPool();
  internal::initBarrier();
  internal::initTermDetect();
}

void Galois::Substrate::kill(void) {
  internal::killTermDetect();
  internal::killBarrier();
  internal::killThreadPool();

}

static Galois::Substrate::internal::BarrierInstance* bPtr = nullptr;

void Galois::Substrate::internal::initBarrier(void) {
  GALOIS_ASSERT(!bPtr, "Double initialization of BarrierInstance");
  bPtr = new BarrierInstance();
}

void Galois::Substrate::internal::killBarrier(void) {
  delete bPtr;
  bPtr = nullptr;
}

Galois::Substrate::Barrier& Galois::Substrate::getBarrier(unsigned numT) {
  GALOIS_ASSERT(bPtr, "BarrierInstance not initialized");
  return bPtr->get(numT);
}

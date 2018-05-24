#include "galois/substrate/Barrier.h"

//anchor vtable
galois::substrate::Barrier::~Barrier() {}

//galois::substrate::Barrier& galois::substrate::getSystemBarrier(unsigned activeThreads) {
//  return benchmarking::getTopoBarrier(activeThreads);
//}

static galois::substrate::internal::BarrierInstance<>* BI = nullptr;

void galois::substrate::internal::setBarrierInstance(internal::BarrierInstance<>* bi) {
  GALOIS_ASSERT(!(bi && BI), "Double initialization of BarrierInstance");
  BI = bi;
}

galois::substrate::Barrier& galois::substrate::getBarrier(unsigned numT) {
  GALOIS_ASSERT(BI, "BarrierInstance not initialized");
  return BI->get(numT);
}


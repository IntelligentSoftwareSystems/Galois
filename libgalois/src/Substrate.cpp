#include "galois/runtime/Substrate.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/Init.h"

galois::substrate::Barrier& galois::runtime::getBarrier(unsigned activeThreads) {
  return galois::substrate::getBarrier(activeThreads);
}

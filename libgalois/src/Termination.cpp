#include "galois/gIO.h"
#include "galois/substrate/Termination.h"

// vtable anchoring
galois::substrate::TerminationDetection::~TerminationDetection(void) {}

static galois::substrate::TerminationDetection* TERM = nullptr;

void galois::substrate::internal::setTermDetect(galois::substrate::TerminationDetection* t) {
  GALOIS_ASSERT(!(TERM && t), "Double initialization of TerminationDetection");
  TERM = t;
}


galois::substrate::TerminationDetection& galois::substrate::getSystemTermination(unsigned activeThreads) {
  TERM->init(activeThreads);
  return *TERM;
}

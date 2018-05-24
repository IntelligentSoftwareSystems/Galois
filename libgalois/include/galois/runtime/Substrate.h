#ifndef GALOIS_RUNTIME_SUBSTRATE_H
#define GALOIS_RUNTIME_SUBSTRATE_H

#include "galois/substrate/Barrier.h"

namespace galois {
namespace runtime {

/**
 * Have a pre-instantiated barrier available for use.
 * This is initialized to the current activeThreads. This barrier
 * is designed to be fast and should be used in the common
 * case.
 *
 * However, there is a race if the number of active threads
 * is modified after using this barrier: some threads may still
 * be in the barrier while the main thread reinitializes this
 * barrier to the new number of active threads. If that may
 * happen, use {@link createSimpleBarrier()} instead.
 */
substrate::Barrier& getBarrier(unsigned activeThreads);


} // end namespace runtime
} // end namespace galois

#endif

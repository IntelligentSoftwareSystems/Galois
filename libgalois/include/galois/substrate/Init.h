#ifndef GALOIS_SUBSTRATE_INIT_H
#define GALOIS_SUBSTRATE_INIT_H

#include "galois/gIO.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/Termination.h"

namespace galois {
namespace substrate {

class SharedMemSubstrate {

  // Order is critical here
  ThreadPool m_tpool;

  internal::LocalTerminationDetection<>*  m_termPtr;
  internal::BarrierInstance<>*  m_biPtr;

public:

  /**
   * Initializes the Substrate library components
   */
  SharedMemSubstrate();

  /**
   * Destroys the Substrate library components
   */
  ~SharedMemSubstrate();

};


}
}

#endif // GALOIS_SUBSTRATE_INIT_H

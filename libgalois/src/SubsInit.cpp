#include "Galois/gIO.h"
#include "Galois/Substrate/Init.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/Termination.h"


#include <memory>

using namespace Galois::Substrate;

SharedMemSubstrate::SharedMemSubstrate(void) {
  internal::setThreadPool(&m_tpool);
  internal::setBarrierInstance(&m_barrier);
  internal::setTermDetect(&m_term);
}

SharedMemSubstrate::~SharedMemSubstrate(void) {
  internal::setTermDetect(nullptr);
  internal::setBarrierInstance(nullptr);
  internal::setThreadPool(nullptr);
}


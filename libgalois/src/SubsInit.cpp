#include "Galois/gIO.h"
#include "Galois/Substrate/Init.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/Termination.h"


#include <memory>

using namespace galois::Substrate;

SharedMemSubstrate::SharedMemSubstrate(void) {
  internal::setThreadPool(&m_tpool);

  // delayed initialization because both call getThreadPool in constructor
  // which is valid only after setThreadPool() above
  m_biPtr = new internal::BarrierInstance<>();
  m_termPtr = new internal::LocalTerminationDetection<>();

  GALOIS_ASSERT(m_biPtr);
  GALOIS_ASSERT(m_termPtr);

  internal::setBarrierInstance(m_biPtr);
  internal::setTermDetect(m_termPtr);
}

SharedMemSubstrate::~SharedMemSubstrate(void) {

  internal::setTermDetect(nullptr);
  internal::setBarrierInstance(nullptr);

  // destructors call getThreadPool(), hence must be destroyed before setThreadPool() below
  delete m_termPtr;
  m_termPtr = nullptr;

  delete m_biPtr;
  m_biPtr = nullptr;

  internal::setThreadPool(nullptr);
}


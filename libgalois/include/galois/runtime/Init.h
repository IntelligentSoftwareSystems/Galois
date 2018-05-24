#ifndef GALOIS_RUNTIME_INIT_H
#define GALOIS_RUNTIME_INIT_H

#include "galois/runtime/Statistics.h"
#include "galois/runtime/PagePool.h"
#include "galois/substrate/Init.h"

#include <string>

namespace galois {
namespace runtime {

template <typename SM>
class SharedMemRuntime: public galois::substrate::SharedMemSubstrate {

  using Base = galois::substrate::SharedMemSubstrate;

  internal::PageAllocState<> m_pa;
  SM m_sm;

public:
  explicit SharedMemRuntime(void)
    :
      Base(),
      m_pa(),
      m_sm()
    {
      internal::setPagePoolState(&m_pa);
      internal::setSysStatManager(&m_sm);
    }

  ~SharedMemRuntime(void) {
    m_sm.print();
    internal::setSysStatManager(nullptr);
    internal::setPagePoolState(nullptr);
  }
};

} // end namespace runtime
} // end namespace galois


#endif// GALOIS_RUNTIME_INIT_H

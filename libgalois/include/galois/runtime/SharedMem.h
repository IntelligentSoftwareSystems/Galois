/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#ifndef GALOIS_RUNTIME_SHAREDMEM_H
#define GALOIS_RUNTIME_SHAREDMEM_H

#include <string>

#include "galois/config.h"
#include "galois/runtime/PagePool.h"
#include "galois/runtime/Statistics.h"
#include "galois/substrate/SharedMem.h"

namespace galois::runtime {

template <typename SM>
class SharedMem : public galois::substrate::SharedMem {
  internal::PageAllocState<> m_pa;
  SM m_sm;

public:
  explicit SharedMem() : m_pa(), m_sm() {
    internal::setPagePoolState(&m_pa);
    internal::setSysStatManager(&m_sm);
  }

  ~SharedMem() {
    m_sm.print();
    internal::setSysStatManager(nullptr);
    internal::setPagePoolState(nullptr);
  }

  SharedMem(const SharedMem&)            = delete;
  SharedMem& operator=(const SharedMem&) = delete;

  SharedMem(SharedMem&&)            = delete;
  SharedMem& operator=(SharedMem&&) = delete;
};

} // namespace galois::runtime

#endif

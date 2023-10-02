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

#ifndef GALOIS_SUBSTRATE_SHAREDMEM_H
#define GALOIS_SUBSTRATE_SHAREDMEM_H

#include <memory>

#include "galois/config.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/Termination.h"

namespace galois::substrate {

class SharedMem {

  // Order is critical here
  ThreadPool m_tpool;

  std::unique_ptr<internal::LocalTerminationDetection<>> m_termPtr;
  std::unique_ptr<internal::BarrierInstance<>> m_biPtr;

public:
  /**
   * Initializes the Substrate library components
   */
  SharedMem();

  /**
   * Destroys the Substrate library components
   */
  ~SharedMem();

  SharedMem(const SharedMem&)            = delete;
  SharedMem& operator=(const SharedMem&) = delete;

  SharedMem(SharedMem&&)            = delete;
  SharedMem& operator=(SharedMem&&) = delete;
};

} // namespace galois::substrate

#endif

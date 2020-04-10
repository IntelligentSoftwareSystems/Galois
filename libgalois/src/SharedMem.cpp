/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/substrate/SharedMem.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/Termination.h"

#include <memory>

galois::substrate::SharedMem::SharedMem() {
  internal::setThreadPool(&m_tpool);

  // delayed initialization because both call getThreadPool in constructor
  // which is valid only after setThreadPool() above
  m_biPtr   = std::make_unique<internal::BarrierInstance<>>();
  m_termPtr = std::make_unique<internal::LocalTerminationDetection<>>();

  internal::setBarrierInstance(m_biPtr.get());
  internal::setTermDetect(m_termPtr.get());
}

galois::substrate::SharedMem::~SharedMem() {
  internal::setTermDetect(nullptr);
  internal::setBarrierInstance(nullptr);

  // destructors can call getThreadPool(), hence must be destroyed before
  // setThreadPool() below
  m_termPtr.reset();
  m_biPtr.reset();

  internal::setThreadPool(nullptr);
}

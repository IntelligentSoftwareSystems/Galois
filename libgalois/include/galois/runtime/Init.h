/** Runtime Init -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 * Initializes the components of galois::runtime library
 *
 * @author M. Amber Hassaan<ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_INIT_H
#define GALOIS_RUNTIME_INIT_H

#include "galois/runtime/Statistics.h"
#include "galois/runtime/PagePool.h"
#include "galois/Substrate/Init.h"

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

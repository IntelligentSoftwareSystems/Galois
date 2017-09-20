/** Galois Barrier master file -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 *
 * Public API for interacting with barriers
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Substrate/Barrier.h"

//anchor vtable
galois::substrate::Barrier::~Barrier() {}

//galois::substrate::Barrier& galois::substrate::getSystemBarrier(unsigned activeThreads) {
//  return benchmarking::getTopoBarrier(activeThreads);
//}

static galois::substrate::internal::BarrierInstance<>* BI = nullptr;

void galois::substrate::internal::setBarrierInstance(internal::BarrierInstance<>* bi) {
  GALOIS_ASSERT(!(bi && BI), "Double initialization of BarrierInstance");
  BI = bi;
}

galois::substrate::Barrier& galois::substrate::getBarrier(unsigned numT) {
  GALOIS_ASSERT(BI, "BarrierInstance not initialized");
  return BI->get(numT);
}


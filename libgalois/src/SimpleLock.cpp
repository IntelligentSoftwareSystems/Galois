/** SimpleLocks -*- C++ -*-
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
 * This contains support for SimpleLock support code.
 * See SimpleLock.h.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
*/

#include "galois/substrate/SimpleLock.h"

void galois::substrate::SimpleLock::slow_lock() const {
  int oldval = 0;
  do {
    while (_lock.load(std::memory_order_acquire) != 0) {
      asmPause();
    }
    oldval = 0;
  } while (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel, std::memory_order_relaxed));
  assert(is_locked());
}

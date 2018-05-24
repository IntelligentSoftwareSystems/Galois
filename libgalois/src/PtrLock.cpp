/** PtrLocks -*- C++ -*-
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
 * This contains support for PtrLock support code.
 * See PtrLock.h.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
*/

#include "galois/substrate/PtrLock.h"

void galois::substrate::internal::ptr_slow_lock(std::atomic<uintptr_t>& _l) {
  uintptr_t oldval;
  do {
    while ((_l.load(std::memory_order_acquire) & 1) != 0) {
      asmPause();
    }
    oldval = _l.fetch_or(1, std::memory_order_acq_rel);
  } while (oldval & 1);
  assert(_l);
}


/** Galois IO routines -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * IO support for galois.  We use this to handle output redirection,
 * and common formating issues.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/ErrorFeedBack.h"
#include "Galois/Runtime/SimpleLock.h"

#include <cassert>
#include <iostream>
#include <mutex>

static Galois::Runtime::SimpleLock plock;

void Galois::Runtime::detail::printFatal(const std::string& s) {
  plock.lock();
  std::cerr << "ERROR: " << s << "\n";
  plock.unlock();
  assert(0 && "Fatal error");
  abort();
}

void Galois::Runtime::detail::printWarning(const std::string& s) {
  std::lock_guard<SimpleLock> lg(plock);
  std::cerr << "WARNING: " << s << "\n";
}
void Galois::Runtime::detail::printTrace(const std::string& s) {
  std::lock_guard<SimpleLock> lg(plock);
  std::cerr << "TRACE: " << s << "\n";
}

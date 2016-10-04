/** Error Reporting -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Report Errors in a unified ways
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_ERRORFEEDBACK_H
#define GALOIS_RUNTIME_ERRORFEEDBACK_H

#include <sstream>

namespace Galois {
namespace Runtime {

namespace detail {
void printFatal(const std::string&);
void printWarning(const std::string&);
void printTrace(const std::string&);
} // namespace detail

template<typename... Args>
void gDie(Args&&... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  detail::printFatal(os.str());
}

template<typename... Args>
void gWarn(Args&&... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  detail::printWarning(os.str());
}

} // end namespace Runtime
} // end namespace Galois

#ifdef NDEBUG
#define TRACE(...) do {} while(false)
#else
template<typename... Args>
void TRACE(Args&&... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  //  os << "\n";
  Galois::Runtime::detail::printTrace(os.str());
}

//comment out to trace
#define TRACE(...) do {} while (false)

#endif

#endif //_HWTOPO_H

/** Galois Distributed Object Tracer -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_TRACER_H
#define GALOIS_RUNTIME_TRACER_H

#include "Galois/Runtime/ll/EnvCheck.h"

#include <sstream>

namespace Galois {
namespace Runtime {

namespace detail {

static inline void traceImpl(std::ostringstream& os, const char* format) {
  os << format;
}

template<typename T, typename... Args>
static inline void traceImpl(std::ostringstream& os, const char* format, T value, Args... args) {
  for (; *format != '\0'; format++) {
    if (*format == '%') {
      os << value;
      traceImpl(os, format + 1, args...);
      return;
    }
    os << *format;
  }
}

void printTrace(std::ostringstream&);

extern bool doTrace;
extern bool initTrace;

} // namespace detail

//FIXME use better forwarding
template<typename... Args>
static inline void trace(const char* format, Args... args) {
  if (!detail::initTrace) {
    detail::doTrace = LL::EnvCheck("GALOIS_DEBUG_TRACE");
    detail::initTrace = true;
  }
  if (detail::doTrace) {
    std::ostringstream os;
    detail::traceImpl(os, format, args...);
    detail::printTrace(os);
  }
}

} // namespace Runtime
} // namespace Galois

#endif

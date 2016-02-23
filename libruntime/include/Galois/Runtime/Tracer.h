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

#include "Galois/Substrate/EnvCheck.h"

#include <sstream>
#include <functional>
#include <vector>

namespace Galois {
namespace Runtime {

namespace detail {

static inline void traceImpl(std::ostringstream& os, const char* format) {
  os << format;
}

template<typename T, typename... Args>
static inline void traceImpl(std::ostringstream& os, const char* format, T&& value, Args&&... args) {
  for (; *format != '\0'; format++) {
    if (*format == '%') {
      os << value;
      traceImpl(os, format + 1, std::forward<Args>(args)...);
      return;
    }
    os << *format;
  }
}

void printTrace(std::ostringstream&);
void print_output_impl(std::ostringstream&);
void print_send_impl(std::vector<uint8_t>, size_t, unsigned);
void print_recv_impl(std::vector<uint8_t>, size_t, unsigned);

extern bool doTrace;
extern bool initTrace;

} // namespace detail

template<typename... Args>
static inline void trace(const char* format, Args&&... args) {
  if (!detail::initTrace) {
    detail::doTrace = Substrate::EnvCheck("GALOIS_DEBUG_TRACE");
    detail::initTrace = true;
  }
  if (detail::doTrace) {
    std::ostringstream os;
    detail::traceImpl(os, format, std::forward<Args>(args)...);
    detail::printTrace(os);
  }
}

template<typename... Args>
static inline void printOutput(const char* format, Args&&... args) {
    std::ostringstream os;
    detail::traceImpl(os, format, std::forward<Args>(args)...);
    detail::print_output_impl(os);
}

static void print_send(std::vector<uint8_t> vec, size_t len, unsigned host){
  detail::print_send_impl(vec, len, host);
}

static void print_recv(std::vector<uint8_t> vec, size_t len, unsigned host){
  detail::print_recv_impl(vec, len, host);
}
} // namespace Runtime
} // namespace Galois

#endif

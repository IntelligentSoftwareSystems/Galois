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

namespace galois {
namespace runtime {

namespace detail {

static inline void traceImpl(std::ostringstream& os) {
  os << "\n";
}

template<typename T, typename... Args>
static inline void traceImpl(std::ostringstream& os, T&& value, Args&&... args) {
  os << value << " ";
  traceImpl(os, std::forward<Args>(args)...);
}

static inline void traceFormatImpl(std::ostringstream& os, const char* format) {
  os << format;
}

template<typename T, typename... Args>
static inline void traceFormatImpl(std::ostringstream& os, const char* format, T&& value, Args&&... args) {
  for (; *format != '\0'; format++) {
    if (*format == '%') {
      os << value;
      traceFormatImpl(os, format + 1, std::forward<Args>(args)...);
      return;
    }
    os << *format;
  }
}

template<typename T, typename A>
class vecPrinter {
  const std::vector<T,A>& v;
public:
  vecPrinter(const std::vector<T,A>& _v) :v(_v) {}
  void print(std::ostream& os) const {
    os << "< " << v.size() << " : ";
    for (auto& i : v)
      os << " " << (int)i;
    os << ">";
  }
};

template<typename T, typename A>
std::ostream& operator<<(std::ostream& os, const vecPrinter<T,A>& vp) {
  vp.print(os);
  return os;
}

void printTrace(std::ostringstream&);
void print_output_impl(std::ostringstream&);
void print_send_impl(std::vector<uint8_t>, size_t, unsigned);
void print_recv_impl(std::vector<uint8_t>, size_t, unsigned);

extern bool doTrace;
extern bool initTrace;

} // namespace detail

template<typename T, typename A>
detail::vecPrinter<T,A> printVec(const std::vector<T,A>& v) {
  return detail::vecPrinter<T,A>(v);
};

#ifdef NDEBUG

template<typename... Args>
static inline void trace(Args&& ...) {}

#else

template<typename... Args>
static inline void trace(Args&&... args) {
  if (!detail::initTrace) {
    detail::doTrace = substrate::EnvCheck("GALOIS_DEBUG_TRACE");
    detail::initTrace = true;
  }
  if (detail::doTrace) {
    std::ostringstream os;
    detail::traceImpl(os, std::forward<Args>(args)...);
    detail::printTrace(os);
  }
}

#endif

template<typename... Args>
static inline void printOutput(const char* format, Args&&... args) {
    std::ostringstream os;
    detail::traceFormatImpl(os, format, std::forward<Args>(args)...);
    detail::print_output_impl(os);
}

static inline void print_send(std::vector<uint8_t> vec, size_t len, unsigned host){
  detail::print_send_impl(vec, len, host);
}

} // namespace runtime
} // namespace galois

#endif

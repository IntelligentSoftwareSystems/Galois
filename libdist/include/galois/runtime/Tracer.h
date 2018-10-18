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

/**
 * @file Tracer.h
 *
 * Includes functions for tracing output and printing data.
 */
#ifndef GALOIS_RUNTIME_TRACER_H
#define GALOIS_RUNTIME_TRACER_H

#include "galois/substrate/EnvCheck.h"
#include "galois/PODResizeableArray.h"

#include <sstream>
#include <functional>
#include <vector>

namespace galois {
namespace runtime {

namespace internal {

/**
 * Base case for traceImpl; ends the line with a new line.
 */
static inline void traceImpl(std::ostringstream& os) { os << "\n"; }

/**
 * Prints out a value to the output stream.
 */
template <typename T, typename... Args>
static inline void traceImpl(std::ostringstream& os, T&& value,
                             Args&&... args) {
  // os << value << " ";
  traceImpl(os, std::forward<Args>(args)...);
}

/**
 * Format string to os.
 */
static inline void traceFormatImpl(std::ostringstream& os, const char* format) {
  os << format;
}

/**
 * Format string to os as well as something else to print.
 */
template <typename T, typename... Args>
static inline void traceFormatImpl(std::ostringstream& os, const char* format,
                                   T&& value, Args&&... args) {
  for (; *format != '\0'; format++) {
    if (*format == '%') {
      os << value;
      traceFormatImpl(os, format + 1, std::forward<Args>(args)...);
      return;
    }
    os << *format;
  }
}

/**
 * Class to print a vector.
 */
template <typename T>
class vecPrinter {
  const galois::PODResizeableArray<T>& v;

public:
  vecPrinter(const galois::PODResizeableArray<T>& _v) : v(_v) {}
  void print(std::ostream& os) const {
    os << "< " << v.size() << " : ";
    for (auto& i : v)
      os << " " << (int)i;
    os << ">";
  }
};

/**
 * Operator to print a vector given a vecPrinter object
 */
template <typename T, typename A>
std::ostream& operator<<(std::ostream& os, const vecPrinter<T>& vp) {
  vp.print(os);
  return os;
}

/**
 * Prints trace data (which has time data included).
 */
void printTrace(std::ostringstream&);

/**
 * Prints out string stream.
 */
void print_output_impl(std::ostringstream&);

extern bool doTrace;
extern bool initTrace;

} // namespace internal

/**
 * Given a vector, returns a vector printer object that is able
 * to print the vector out onto an output stream.
 */
template <typename T>
internal::vecPrinter<T> printVec(const galois::PODResizeableArray<T>& v) {
  return internal::vecPrinter<T>(v);
};

/**
 * Prints a trace log of the provided arguments if debug mode is on.
 */
#ifdef NDEBUG
template <typename... Args>
static inline void trace(Args&&...) {}
#else
template <typename... Args>
static inline void trace(Args&&... args) {
  if (!internal::initTrace) {
    internal::doTrace   = substrate::EnvCheck("GALOIS_DEBUG_TRACE");
    internal::initTrace = true;
  }
  if (internal::doTrace) {
    std::ostringstream os;
    internal::traceImpl(os, std::forward<Args>(args)...);
    internal::printTrace(os);
  }
}
#endif

/**
 * Prints data to an output stream.
 *
 * @param format Format string
 * @param args data to print
 */
template <typename... Args>
static inline void printOutput(const char* format, Args&&... args) {
  std::ostringstream os;
  internal::traceFormatImpl(os, format, std::forward<Args>(args)...);
  internal::print_output_impl(os);
}
} // namespace runtime
} // namespace galois

#endif

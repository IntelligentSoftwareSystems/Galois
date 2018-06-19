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

#ifndef GALOIS_GIO_H
#define GALOIS_GIO_H

#include <sstream>
#include <cerrno>
#include <cstdlib>
#include <string.h>

// FIXME: move to Runtime

namespace galois {

//! Prints a string
void gPrintStr(const std::string&);
//! Prints an info string (for easy parsing)
void gInfoStr(const std::string&);
//! Prints a warning string (for easy parsing)
void gWarnStr(const std::string&);
//! Prints a debug string (for easy parsing)
void gDebugStr(const std::string&);
//! Prints an error string (for easy parsing)
void gErrorStr(const std::string&);

//! Prints a sequence of things
template <typename... Args>
void gPrint(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gPrintStr(os.str());
}

//! Prints an info string from a sequence of things
template <typename... Args>
void gInfo(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gInfoStr(os.str());
}

//! Prints a warning string from a sequence of things
template <typename... Args>
void gWarn(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gWarnStr(os.str());
}

//! Prints a debug string from a sequence of things; prints nothing if NDEBUG
//! is defined.
template <typename... Args>
void gDebug(const Args&... args) {
#ifndef NDEBUG
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gDebugStr(os.str());
#endif
}

//! Prints error message
template <typename... Args>
void gError(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gErrorStr(os.str());
}

void gFlush();

#define GALOIS_SYS_DIE(...)                                                    \
  do {                                                                         \
    galois::gError(__FILE__, ":", __LINE__, ": ", strerror(errno), ": ",       \
                   ##__VA_ARGS__);                                             \
    abort();                                                                   \
  } while (0)
#define GALOIS_DIE(...)                                                        \
  do {                                                                         \
    galois::gError(__FILE__, ":", __LINE__, ": ", ##__VA_ARGS__);              \
    abort();                                                                   \
  } while (0)
//! Like assert but unconditionally executed
#define GALOIS_ASSERT(cond, ...)                                               \
  do {                                                                         \
    bool b = (cond);                                                           \
    if (!b) {                                                                  \
      galois::gError(__FILE__, ":", __LINE__, ": assertion failed: ", #cond,   \
                     " ", ##__VA_ARGS__);                                      \
      abort();                                                                 \
    }                                                                          \
  } while (0)

template <unsigned ENABLE>
struct debug {
  template <typename... Args>
  static void print(const Args&... args) {
    gDebug(args...);
  }
};

template <>
struct debug<0> {
  template <typename... Args>
  inline static void print(const Args&... args) {}
};

} // end namespace galois

#endif //_GIO_H

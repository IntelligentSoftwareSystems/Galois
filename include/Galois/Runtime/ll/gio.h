/** Galois IO routines -*- C++ -*-
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
 * @section Description
 *
 * IO support for galois.  We use this to handle output redirection,
 * and common formating issues.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_LL_GIO_H
#define GALOIS_RUNTIME_LL_GIO_H

#include <sstream>
#include <cerrno>

namespace Galois {
namespace Runtime {
namespace LL {

//! Prints a string
void gPrintStr(const std::string&);
//! Prints an info string (for easy parsing)
void gInfoStr(const std::string&);
//! Prints a warning string (for easy parsing)
void gWarnStr(const std::string&);
//! Prints a debug string (for easy parsing)
void gDebugStr(const std::string&);
//! Prints an error string (for easy parsing)
void gErrorStr(const char* filename, int lineno, const std::string&);
//! Prints an error string (for easy parsing)
void gSysErrorStr(const char* filename, int lineno, int err, const std::string&);

//! Converts a sequence of things to a string
template<typename T>
bool toString(std::ostringstream& os, const T& val) { os << val; return true; }

//! Prints a sequence of things
template<typename... Args>
void gPrint(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) bool tmp[] = {toString(os, args)...};
  gPrintStr(os.str());
}

//! Prints an info string from a sequence of things
template<typename... Args>
void gInfo(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) bool tmp[] = {toString(os, args)...};
  gInfoStr(os.str());
}

//! Prints a warning string from a sequence of things
template<typename... Args>
void gWarn(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) bool tmp[] = {toString(os, args)...};
  gWarnStr(os.str());
}

//! Prints a debug string from a sequence of things; prints nothing if NDEBUG
//! is defined.
template<typename... Args>
void gDebug(Args... args) {
#ifndef NDEBUG
  std::ostringstream os;
  __attribute__((unused)) bool tmp[] = {toString(os, args)...};
  gDebugStr(os.str());
#endif
}

//! Prints error message
template<typename... Args>
void gError(const char* filename, int lineno, Args... args) {
  std::ostringstream os;
  __attribute__((unused)) bool tmp[] = {toString(os, args)...};
  gErrorStr(filename, lineno, os.str());
}

/**
 * Prints error message for function that set errno
 */
template<typename... Args>
void gSysError(const char* filename, int lineno, Args... args) {
  int err = errno;
  std::ostringstream os;
  __attribute__((unused)) bool tmp[] = {toString(os, args)...};
  gSysErrorStr(filename, lineno, err, os.str());
}

void gFlush();

#define GALOIS_SYS_ERROR(...) { Galois::Runtime::LL::gSysError(__FILE__, __LINE__, ##__VA_ARGS__); }
#define GALOIS_ERROR(...) { Galois::Runtime::LL::gError(__FILE__, __LINE__, ##__VA_ARGS__); }
#define GALOIS_SYS_DIE(...) { Galois::Runtime::LL::gSysError(__FILE__, __LINE__, ##__VA_ARGS__); abort(); }
#define GALOIS_DIE(...) { Galois::Runtime::LL::gError(__FILE__, __LINE__, ##__VA_ARGS__); abort(); }

}
}
} // end namespace Galois

#endif //_GIO_H

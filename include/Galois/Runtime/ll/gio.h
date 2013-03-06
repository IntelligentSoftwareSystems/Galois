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

namespace Galois {
namespace Runtime {
namespace LL {

//Print a string
void gPrintStr(const std::string&);
//print an info string (for easy parsing)
void gInfoStr(const std::string&);
//print a warning string (for easy parsing)
void gWarnStr(const std::string&);
//print a debug string (for easy parsing)
void gDebugStr(const std::string&);

//Convert a sequence of things to a string
inline void toString(std::ostringstream& os) { }

template<typename T>
void toString(std::ostringstream& os, const T& val) {
  os << val;
}

template<typename T1, typename T2, typename... Args>
void toString(std::ostringstream& os, const T1& val1, const T2& val2, Args ...args) {
  toString(os, val1);
  toString(os, val2);
  toString(os, args...);
}

template<typename... Args>
void gPrint(Args... args) {
  std::ostringstream os;
  toString(os, args...);
  gPrintStr(os.str());
}

template<typename... Args>
void gInfo(Args... args) {
  std::ostringstream os;
  toString(os, args...);
  gInfoStr(os.str());
}

template<typename... Args>
void gWarn(Args... args) {
  std::ostringstream os;
  toString(os, args...);
  gWarnStr(os.str());
}

template<typename... Args>
void gDebug(Args... args) {
#ifndef NDEBUG
  std::ostringstream os;
  toString(os, args...);
  gDebugStr(os.str());
#endif
}

void gError(bool doabort, const char* filename, int lineno, const char* format, ...);
void gSysError(bool doabort, const char* filename, int lineno, const char* format, ...);
void gFlush();

#ifndef NDEBUG
#define GALOIS_DEBUG(...) { Galois::Runtime::LL::gDebug (__VA_ARGS__); }
#else
#define GALOIS_DEBUG(...) { do {} while (false); }
#endif

#define GALOIS_SYS_ERROR(doabort, ...) { Galois::Runtime::LL::gSysError(doabort, __FILE__, __LINE__, ##__VA_ARGS__); }
#define GALOIS_ERROR(doabort, ...) { Galois::Runtime::LL::gError(doabort, __FILE__, __LINE__, ##__VA_ARGS__); }
}
}
} // end namespace Galois

#endif //_GIO_H

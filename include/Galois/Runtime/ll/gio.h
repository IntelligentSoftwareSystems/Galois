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

namespace GaloisRuntime {
namespace LL {

void gPrint(const char* format, ...);
void gDebug(const char* format, ...);
void gInfo(const char* format, ...);
void gWarn(const char* format, ...);
void gError(bool doabort, const char* filename, int lineno, const char* format, ...);
void gSysError(bool doabort, const char* filename, int lineno, const char* format, ...);
void gFlush();

#ifndef NDEBUG
#define GALOIS_DEBUG(...) { GaloisRuntime::LL::gDebug (__VA_ARGS__); }
#else
#define GALOIS_DEBUG(...) { do {} while (false); }
#endif

#define GALOIS_SYS_ERROR(doabort, ...) { GaloisRuntime::LL::gSysError(doabort, __FILE__, __LINE__, ##__VA_ARGS__); }
#define GALOIS_ERROR(doabort, ...) { GaloisRuntime::LL::gError(doabort, __FILE__, __LINE__, ##__VA_ARGS__); }
}
}

#endif //_GIO_H

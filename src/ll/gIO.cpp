/** Galois IO routines -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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

#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/ll/EnvCheck.h"

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <cstdarg>
#include <cerrno>
#include <iostream>
#include <fstream>
#include <iomanip>

static Galois::Runtime::LL::SimpleLock<true> IOLock;

void Galois::Runtime::LL::gDebugStr(const std::string& s) {
#ifndef NDEBUG
  static bool skip = EnvCheck("GALOIS_DEBUG_SKIP");
  if (skip) return;
  static const unsigned TIME_STR_SIZE = 32;
  char time_str[TIME_STR_SIZE];
  time_t rawtime;
  struct tm* timeinfo;

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(time_str, TIME_STR_SIZE, "[%H:%M:%S]", timeinfo);

  IOLock.lock ();
  if (EnvCheck("GALOIS_DEBUG_TO_FILE")) {
    static std::ofstream debugOut;
    if (!debugOut.is_open()) {
      char fname[] = "gdebugXXXXXX";
      debugOut.open(mktemp(fname));
      IOLock.unlock();
      gInfo("Debug output going to ", fname);
      IOLock.lock();
    }

    debugOut << "[" << time_str << " " << std::setw(3) << getTID() << "] " << s << "\n";
    debugOut.flush();
  } else {
    std::cerr << "[" << time_str << " " << std::setw(3) << getTID() << "] " << s << "\n";
  }
  IOLock.unlock();
#endif
}

void Galois::Runtime::LL::gPrintStr(const std::string& s) {
  IOLock.lock();
  std::cout << s;
  IOLock.unlock();
}

void Galois::Runtime::LL::gInfoStr(const std::string& s) {
  IOLock.lock();
  std::cout << "INFO: " << s << "\n";
  IOLock.unlock();
}

void Galois::Runtime::LL::gWarnStr(const std::string& s) {
  IOLock.lock();
  std::cout << "WARNING: " << s << "\n";
  IOLock.unlock();
}

void Galois::Runtime::LL::gError(bool doabort, const char* filename, int lineno, const char* format, ...) {
  IOLock.lock();
  va_list ap;
  va_start(ap, format);
  fprintf(stderr, "ERROR: %s:%d ", filename, lineno);
  vfprintf(stderr, format, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  IOLock.unlock();
  if (doabort)
    abort();
}

void Galois::Runtime::LL::gSysError(bool doabort, const char* filename, int lineno, const char* format, ...) {
  int err_saved = errno;
  IOLock.lock();
  va_list ap;
  va_start(ap, format);
  fprintf(stderr, "ERROR: %s:%d: %s: ", filename, lineno, strerror(err_saved));
  vfprintf(stderr, format, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  IOLock.unlock();
  if (doabort)
    abort();
}

void Galois::Runtime::LL::gFlush() {
  fflush(stdout);
}

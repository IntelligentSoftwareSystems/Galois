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

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <cstdarg>
#include <cerrno>

static GaloisRuntime::LL::SimpleLock<true> IOLock;

void GaloisRuntime::LL::gPrint(const char* format, ...) {
  IOLock.lock();
  va_list ap;
  va_start(ap, format);
  vprintf(format, ap);
  va_end(ap);
  IOLock.unlock();
}

void GaloisRuntime::LL::gDebug(const char* format, ...) {
  static const unsigned TIME_STR_SIZE = 32;
  char time_str[TIME_STR_SIZE];
  time_t rawtime;
  struct tm* timeinfo;

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(time_str, TIME_STR_SIZE, "[%H:%M:%S]", timeinfo);

  // IOLock.lock ();
  va_list ap;

  va_start (ap, format);

  char msg[1024];
  vsprintf (msg, format, ap);
  va_end(ap);
  // vprintf (format, ap);
  printf ("[%s Thrd:%-3d] %s\n", time_str, GaloisRuntime::LL::getTID (), msg);
  // fflush (stdout);

  // IOLock.unlock();
}

void GaloisRuntime::LL::gInfo(const char* format, ...) {
  IOLock.lock();
  va_list ap;
  va_start(ap, format);
  printf("INFO: ");
  vprintf(format, ap);
  printf("\n");
  va_end(ap);
  IOLock.unlock();
}

void GaloisRuntime::LL::gWarn(const char* format, ...) {
  IOLock.lock();
  va_list ap;
  va_start(ap, format);
  fprintf(stderr, "WARNING: ");
  vfprintf(stderr, format, ap);
  printf("\n");
  va_end(ap);
  IOLock.unlock();
}

void GaloisRuntime::LL::gError(bool doabort, const char* filename, int lineno, const char* format, ...) {
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

void GaloisRuntime::LL::gSysError(bool doabort, const char* filename, int lineno, const char* format, ...) {
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

void GaloisRuntime::LL::gFlush() {
  fflush(stdout);
}

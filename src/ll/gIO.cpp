/** Galois IO routines -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#include "Galois/config.h"
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
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include GALOIS_CXX11_STD_HEADER(mutex)

static void printString(bool error, bool newline, const std::string prefix, const std::string s) {
  static Galois::Runtime::LL::SimpleLock<true> IOLock;
//  if (Galois::Runtime::networkHostID == 0) {
    std::lock_guard<decltype(IOLock)> lock(IOLock);
    std::ostream& o = error ? std::cerr : std::cout;
    if (prefix.length()) o << prefix << ": ";
    o << s;
    if (newline) o << "\n";
//  } else {
//    Galois::Runtime::getSystemNetworkInterface().sendAlt(0, printString, error, newline, host, prefix, s);
//  }
}

void Galois::Runtime::LL::gDebugStr(const std::string& s) {
  static bool skip = EnvCheck("GALOIS_DEBUG_SKIP");
  if (skip) return;
  static const unsigned TIME_STR_SIZE = 32;
  char time_str[TIME_STR_SIZE];
  time_t rawtime;
  struct tm* timeinfo;

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(time_str, TIME_STR_SIZE, "[%H:%M:%S]", timeinfo);

  std::ostringstream os;
  os << "[" << time_str << " " << std::setw(3) << getTID() << "] " << s;

  if (EnvCheck("GALOIS_DEBUG_TO_FILE")) {
    static Galois::Runtime::LL::SimpleLock<true> dIOLock;
    std::lock_guard<decltype(dIOLock)> lock(dIOLock);
    static std::ofstream debugOut;
    if (!debugOut.is_open()) {
      char fname[] = "gdebugXXXXXX";
      int fd = mkstemp(fname);
      close(fd);
      debugOut.open(fname);
      gInfo("Debug output going to ", fname);
    }
    debugOut << os.str() << "\n";
    debugOut.flush();
  } else {
    printString(true, true, "DEBUG", os.str());
  }
}

void Galois::Runtime::LL::gPrintStr(const std::string& s) {
  printString(false, false, "", s);
}

void Galois::Runtime::LL::gInfoStr(const std::string& s) {
  printString(false, true, "INFO", s);
}

void Galois::Runtime::LL::gWarnStr(const std::string& s) {
  printString(false, true, "WARNING", s);
}

void Galois::Runtime::LL::gErrorStr(const std::string& s) {
  printString(false, true, "ERROR", s);
}

void Galois::Runtime::LL::gFlush() {
  fflush(stdout);
}

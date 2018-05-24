#include "galois/gIO.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/EnvCheck.h"
#include "galois/substrate/ThreadPool.h"

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
#include <mutex>

static void printString(bool error, bool newline, const std::string& prefix, const std::string& s) {
  static galois::substrate::SimpleLock IOLock;
  std::lock_guard<decltype(IOLock)> lock(IOLock);
  std::ostream& o = error ? std::cerr : std::cout;
  if (prefix.length()) 
    o << prefix << ": ";
  o << s;
  if (newline) 
    o << "\n";
}

void galois::gDebugStr(const std::string& s) {
  static bool skip = galois::substrate::EnvCheck("GALOIS_DEBUG_SKIP");
  if (skip) return;
  static const unsigned TIME_STR_SIZE = 32;
  char time_str[TIME_STR_SIZE];
  time_t rawtime;
  struct tm* timeinfo;

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(time_str, TIME_STR_SIZE, "[%H:%M:%S]", timeinfo);

  std::ostringstream os;
  os << "[" << time_str << " " << std::setw(3) << galois::substrate::ThreadPool::getTID() << "] " << s;

  if (galois::substrate::EnvCheck("GALOIS_DEBUG_TO_FILE")) {
    static galois::substrate::SimpleLock dIOLock;
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

void galois::gPrintStr(const std::string& s) {
  printString(false, false, "", s);
}

void galois::gInfoStr(const std::string& s) {
  printString(false, true, "INFO", s);
}

void galois::gWarnStr(const std::string& s) {
  printString(false, true, "WARNING", s);
}

void galois::gErrorStr(const std::string& s) {
  printString(true, true, "ERROR", s);
}

void galois::gFlush() {
  fflush(stdout);
}

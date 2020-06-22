#include "galois/Logging.h"

#include <iostream>
#include <mutex>

#include "galois/GetEnv.h"

namespace {

void PrintString(bool error, bool flush, const std::string& prefix,
                 const std::string& s) {
  static std::mutex lock;
  std::lock_guard<std::mutex> lg(lock);

  std::ostream& o = error ? std::cerr : std::cout;
  if (!prefix.empty()) {
    o << prefix << ": ";
  }
  o << s << "\n";
  if (flush) {
    o.flush();
  }
}

} // end unnamed namespace

void galois::internal::LogString(galois::LogLevel level, const std::string& s) {
  switch (level) {
  case LogLevel::Debug:
    return PrintString(true, false, "DEBUG", s);
  case LogLevel::Verbose:
    if (galois::GetEnv("GALOIS_LOG_VERBOSE")) {
      return PrintString(true, false, "VERBOSE", s);
    }
    return;
  case LogLevel::Warning:
    return PrintString(true, false, "WARNING", s);
  case LogLevel::Error:
    return PrintString(true, false, "ERROR", s);
  default:
    std::abort();
  }
}

#ifndef GALOIS_GIO_H
#define GALOIS_GIO_H

#include <sstream>
#include <cerrno>
#include <cstdlib>
#include <string.h>

//FIXME: move to Runtime

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
template<typename... Args>
void gPrint(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gPrintStr(os.str());
}

//! Prints an info string from a sequence of things
template<typename... Args>
void gInfo(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gInfoStr(os.str());
}

//! Prints a warning string from a sequence of things
template<typename... Args>
void gWarn(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gWarnStr(os.str());
}

//! Prints a debug string from a sequence of things; prints nothing if NDEBUG
//! is defined.
template<typename... Args>
void gDebug(const Args&... args) {
#ifndef NDEBUG
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gDebugStr(os.str());
#endif
}

//! Prints error message
template<typename... Args>
void gError(Args... args) {
  std::ostringstream os;
  __attribute__((unused)) int tmp[] = {(os << args, 0)...};
  gErrorStr(os.str());
}

void gFlush();

#define GALOIS_SYS_DIE(...)   do { galois::gError(__FILE__, ":", __LINE__, ": ", strerror(errno), ": ",##__VA_ARGS__); abort(); } while (0)
#define GALOIS_DIE(...)       do { galois::gError(__FILE__, ":", __LINE__, ": ", ##__VA_ARGS__); abort(); } while (0)
//! Like assert but unconditionally executed
#define GALOIS_ASSERT(cond, ...) do { bool b = (cond); if (!b) { galois::gError(__FILE__, ":", __LINE__, ": assertion failed: ", #cond, " ", ##__VA_ARGS__); abort(); } } while (0)


template <unsigned ENABLE>
struct debug {
  template <typename... Args>
  static void print (const Args&... args) {
    gDebug (args...);
  }
};

template <>
struct debug<0> {
  template <typename... Args>
  inline static void print (const Args&... args) {}
};


} // end namespace galois

#endif //_GIO_H

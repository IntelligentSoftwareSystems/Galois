#ifndef GALOIS_RUNTIME_TRACER_H
#define GALOIS_RUNTIME_TRACER_H

#include "galois/substrate/EnvCheck.h"

#include <sstream>
#include <functional>
#include <vector>

namespace galois {
namespace runtime {

namespace internal {

static inline void traceImpl(std::ostringstream& os) {
  os << "\n";
}

template<typename T, typename... Args>
static inline void traceImpl(std::ostringstream& os, T&& value, Args&&... args) {
  os << value << " ";
  traceImpl(os, std::forward<Args>(args)...);
}

static inline void traceFormatImpl(std::ostringstream& os, const char* format) {
  os << format;
}

template<typename T, typename... Args>
static inline void traceFormatImpl(std::ostringstream& os, const char* format, T&& value, Args&&... args) {
  for (; *format != '\0'; format++) {
    if (*format == '%') {
      os << value;
      traceFormatImpl(os, format + 1, std::forward<Args>(args)...);
      return;
    }
    os << *format;
  }
}

template<typename T, typename A>
class vecPrinter {
  const std::vector<T,A>& v;
public:
  vecPrinter(const std::vector<T,A>& _v) :v(_v) {}
  void print(std::ostream& os) const {
    os << "< " << v.size() << " : ";
    for (auto& i : v)
      os << " " << (int)i;
    os << ">";
  }
};

template<typename T, typename A>
std::ostream& operator<<(std::ostream& os, const vecPrinter<T,A>& vp) {
  vp.print(os);
  return os;
}

void printTrace(std::ostringstream&);
void print_output_impl(std::ostringstream&);
void print_send_impl(std::vector<uint8_t>, size_t, unsigned);
void print_recv_impl(std::vector<uint8_t>, size_t, unsigned);

extern bool doTrace;
extern bool initTrace;

} // namespace internal

template<typename T, typename A>
internal::vecPrinter<T,A> printVec(const std::vector<T,A>& v) {
  return internal::vecPrinter<T,A>(v);
};

#ifdef NDEBUG
template<typename... Args>
static inline void trace(Args&& ...) {}
#else
template<typename... Args>
static inline void trace(Args&&... args) {
  if (!internal::initTrace) {
    internal::doTrace = substrate::EnvCheck("GALOIS_DEBUG_TRACE");
    internal::initTrace = true;
  }
  if (internal::doTrace) {
    std::ostringstream os;
    internal::traceImpl(os, std::forward<Args>(args)...);
    internal::printTrace(os);
  }
}
#endif

template<typename... Args>
static inline void printOutput(const char* format, Args&&... args) {
    std::ostringstream os;
    internal::traceFormatImpl(os, format, std::forward<Args>(args)...);
    internal::print_output_impl(os);
}

static inline void print_send(std::vector<uint8_t> vec, size_t len, unsigned host){
  internal::print_send_impl(vec, len, host);
}

} // namespace runtime
} // namespace galois

#endif

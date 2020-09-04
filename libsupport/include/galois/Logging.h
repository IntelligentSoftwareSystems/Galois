#ifndef GALOIS_LIBSUPPORT_GALOIS_LOGGING_H_
#define GALOIS_LIBSUPPORT_GALOIS_LOGGING_H_

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <sstream>
#include <string>
#include <system_error>

// Small patch to work with libfmt 4.0, which is the version in Ubuntu 18.04.
#ifndef FMT_STRING
#define FMT_STRING(...) __VA_ARGS__
#endif

#if FMT_VERSION >= 60000
/// Introduce std::error_code to the fmt library. Otherwise, they will be
/// printed using ostream<< formatting (i.e., as an int).
template <>
struct fmt::formatter<std::error_code> : formatter<string_view> {

  template <typename FormatterContext>
  auto format(std::error_code c, FormatterContext& ctx) {
    return formatter<string_view>::format(c.message(), ctx);
  }
};
#endif

namespace galois {

enum class LogLevel {
  Debug   = 0,
  Verbose = 1,
  // Info = 2,  currently unused
  Warning = 3,
  Error   = 4,
};

namespace internal {

void LogString(LogLevel level, const std::string& s);

}

/// Log at a specific LogLevel.
///
/// \tparam F         string-like type
/// \param fmt_string a C++20-style fmt string (e.g., "hello {}")
/// \param args       arguments to fmt interpolation
template <typename F, typename... Args>
void Log(LogLevel level, F fmt_string, Args&&... args) {
  std::string s = fmt::format(fmt_string, std::forward<Args>(args)...);
  internal::LogString(level, s);
}

/// Log at a specific LogLevel with source code information.
///
/// \tparam F         string-like type
/// \param file_name  file name
/// \param line_no    line number
/// \param fmt_string a C++20-style fmt string (e.g., "hello {}")
/// \param args       arguments to fmt interpolation
template <typename F, typename... Args>
void LogLine(LogLevel level, const char* file_name, int line_no, F fmt_string,
             Args&&... args) {
  std::string s         = fmt::format(fmt_string, std::forward<Args>(args)...);
  std::string with_line = fmt::format("{}:{}: {}", file_name, line_no, s);
  internal::LogString(level, with_line);
}

} // end namespace galois

#define GALOIS_LOG_FATAL(fmt_string, ...)                                      \
  do {                                                                         \
    ::galois::LogLine(::galois::LogLevel::Error, __FILE__, __LINE__,           \
                      FMT_STRING(fmt_string), ##__VA_ARGS__);                  \
    ::std::abort();                                                            \
  } while (0)
#define GALOIS_LOG_ERROR(fmt_string, ...)                                      \
  do {                                                                         \
    ::galois::LogLine(::galois::LogLevel::Error, __FILE__, __LINE__,           \
                      FMT_STRING(fmt_string), ##__VA_ARGS__);                  \
  } while (0)
#define GALOIS_LOG_WARN(fmt_string, ...)                                       \
  do {                                                                         \
    ::galois::LogLine(::galois::LogLevel::Warning, __FILE__, __LINE__,         \
                      FMT_STRING(fmt_string), ##__VA_ARGS__);                  \
  } while (0)
#define GALOIS_LOG_VERBOSE(fmt_string, ...)                                    \
  do {                                                                         \
    ::galois::LogLine(::galois::LogLevel::Verbose, __FILE__, __LINE__,         \
                      FMT_STRING(fmt_string), ##__VA_ARGS__);                  \
  } while (0)

#ifndef NDEBUG
#define GALOIS_LOG_DEBUG(fmt_string, ...)                                      \
  do {                                                                         \
    ::galois::LogLine(::galois::LogLevel::Debug, __FILE__, __LINE__,           \
                      FMT_STRING(fmt_string), ##__VA_ARGS__);                  \
  } while (0)
#else
#define GALOIS_LOG_DEBUG(...)
#endif

#define GALOIS_LOG_ASSERT(cond)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ::galois::LogLine(::galois::LogLevel::Error, __FILE__, __LINE__,         \
                        "assertion not true: {}", #cond);                      \
      ::std::abort();                                                          \
    }                                                                          \
  } while (0)

#endif

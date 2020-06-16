#include "galois/Logging.h"

#include <system_error>

int main() {
  GALOIS_LOG_ERROR("string");
  GALOIS_LOG_ERROR("format string: {}", 42);
  GALOIS_LOG_ERROR("format string: {:d}", 42);
  // The following correctly fails with a compile time error
  // GALOIS_LOG_ERROR("basic format string {:s}", 42);
  GALOIS_LOG_WARN("format number: {:.2f}", 2.0 / 3.0);
  GALOIS_LOG_WARN("format error code: {}",
                  std::make_error_code(std::errc::invalid_argument));
  GALOIS_LOG_VERBOSE(
      "will be printed when environment variable GALOIS_LOG_VERBOSE=1");
  GALOIS_LOG_DEBUG("this will only be printed in debug builds");
  GALOIS_LOG_ASSERT(1 == 1);

  return 0;
}

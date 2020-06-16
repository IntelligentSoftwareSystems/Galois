#include "galois/GetEnv.h"
#include "galois/Logging.h"

int main() {
  GALOIS_LOG_ASSERT(galois::GetEnv("PATH"));

  std::string s;
  GALOIS_LOG_ASSERT(galois::GetEnv("PATH", &s));

  int i{};
  GALOIS_LOG_ASSERT(!galois::GetEnv("PATH", &i));

  double d{};
  GALOIS_LOG_ASSERT(!galois::GetEnv("PATH", &d));

  bool b{};
  GALOIS_LOG_ASSERT(!galois::GetEnv("PATH", &b));

  return 0;
}

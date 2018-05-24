#ifndef GALOIS_RUNTIME_COMPILER_HELPER_FUNCTIONS_H
#define GALOIS_RUNTIME_COMPILER_HELPER_FUNCTIONS_H
#include <atomic>
#include <algorithm>
#include <vector>
namespace galois {
  template<typename... Args>
  int read_set(Args... args) {
    // Nothing for now.
    return 0;
  }

  template<typename... Args>
  int write_set(Args... args) {
    // Nothing for now.
    return 0;
  }
} // end namespace galois
#endif

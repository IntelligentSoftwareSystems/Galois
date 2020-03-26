#ifndef GALOIS_LIBTSUBA_TSUBA_INTERNAL_H_
#define GALOIS_LIBTSUBA_TSUBA_INTERNAL_H_

#include <cerrno>
#include <thread>
#include <cstdint>

/* NOLINTNEXTLINE */
#define EXPORT_SYM extern "C" __attribute__((__visibility__("default")))

constexpr const uint64_t kKBShift = 10;
constexpr const uint64_t kMBShift = 20;
constexpr const uint64_t kGBShift = 30;

template <typename T>
constexpr T KB(T v) {
  return v << kKBShift;
}
template <typename T>
constexpr T MB(T v) {
  return v << kMBShift;
}
template <typename T>
constexpr T GB(T v) {
  return v << kGBShift;
}

namespace tsuba {

/* set errno and return */
template <typename T>
static inline T ERRNO_RET(int errno_val, T ret) {
  errno = errno_val;
  return ret;
}

} /* namespace tsuba */

#endif

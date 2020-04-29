#ifndef GALOIS_LIBTSUBA_TSUBA_TSUBA_H_
#define GALOIS_LIBTSUBA_TSUBA_TSUBA_H_

#include "tsuba/tsuba_api.h"

#include <string_view>

template <typename T>
constexpr T TsubaRoundDownToBlock(T val) {
  return val & TSUBA_BLOCK_MASK;
}
template <typename T>
constexpr T TsubaRoundUpToBlock(T val) {
  return TsubaRoundDownToBlock(val + TSUBA_BLOCK_OFFSET_MASK);
}

/* Check to see if the name is formed in a way that tsuba expects */
static inline bool TsubaIsUri(std::string_view uri) {
  return uri.find("s3://") == 0;
}

template <typename StrType, typename T>
static inline int TsubaPeek(const StrType& filename, T* obj) {
  return TsubaPeek(filename.c_str(),
                   reinterpret_cast<uint8_t*>(obj), /* NOLINT */
                   0, sizeof(*obj));
}

#endif

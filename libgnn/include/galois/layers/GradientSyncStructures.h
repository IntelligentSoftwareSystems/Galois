#pragma once
#include "galois/GNNTypes.h"

namespace galois {

//! Simple summation of values
struct WeightGradientSummation {
  using ValTy = GNNFloat;
  static ValTy extract(uint32_t, ValTy& weight) { return weight; }
  static bool reduce(uint32_t, ValTy& weight, ValTy y) {
    weight += y;
    return true;
  }

  //! reset weight to 0
  static void reset(uint32_t, ValTy& weight) { weight = 0.0; }

  //! save weight
  static void setVal(uint32_t, ValTy& weight, ValTy y) { weight = y; }

  // GPU options TODO for GPU
  static bool extract_batch(unsigned, uint8_t*, size_t*, DataCommMode*) {
    return false;
  }
  static bool extract_batch(unsigned, uint8_t*) { return false; }
  static bool extract_reset_batch(unsigned, uint8_t*, size_t*, DataCommMode*) {
    return false;
  }
  static bool extract_reset_batch(unsigned, uint8_t*) { return false; }
  static bool reduce_batch(unsigned, uint8_t*, DataCommMode) { return false; }
  static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) {
    return false;
  }
  static bool setVal_batch(unsigned, uint8_t*, DataCommMode) { return false; }
};

struct WeightGradientSet {
  using ValTy = GNNFloat;
  static ValTy extract(uint32_t, ValTy& weight) { return weight; }
  static bool reduce(uint32_t, ValTy&, ValTy) { return true; }

  //! reset weight to 0
  static void reset(uint32_t, ValTy& weight) { weight = 0.0; }

  //! save weight
  static void setVal(uint32_t, ValTy& weight, ValTy y) { weight = y; }

  // GPU options TODO for GPU
  static bool extract_batch(unsigned, uint8_t*, size_t*, DataCommMode*) {
    return false;
  }
  static bool extract_batch(unsigned, uint8_t*) { return false; }
  static bool extract_reset_batch(unsigned, uint8_t*, size_t*, DataCommMode*) {
    return false;
  }
  static bool extract_reset_batch(unsigned, uint8_t*) { return false; }
  static bool reduce_batch(unsigned, uint8_t*, DataCommMode) { return false; }
  static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) {
    return false;
  }
  static bool setVal_batch(unsigned, uint8_t*, DataCommMode) { return false; }
};

} // namespace galois

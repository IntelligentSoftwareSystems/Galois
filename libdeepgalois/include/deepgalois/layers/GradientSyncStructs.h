#ifndef GALOIS_ENABLE_GPU
#ifndef __GRAD_SYNC_STRUCT__
#define __GRAD_SYNC_STRUCT__

#include "deepgalois/types.h"

struct GradientSync {
  using ValTy = float_t;

  static ValTy extract(uint32_t, float_t& weight) { return weight; }

  static bool reduce(uint32_t, float_t& weight, ValTy y) {
    // TODO merge function here
    // for now make sure the weights are close enough
    // if (std::abs(weight - y) > 0.00001) {
    //  galois::gInfo("weight ", node_id, " not consistent with one received");
    //}
    if (y == 0) {
      galois::gPrint("nothing important\n");
    }
    weight += y;
    // need a post process divide all step
    //weight /= 2;
    return true;
  }

  //! reset weight to 0
  static void reset(uint32_t, float_t& weight) { weight = 0; }

  //! save weight
  static void setVal(uint32_t, float_t& weight, ValTy y) { weight = y; }

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

// no bitset; everything is sent anyways
#endif
#endif

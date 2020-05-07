#ifndef __GRAPH_CONV_SYNC_STRUCT__
#define __GRAPH_CONV_SYNC_STRUCT__

struct GraphConvSync {
  using ValTy = std::vector<float>;

  //! return a vector of floats to sync
  static ValTy extract(uint32_t node_id, char&) {
    // TODO figure out how to avoid copy from C array to vector; best
    // way is if original data is in a vector probably, but that has the
    // issue of not being able to directly call BLAS
    ValTy vecToReturn;
    // allocate space
    vecToReturn.resize(deepgalois::_syncVectorSize);
    // copy the node's data to vector to serialize/send
    for (unsigned i = 0; i < deepgalois::_syncVectorSize; i++) {
      vecToReturn[i] =
          deepgalois::_dataToSync[node_id * deepgalois::_syncVectorSize + i];
    }
    // move constructor should kick in here to avoid return copy
    return vecToReturn;
  }

  //! reduction is addition in this case; add received vector to
  //! own vector
  static bool reduce(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == deepgalois::_syncVectorSize);
    // loop and do addition
    for (unsigned i = 0; i < deepgalois::_syncVectorSize; i++) {
      deepgalois::_dataToSync[node_id * deepgalois::_syncVectorSize + i] +=
          y[i];
    }
    return true;
  }

  //! do nothing (waste of a write)
  static void reset(uint32_t, char&) {}

  //! element wise set
  static void setVal(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == deepgalois::_syncVectorSize);
    // loop and do addition
    for (unsigned i = 0; i < deepgalois::_syncVectorSize; i++) {
      deepgalois::_dataToSync[node_id * deepgalois::_syncVectorSize + i] = y[i];
    }
  }

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

#endif

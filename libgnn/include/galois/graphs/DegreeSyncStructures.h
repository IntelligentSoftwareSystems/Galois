#include "galois/GNNTypes.h"

namespace galois {
namespace graphs {

extern uint32_t* gnn_degree_vec_1_;
extern uint32_t* gnn_degree_vec_2_;

struct InitialDegreeSync {
  using ValTy = std::pair<uint32_t, uint32_t>;

  //! return a vector of floats to sync
  static ValTy extract(uint32_t lid, char&) {
    return std::make_pair(gnn_degree_vec_1_[lid], gnn_degree_vec_2_[lid]);
  }

  //! reduction is addition in this case; add received vector to
  //! own vector
  static bool reduce(uint32_t lid, char&, ValTy y) {
    gnn_degree_vec_1_[lid] += y.first;
    gnn_degree_vec_2_[lid] += y.second;
    if (y.first || y.second) {
      return true;
    } else {
      return false;
    }
  }

  //! No-op: readAny = overwritten anyways
  static void reset(uint32_t lid, char&) {
    gnn_degree_vec_1_[lid] = 0;
    gnn_degree_vec_2_[lid] = 0;
  }

  //! element wise set
  static void setVal(uint32_t lid, char&, ValTy y) {
    gnn_degree_vec_1_[lid] = y.first;
    gnn_degree_vec_2_[lid] = y.second;
  }

  // GPU options TODO for GPU
  static bool extract_batch(unsigned, uint8_t*, size_t*, DataCommMode*) {
    return false;
  }
  static bool extract_batch(unsigned, uint8_t*) { return false; }
  static bool reduce_batch(unsigned, uint8_t*, DataCommMode) { return false; }
  static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) {
    return false;
  }
  static bool setVal_batch(unsigned, uint8_t*, DataCommMode) { return false; }
  static bool extract_reset_batch(unsigned, uint8_t*, size_t*, DataCommMode*) {
    return false;
  }
  static bool extract_reset_batch(unsigned, uint8_t*) { return false; }
};

} // namespace graphs
} // namespace galois

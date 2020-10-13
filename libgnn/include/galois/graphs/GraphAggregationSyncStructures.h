// defined in GNNGraph.cpp; set in order to control which matrix
// gets synchronized
#include "galois/GNNTypes.h"
#include "galois/BufferWrapper.h"

namespace galois {
namespace graphs {

extern GNNFloat* gnn_matrix_to_sync_;
extern size_t gnn_matrix_to_sync_column_length_;

struct GNNSumAggregate {
  using ValTy = galois::BufferWrapper<GNNFloat>;

  //! return a vector of floats to sync
  static ValTy extract(uint32_t node_id, char&) {
    ValTy extracted_vec(
        &gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_],
        gnn_matrix_to_sync_column_length_);
    // move constructor should kick in here to avoid return copy
    return extracted_vec;
  }

  //! reduction is addition in this case; add received vector to
  //! own vector
  static bool reduce(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] +=
          y[i];
    }
    return true;
  }

  //! do nothing (waste of a write)
  static void reset(uint32_t, char&) {}

  //! element wise set
  static void setVal(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] =
          y[i];
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

} // namespace graphs
} // namespace galois

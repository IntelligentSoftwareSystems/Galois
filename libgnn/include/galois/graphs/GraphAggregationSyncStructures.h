// defined in GNNGraph.cpp; set in order to control which matrix
// gets synchronized
#include "galois/GNNTypes.h"
#ifdef GALOIS_ENABLE_GPU
#include "galois/GNNCudaContextHostDecls.h"
#endif

namespace galois {
namespace graphs {

extern GNNFloat* gnn_matrix_to_sync_;
extern size_t gnn_matrix_to_sync_column_length_;
extern galois::DynamicBitSet bitset_graph_aggregate;
extern galois::LargeArray<uint32_t>* gnn_lid_to_sid_pointer_;
extern galois::DynamicBitSet bitset_sample_flag_;
extern size_t subgraph_size_;
#ifdef GALOIS_ENABLE_GPU
extern struct CUDA_Context* cuda_ctx_for_sync;
extern unsigned layer_number_to_sync;
#endif

struct SampleFlagSync {
  using ValTy = char;

  //! return a vector of floats to sync
  static ValTy extract(uint32_t, char& i) { return i; }

  static bool reduce(uint32_t, char& i, ValTy y) {
    if (y) {
      i = y;
      assert(i == 1);
      return true;
    } else {
      return false;
    }
  }

  //! No-op: readAny = overwritten anyways
  static void reset(uint32_t, char&) {}

  //! element wise set
  static void setVal(uint32_t, char& i, ValTy y) { i = y; }

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

struct SampleFlagBitset {
  static constexpr bool is_vector_bitset() { return false; }
  static constexpr bool is_valid() { return true; }
  static galois::DynamicBitSet& get() { return bitset_sample_flag_; }
  static void reset_range(size_t begin, size_t end) {
    bitset_sample_flag_.reset(begin, end);
  }
};

struct GNNSumAggregate {
  using ValTy = galois::gstl::Vector<GNNFloat>;

  static size_t FeatVecSize() { return gnn_matrix_to_sync_column_length_; }

  //! return a vector of floats to sync
  static ValTy extract(uint32_t node_id, char&) {
    // It should be a CPU synchronizing substrate.
    // If the GPU flag is turned off, then personality does not exist.
    // assert(device_personality == DevicePersonality::CPU);
    ValTy extracted_vec;
    extracted_vec.reserve(gnn_matrix_to_sync_column_length_);
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // XXX memcpy
      extracted_vec.emplace_back(
          gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i]);
    }
    // move constructor should kick in here to avoid return copy
    return extracted_vec;
  }

  //! return a vector of floats to sync
  static void ExtractDirect(uint32_t node_id,
                            typename ValTy::value_type* to_write) {
    std::memcpy(
        to_write,
        (char*)&(
            gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_]),
        gnn_matrix_to_sync_column_length_ * sizeof(typename ValTy::value_type));
  }

  //! reduction is addition in this case; add received vector to
  //! own vector
  static bool reduce(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // XXX vectorized add
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] +=
          y[i];
    }
    return true;
  }

  static bool reduce(uint32_t node_id, char&, const ValTy::value_type* y) {
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // XXX vectorized add
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] +=
          y[i];
    }
    return true;
  }

  //! No-op: readAny = overwritten anyways
  static void reset(uint32_t, char&) {}
  // Reset is here in case anyone wants to bring it back
  // static void reset(uint32_t node_id, char&) {
  //  for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
  //    gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] =
  //    0;
  //  }
  //}

  //! element wise set
  static void setVal(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] =
          y[i];
    }
  }

  static void setVal(uint32_t node_id, char&, const ValTy::value_type* y) {
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

struct GNNSampleSumAggregate {
  using ValTy = galois::gstl::Vector<GNNFloat>;

  static size_t FeatVecSize() { return gnn_matrix_to_sync_column_length_; }

  //! return a vector of floats to sync
  static ValTy extract(uint32_t node_id, char&) {
    // It should be a CPU synchronizing substrate.
    // If the GPU flag is turned off, then personality does not exist.
    // assert(device_personality == DevicePersonality::CPU);
    // ValTy extracted_vec(gnn_matrix_to_sync_column_length_);
    ValTy extracted_vec;
    extracted_vec.reserve(gnn_matrix_to_sync_column_length_);
    if ((*gnn_lid_to_sid_pointer_)[node_id] ==
        std::numeric_limits<uint32_t>::max()) {
      // need to have correct size because serializer will expect
      // it to be of a certain length
      extracted_vec.resize(gnn_matrix_to_sync_column_length_, 0);
      return extracted_vec;
    }

    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // XXX memcpy
      extracted_vec.emplace_back(
          gnn_matrix_to_sync_[(*gnn_lid_to_sid_pointer_)[node_id] *
                                  gnn_matrix_to_sync_column_length_ +
                              i]);
    }
    // move constructor should kick in here to avoid return copy
    return extracted_vec;
  }

  static void ExtractDirect(uint32_t node_id,
                            typename ValTy::value_type* to_write) {
    if ((*gnn_lid_to_sid_pointer_)[node_id] ==
        std::numeric_limits<uint32_t>::max()) {
      return;
    }
    std::memcpy(
        to_write,
        (char*)&(gnn_matrix_to_sync_[(*gnn_lid_to_sid_pointer_)[node_id] *
                                     gnn_matrix_to_sync_column_length_]),
        gnn_matrix_to_sync_column_length_ * sizeof(typename ValTy::value_type));
  }

  //! reduction is addition in this case; add received vector to
  //! own vector
  static bool reduce(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    if ((*gnn_lid_to_sid_pointer_)[node_id] ==
        std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    assert((*gnn_lid_to_sid_pointer_)[node_id] < subgraph_size_);

    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // galois::gPrint("write ", (*gnn_lid_to_sid_pointer_)[node_id] *
      //                        gnn_matrix_to_sync_column_length_ + i, "\n");
      gnn_matrix_to_sync_[(*gnn_lid_to_sid_pointer_)[node_id] *
                              gnn_matrix_to_sync_column_length_ +
                          i] += y[i];
    }
    return true;
  }

  static bool reduce(uint32_t node_id, char&, ValTy::value_type* y) {
    if ((*gnn_lid_to_sid_pointer_)[node_id] ==
        std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    assert((*gnn_lid_to_sid_pointer_)[node_id] < subgraph_size_);

    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // galois::gPrint(galois::runtime::getSystemNetworkInterface().ID,  "]
      // nodeid ", node_id, " sid ",  (*gnn_lid_to_sid_pointer_)[node_id],
      //               " write ", (*gnn_lid_to_sid_pointer_)[node_id] *
      //                        gnn_matrix_to_sync_column_length_ + i, "\n");
      gnn_matrix_to_sync_[(*gnn_lid_to_sid_pointer_)[node_id] *
                              gnn_matrix_to_sync_column_length_ +
                          i] += y[i];
    }
    return true;
  }

  //! No-op: readAny = overwritten anyways
  static void reset(uint32_t, char&) {}

  //! element wise set
  static void setVal(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    if ((*gnn_lid_to_sid_pointer_)[node_id] ==
        std::numeric_limits<uint32_t>::max()) {
      return;
    }
    assert((*gnn_lid_to_sid_pointer_)[node_id] < subgraph_size_);

    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      gnn_matrix_to_sync_[(*gnn_lid_to_sid_pointer_)[node_id] *
                              gnn_matrix_to_sync_column_length_ +
                          i] = y[i];
    }
  }
  static void setVal(uint32_t node_id, char&, ValTy::value_type* y) {
    if ((*gnn_lid_to_sid_pointer_)[node_id] ==
        std::numeric_limits<uint32_t>::max()) {
      return;
    }

    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // galois::gPrint(galois::runtime::getSystemNetworkInterface().ID,  "]
      // broadxast nodeid ", node_id, " sid ",
      // (*gnn_lid_to_sid_pointer_)[node_id],
      //               " write ", (*gnn_lid_to_sid_pointer_)[node_id] *
      //                        gnn_matrix_to_sync_column_length_ + i, "\n");
      gnn_matrix_to_sync_[(*gnn_lid_to_sid_pointer_)[node_id] *
                              gnn_matrix_to_sync_column_length_ +
                          i] = y[i];
    }
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

#ifdef GALOIS_ENABLE_GPU
extern struct CUDA_Context* cuda_ctx;
GALOIS_SYNC_STRUCTURE_GNN_LAYER(layer_input, cuda_ctx_for_sync,
                                gnn_matrix_to_sync_column_length_,
                                layer_number_to_sync);
GALOIS_SYNC_STRUCTURE_GNN_LAYER(layer_output, cuda_ctx_for_sync,
                                gnn_matrix_to_sync_column_length_,
                                layer_number_to_sync);
#endif
GALOIS_SYNC_STRUCTURE_BITSET(graph_aggregate);

} // namespace graphs
} // namespace galois

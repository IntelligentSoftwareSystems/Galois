// defined in GNNGraph.cpp; set in order to control which matrix
// gets synchronized
#include "galois/GNNTypes.h"
#ifdef GALOIS_ENABLE_GPU
#include "galois/GNNCudaContextHostDecls.h"
#endif

namespace galois {
namespace graphs {

extern std::vector<char>* sampled_nodes_;
extern GNNFloat* gnn_matrix_to_sync_;
extern size_t gnn_matrix_to_sync_column_length_;
extern galois::DynamicBitSet bitset_graph_aggregate;
extern galois::LargeArray<uint32_t>* gnn_lid_to_sid_pointer_;
extern galois::DynamicBitSet bitset_sample_flag_;
extern size_t subgraph_size_;
extern size_t num_active_layer_rows_;
#ifdef GALOIS_ENABLE_GPU
extern struct CUDA_Context* cuda_ctx_for_sync;
extern unsigned layer_number_to_sync;
#endif

// NodeTy is always a node data type of a "graph" type.
// This type is used by GluonSubstrate to reset a value.
// ValTy is either a node data type of a graph or the ones
// that are stored in separate objects.
template <typename NTy>
struct SampleFlagSync {
  using NodeTy = NTy;
  using ValTy  = char;

  //! return a vector of floats to sync
  static ValTy extract(uint32_t lid, NodeTy&) { return (*sampled_nodes_)[lid]; }

  static bool reduce(uint32_t lid, NodeTy&, ValTy y) {
    if (y) {
      (*sampled_nodes_)[lid] = y;
      assert((*sampled_nodes_)[lid] == 1);
      return true;
    } else {
      return false;
    }
  }

  //! No-op: readAny = overwritten anyways
  static void reset(uint32_t, NodeTy&) {}

  //! element wise set
  static void setVal(uint32_t lid, NodeTy&, ValTy y) {
    (*sampled_nodes_)[lid] = y;
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

struct SampleFlagBitset {
  static constexpr bool is_vector_bitset() { return false; }
  static constexpr bool is_valid() { return true; }
  static galois::DynamicBitSet& get() { return bitset_sample_flag_; }
  static void reset_range(size_t begin, size_t end) {
    bitset_sample_flag_.reset(begin, end);
  }
};

template <typename NTy>
struct GNNSumAggregate {
  using ValTy  = galois::gstl::Vector<GNNFloat>;
  using NodeTy = NTy;

  static size_t FeatVecSize() { return gnn_matrix_to_sync_column_length_; }

  //! return a vector of floats to sync
  static ValTy extract(uint32_t node_id, NodeTy&) {
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
  static bool reduce(uint32_t node_id, NodeTy&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // XXX vectorized add
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] +=
          y[i];
    }
    return true;
  }

  static bool reduce(uint32_t node_id, NodeTy&, const ValTy::value_type* y) {
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      // XXX vectorized add
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] +=
          y[i];
    }
    return true;
  }

  //! No-op: readAny = overwritten anyways
  static void reset(uint32_t, NodeTy&) {}
  // Reset is here in case anyone wants to bring it back
  // static void reset(uint32_t node_id, char&) {
  //  for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
  //    gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] =
  //    0;
  //  }
  //}

  //! element wise set
  static void setVal(uint32_t node_id, NodeTy&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] =
          y[i];
    }
  }

  static void setVal(uint32_t node_id, NodeTy&, const ValTy::value_type* y) {
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

template <typename NTy>
struct GNNSampleSumAggregate {
  using ValTy  = galois::gstl::Vector<GNNFloat>;
  using NodeTy = NTy;

  static size_t FeatVecSize() { return gnn_matrix_to_sync_column_length_; }

  //! return a vector of floats to sync
  static ValTy extract(uint32_t node_id, NodeTy&) {
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
  static bool reduce(uint32_t node_id, NodeTy&, ValTy y) {
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

  static bool reduce(uint32_t node_id, NodeTy&, ValTy::value_type* y) {
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
  static void reset(uint32_t, NodeTy&) {}

  // version where you have a vector object
  static void setVal(uint32_t node_id, NodeTy&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    uint32_t converted_sid = (*gnn_lid_to_sid_pointer_)[node_id];
    if (converted_sid >= num_active_layer_rows_ ||
        converted_sid == std::numeric_limits<uint32_t>::max()) {
      return;
    }
    assert(converted_sid < subgraph_size_);

    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      gnn_matrix_to_sync_[converted_sid * gnn_matrix_to_sync_column_length_ +
                          i] = y[i];
    }
  }

  // version where you have a pointer only (more efficient because this
  // version is for reading directly from the recv buffer)
  static void setVal(uint32_t node_id, NodeTy&, ValTy::value_type* y) {
    uint32_t converted_sid = (*gnn_lid_to_sid_pointer_)[node_id];
    if (converted_sid >= num_active_layer_rows_ ||
        converted_sid == std::numeric_limits<uint32_t>::max()) {
      return;
    }

    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
      gnn_matrix_to_sync_[converted_sid * gnn_matrix_to_sync_column_length_ +
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

template <typename NTy>
struct SHADGNNSumAggregate {
  using ValTy  = galois::gstl::Vector<GNNFloat>;
  using NodeTy = NTy;

  static size_t FeatVecSize() { return gnn_matrix_to_sync_column_length_ / 2; }

  //! return a vector of floats to sync
  static ValTy extract(uint32_t node_id, NodeTy&) {
    // It should be a CPU synchronizing substrate.
    // If the GPU flag is turned off, then personality does not exist.
    // assert(device_personality == DevicePersonality::CPU);

    // It should extract the last half of features of the adjacent neighbors
    // (So, source of feature aggregation).
    ValTy extracted_vec;
    extracted_vec.reserve(gnn_matrix_to_sync_column_length_ / 2);
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_ / 2; i++) {
      // XXX memcpy
      extracted_vec.emplace_back(
          gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i +
                              gnn_matrix_to_sync_column_length_ / 2]);
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
            gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ +
                                gnn_matrix_to_sync_column_length_ / 2]),
        (gnn_matrix_to_sync_column_length_ / 2) *
            sizeof(typename ValTy::value_type));
  }

  //! reduction is addition in this case; add received vector to
  //! own vector
  static bool reduce(uint32_t node_id, char&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_ / 2);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_ / 2; i++) {
      // XXX vectorized add
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i +
                          gnn_matrix_to_sync_column_length_ / 2] += y[i];
    }
    return true;
  }

  static bool reduce(uint32_t node_id, NodeTy&, const ValTy::value_type* y) {
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_ / 2; i++) {
      // XXX vectorized add
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i +
                          gnn_matrix_to_sync_column_length_ / 2] += y[i];
    }
    return true;
  }

  //! No-op: readAny = overwritten anyways
  static void reset(uint32_t, NodeTy&) {}
  // Reset is here in case anyone wants to bring it back
  // static void reset(uint32_t node_id, char&) {
  //  for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_; i++) {
  //    gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i] =
  //    0;
  //  }
  //}

  //! element wise set
  static void setVal(uint32_t node_id, NodeTy&, ValTy y) {
    assert(y.size() == gnn_matrix_to_sync_column_length_);
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_ / 2; i++) {
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i +
                          gnn_matrix_to_sync_column_length_ / 2] = y[i];
    }
  }

  static void setVal(uint32_t node_id, NodeTy&, const ValTy::value_type* y) {
    // loop and do addition
    for (unsigned i = 0; i < gnn_matrix_to_sync_column_length_ / 2; i++) {
      gnn_matrix_to_sync_[node_id * gnn_matrix_to_sync_column_length_ + i +
                          gnn_matrix_to_sync_column_length_ / 2] = y[i];
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

// XXX include net interface if necessary
#include "galois/graphs/GNNGraph.h"

namespace galois {
namespace graphs {

std::vector<char>* sampled_nodes_ = nullptr;
// Sync structure variables; global to get around sync structure
// limitations at the moment
GNNFloat* gnn_matrix_to_sync_            = nullptr;
size_t gnn_matrix_to_sync_column_length_ = 0;
size_t subgraph_size_                    = 0;
//! For synchronization of graph aggregations
galois::DynamicBitSet bitset_graph_aggregate;
galois::LargeArray<uint32_t>* gnn_lid_to_sid_pointer_ = nullptr;
size_t num_active_layer_rows_                         = 0;
//! It specifies offset for feature aggregation
size_t feature_aggregation_offset_ = 0;
uint32_t* gnn_degree_vec_1_;
uint32_t* gnn_degree_vec_2_;

galois::DynamicBitSet bitset_sample_flag_;

//! For synchronization of sampled degrees
galois::DynamicBitSet bitset_sampled_degrees_;
std::vector<galois::LargeArray<uint32_t>>* gnn_sampled_out_degrees_;

#ifdef GALOIS_ENABLE_GPU
struct CUDA_Context* cuda_ctx_for_sync;
struct CUDA_Context* cuda_ctx;
unsigned layer_number_to_sync;
#endif

}; // namespace graphs
}; // namespace galois

#include "galois/GNNMath.cuh"
#include "galois/layers/GraphConvolutionalLayer.cuh"

galois::GCNGPUAllocations::~GCNGPUAllocations() {
  GALOIS_LOG_VERBOSE("Freeing GCN layer allocations");
  CUDA_FREE(in_temp_1_);
  CUDA_FREE(in_temp_2_);
  CUDA_FREE(out_temp_);
}

void galois::GCNGPUAllocations::Allocate(size_t input_elements,
                                         size_t output_elements) {
  CUDA_CHECK(
      cudaMalloc((void**)(&in_temp_1_), input_elements * sizeof(GNNFloat)));
  CUDA_CHECK(
      cudaMalloc((void**)(&in_temp_2_), input_elements * sizeof(GNNFloat)));
  CUDA_CHECK(
      cudaMalloc((void**)(&out_temp_), output_elements * sizeof(GNNFloat)));
}

namespace {
// GPU side aggregation call: no matrix multiply, just regular dst accesses
__global__ void AggregateAllKernel(unsigned num_nodes, size_t column_length,
                                   const int* edge_index,
                                   const int* edge_destination,
                                   const galois::GNNFloat* node_embeddings,
                                   galois::GNNFloat* aggregate_output) {
  const unsigned thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const unsigned thread_lane =
      threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const unsigned warp_id = thread_id / WARP_SIZE; // global warp index
  const unsigned warp_lane =
      threadIdx.x / WARP_SIZE; // warp index within the CTA
  const unsigned num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  // each warp gets a source: this var holds the first/last edge worked on by
  // that warp
  __shared__ int edge_begin_end[BLOCK_SIZE / WARP_SIZE][2];

  // each warp works on a source: threads in warp split the feature
  for (int src = warp_id; src < static_cast<int>(num_nodes); src += num_warps) {
    if (thread_lane < 2) {
      edge_begin_end[warp_lane][thread_lane] = edge_index[src + thread_lane];
    }
    // essentially what this is doing is making 2 of the threads set edge
    // begin/end; all threads wait for sync
    __syncthreads();

    const int row_begin     = edge_begin_end[warp_lane][0];
    const int row_end       = edge_begin_end[warp_lane][1];
    unsigned base_src_index = src * column_length;

    for (int offset = row_begin; offset < row_end; offset++) {
      int dst                 = edge_destination[offset];
      unsigned base_dst_index = dst * column_length;

      // NOTE: this is where warp diverges
      // the feature aggregation is split among thread in a warp
      for (int i = 0; i < column_length; i += WARP_SIZE) {
        if ((thread_lane + i) < column_length) {
          aggregate_output[base_src_index + thread_lane + i] +=
              node_embeddings[base_dst_index + thread_lane + i];
        }
      }
    }
  }
}

} // namespace

void galois::GCNGPUAllocations::AggregateAllGPU(
    const graphs::GNNGraphGPUAllocations& gpu_graph, size_t num_nodes,
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output) {
  AggregateAllKernel<<<(num_nodes - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(
      num_nodes, column_length, gpu_graph.edge_index(),
      gpu_graph.edge_destinations(), node_embeddings, aggregate_output);
  CUDA_TEST("GPU aggregate all failure");
}

void galois::GCNGPUAllocations::UpdateEmbeddingsGPU(
    size_t num_nodes, size_t input_columns, size_t output_columns,
    const GNNFloat* node_embeddings, const GNNFloat* layer_weights,
    GNNFloat* output) {
  CBlasSGEMMGPU(CUBLAS_OP_N, CUBLAS_OP_N, num_nodes, input_columns,
                output_columns, node_embeddings, layer_weights, output);
}

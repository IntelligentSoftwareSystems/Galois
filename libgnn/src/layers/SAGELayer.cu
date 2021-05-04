#include "gg.h"
#include "ggcuda.h"
#include "galois/cuda/DynamicBitset.h"
#include "galois/GNNMath.cuh"
#include "galois/layers/SAGELayer.cuh"

extern Shared<DynamicBitset> cuda_bitset_graph_aggregate;

galois::SAGEGPUAllocations::~SAGEGPUAllocations() {
  GALOIS_LOG_VERBOSE("Freeing SAGE layer allocations");
  CUDA_FREE(in_temp_1_);
  CUDA_FREE(in_temp_2_);
  CUDA_FREE(out_temp_);
  CUDA_FREE(layer_weights_2_);
  CUDA_FREE(layer_weight_gradients_2_);
}

void galois::SAGEGPUAllocations::AllocateWeight2(const size_t size) {
  CUDA_CHECK(cudaMalloc((void**)(&layer_weights_2_), size * sizeof(GNNFloat)));
}

void galois::SAGEGPUAllocations::AllocateWeightGradient2(const size_t size) {
  CUDA_CHECK(cudaMalloc((void**)(&layer_weight_gradients_2_),
                        size * sizeof(GNNFloat)));
}

void galois::SAGEGPUAllocations::AllocateInTemp1(const size_t size) {
  CUDA_CHECK(cudaMalloc((void**)(&in_temp_1_), size * sizeof(GNNFloat)));
}

void galois::SAGEGPUAllocations::AllocateInTemp2(const size_t size) {
  CUDA_CHECK(cudaMalloc((void**)(&in_temp_2_), size * sizeof(GNNFloat)));
}

void galois::SAGEGPUAllocations::AllocateOutTemp(const size_t size) {
  CUDA_CHECK(cudaMalloc((void**)(&out_temp_), size * sizeof(GNNFloat)));
}

namespace {
// GPU side aggregation call: no matrix multiply, just regular dst accesses
__global__ void AggregateAllKernel(
    unsigned num_nodes, size_t column_length, const int* edge_index,
    const int* edge_destination, const uint32_t* degree_for_norm,
    const galois::GNNFloat* node_embeddings, galois::GNNFloat* aggregate_output,
    DynamicBitset* cuda_bitset_graph_aggregate, bool is_backward) {
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
    galois::GNNFloat norm_to_use = 0.0;

    if (degree_for_norm != nullptr && !is_backward) {
      norm_to_use = (degree_for_norm[src]) ? (1.0 / degree_for_norm[src]) : 0.0;
    }

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
      cuda_bitset_graph_aggregate->set(src);
      int dst                 = edge_destination[offset];
      unsigned base_dst_index = dst * column_length;

      if (degree_for_norm != nullptr && is_backward) {
        norm_to_use =
            (degree_for_norm[dst]) ? (1.0 / degree_for_norm[dst]) : 0.0;
      }

      // NOTE: this is where warp diverges
      // the feature aggregation is split among thread in a warp
      for (int i = 0; i < column_length; i += WARP_SIZE) {
        if ((thread_lane + i) < column_length) {
          if (degree_for_norm != nullptr) {
            aggregate_output[base_src_index + thread_lane + i] +=
                node_embeddings[base_dst_index + thread_lane + i] * norm_to_use;
          } else {
            aggregate_output[base_src_index + thread_lane + i] +=
                node_embeddings[base_dst_index + thread_lane + i];
          }
        }
      }
    }
  }
}

} // namespace

// TODO(lhc) Will need to iterate over in-edges if is_backward is on
void galois::SAGEGPUAllocations::AggregateAllGPU(
    const graphs::GNNGraphGPUAllocations& gpu_graph, size_t num_nodes,
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output, bool use_norm, bool is_backward) {
  // num_nodes should be greater than 0 to avoid negative number of thread
  if (num_nodes == 0) {
    return;
  }

  CUDA_CHECK(cudaMemset(aggregate_output, 0,
                        num_nodes * column_length * sizeof(GNNFloat)));
  if (use_norm) {
    uint32_t* degree_for_norm{nullptr};
    // TODO(lhc) will be added for sampling
    // if (use_subgraph_) {
    //} else {
    degree_for_norm = gpu_graph.get_global_degrees();
    //}
    AggregateAllKernel<<<(num_nodes - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(
        num_nodes, column_length, gpu_graph.edge_index(),
        gpu_graph.edge_destinations(), degree_for_norm, node_embeddings,
        aggregate_output, cuda_bitset_graph_aggregate.gpu_wr_ptr(),
        is_backward);
  } else {
    AggregateAllKernel<<<(num_nodes - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(
        num_nodes, column_length, gpu_graph.edge_index(),
        gpu_graph.edge_destinations(), nullptr, node_embeddings,
        aggregate_output, cuda_bitset_graph_aggregate.gpu_wr_ptr(),
        is_backward);
  }
  CUDA_TEST("GPU aggregate all failure");
}

void galois::SAGEGPUAllocations::UpdateEmbeddingsGPU(
    size_t num_nodes, size_t input_columns, size_t output_columns,
    const GNNFloat* node_embeddings, const GNNFloat* layer_weights,
    GNNFloat* output) {
  CBlasSGEMMGPU(CUBLAS_OP_N, CUBLAS_OP_N, num_nodes, input_columns,
                output_columns, node_embeddings, layer_weights, output);
}

void galois::SAGEGPUAllocations::UpdateEmbeddingsDerivativeGPU(
    size_t num_nodes, size_t input_columns, size_t output_columns,
    const GNNFloat* gradients, const GNNFloat* layer_weights,
    GNNFloat* output) {
  // note output clumns/input columns are flipped due to transpose of the
  // layer weights
  CBlasSGEMMGPU(CUBLAS_OP_N, CUBLAS_OP_T, num_nodes, output_columns,
                input_columns, gradients, layer_weights, output);
}

void galois::SAGEGPUAllocations::GetWeightGradientsGPU(
    size_t num_nodes, size_t input_columns, size_t output_columns,
    const GNNFloat* prev_input, const GNNFloat* gradients, GNNFloat* output) {
  CBlasSGEMMGPU(CUBLAS_OP_T, CUBLAS_OP_N, input_columns, num_nodes,
                output_columns, prev_input, gradients, output);
}

void galois::SAGEGPUAllocations::SelfFeatureUpdateEmbeddingsGPU(
    size_t input_rows, size_t input_columns, size_t output_columns,
    const GNNFloat* node_embeddings, GNNFloat* output) {
  CBlasSGEMMGPU(CUBLAS_OP_N, CUBLAS_OP_N, input_rows, input_columns,
                output_columns, node_embeddings, layer_weights_2_, output,
                true);
}

void galois::SAGEGPUAllocations::SelfFeatureUpdateEmbeddingsDerivativeGPU(
    size_t input_rows, size_t output_columns, size_t input_columns,
    const GNNFloat* gradients, GNNFloat* output) {
  CBlasSGEMMGPU(CUBLAS_OP_N, CUBLAS_OP_T, input_rows, output_columns,
                input_columns, gradients, layer_weights_2_, output, true);
}

void galois::SAGEGPUAllocations::UpdateWeight2DerivativeGPU(
    size_t input_columns, size_t input_rows, size_t output_columns,
    const GNNFloat* prev_layer_inputs, const GNNFloat* input_gradients,
    GNNFloat* output) {
  CBlasSGEMMGPU(CUBLAS_OP_T, CUBLAS_OP_N, input_columns, input_rows,
                output_columns, prev_layer_inputs, input_gradients, output);
}

void galois::SAGEGPUAllocations::CopyToWeights2(
    const std::vector<GNNFloat>& cpu_layer_weights) {
  CUDA_CHECK(cudaMemcpy(layer_weights_2_, cpu_layer_weights.data(),
                        cpu_layer_weights.size() * sizeof(GNNFloat),
                        cudaMemcpyHostToDevice));
}

void galois::SAGEGPUAllocations::CopyToWeight2Gradients(
    const std::vector<GNNFloat>& cpu_gradients) {
  CUDA_CHECK(cudaMemcpy(layer_weight_gradients_2_, cpu_gradients.data(),
                        cpu_gradients.size() * sizeof(GNNFloat),
                        cudaMemcpyHostToDevice));
}

void galois::SAGEGPUAllocations::CopyWeight2GradientsToCPU(
    std::vector<GNNFloat>* cpu_gradients) {
  CUDA_CHECK(cudaMemcpy(cpu_gradients->data(), layer_weight_gradients_2_,
                        cpu_gradients->size() * sizeof(GNNFloat),
                        cudaMemcpyDeviceToHost));
}

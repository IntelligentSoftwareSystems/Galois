#include "gg.h"
#include "ggcuda.h"

#include "galois/cuda/DynamicBitset.h"

#include "galois/CUDAUtil.h"
#include "galois/graphs/GNNGraph.cuh"
#include "sharedptr.h"

Shared<DynamicBitset> cuda_bitset_graph_aggregate;

galois::graphs::GNNGraphGPUAllocations::~GNNGraphGPUAllocations() {
  GALOIS_LOG_VERBOSE("Freeing GPU graph allocations");
  CUDA_FREE(num_features_);
  CUDA_FREE(feature_length_);
  CUDA_FREE(num_edges_);
  CUDA_FREE(edge_index_);
  CUDA_FREE(edge_destinations_);
  CUDA_FREE(feature_vector_);
  CUDA_FREE(ground_truth_);
  CUDA_FREE(local_training_mask_);
  CUDA_FREE(local_validation_mask_);
  CUDA_FREE(local_testing_mask_);
  CUDA_FREE(global_degrees_);
  CUDA_FREE(global_train_degrees_);
}

void galois::graphs::GNNGraphGPUAllocations::SetGraphTopology(
    const std::vector<int>& edge_index, const std::vector<int>& edge_dests) {
  // num edges variable
  CUDA_CHECK(cudaMalloc((void**)(&num_edges_), sizeof(unsigned)));
  unsigned num_edges = edge_dests.size();
  CUDA_CHECK(cudaMemcpy(num_edges_, &num_edges, sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  // topology; assumes caller already setup vectors accordingly
  CUDA_CHECK(
      cudaMalloc((void**)(&edge_index_), edge_index.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)(&edge_destinations_),
                        edge_dests.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(edge_index_, edge_index.data(),
                        edge_index.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(edge_destinations_, edge_dests.data(),
                        edge_dests.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
}

void galois::graphs::GNNGraphGPUAllocations::SetFeatures(
    const std::vector<GNNFeature>& features, unsigned num_features) {
  // feature count & length
  CUDA_CHECK(cudaMalloc((void**)(&num_features_), sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc((void**)(&feature_length_), sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(num_features_, &num_features, sizeof(unsigned),
                        cudaMemcpyHostToDevice));
  unsigned feature_length = features.size() / num_features;
  CUDA_CHECK(cudaMemcpy(feature_length_, &feature_length, sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  // features themselves
  CUDA_CHECK(cudaMalloc((void**)(&feature_vector_),
                        features.size() * sizeof(GNNFeature)));
  CUDA_CHECK(cudaMemcpy(feature_vector_, features.data(),
                        features.size() * sizeof(GNNFeature),
                        cudaMemcpyHostToDevice));
}

void galois::graphs::GNNGraphGPUAllocations::SetLabels(
    const std::vector<GNNLabel>& ground_truth) {
  CUDA_CHECK(cudaMalloc((void**)(&ground_truth_),
                        ground_truth.size() * sizeof(GNNLabel)));
  CUDA_CHECK(cudaMemcpy(ground_truth_, ground_truth.data(),
                        ground_truth.size() * sizeof(GNNLabel),
                        cudaMemcpyHostToDevice));
}

void galois::graphs::GNNGraphGPUAllocations::SetMasks(
    const std::vector<char>& train, const std::vector<char>& val,
    const std::vector<char>& test) {
  CUDA_CHECK(
      cudaMalloc((void**)(&local_training_mask_), train.size() * sizeof(char)));
  CUDA_CHECK(cudaMemcpy(local_training_mask_, train.data(),
                        train.size() * sizeof(char), cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMalloc((void**)(&local_validation_mask_), val.size() * sizeof(char)));
  CUDA_CHECK(cudaMemcpy(local_validation_mask_, val.data(),
                        val.size() * sizeof(char), cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMalloc((void**)(&local_testing_mask_), test.size() * sizeof(char)));
  CUDA_CHECK(cudaMemcpy(local_testing_mask_, test.data(),
                        test.size() * sizeof(char), cudaMemcpyHostToDevice));
}

void galois::graphs::GNNGraphGPUAllocations::InitNormFactor(size_t num_nodes) {
  GALOIS_LOG_ASSERT(global_degrees_ == nullptr);
  GALOIS_LOG_ASSERT(global_train_degrees_ == nullptr);

  CUDA_CHECK(
      cudaMalloc((void**)(&global_degrees_), sizeof(uint32_t) * num_nodes));
  CUDA_CHECK(cudaMalloc((void**)(&global_train_degrees_),
                        sizeof(uint32_t) * num_nodes));
  global_degree_size_       = num_nodes;
  global_train_degree_size_ = num_nodes;
}

#if 0 // TODO(lhc) will be added
__global__ void CalculateFullNormFactorGPU() {
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
    __syncthreads();

    const int edge_begin = edge_begin_end[warp_lane][0];
    const int edge_end   = edge_begin_end[warp_lane][1];
    for (int offest = edge_begin; offset < edge_end; offset++) {

    }
  }
}

void galois::graphs::GNNGraphGPUAllocations::CalculateFullNormFactor() {

}
#endif

void galois::graphs::GNNGraphGPUAllocations::SetGlobalDegrees(
    const std::vector<uint32_t> global_degrees) {
  if (global_degree_size_ < global_degrees.size()) {
    if (global_degree_size_ > 0) {
      CUDA_CHECK(cudaFree(global_degrees_));
    }
    CUDA_CHECK(cudaMalloc((void**)(&global_degrees_),
                          global_degrees.size() * sizeof(uint32_t)));
    global_degree_size_ = global_degrees.size();
  }

  CUDA_CHECK(cudaMemcpy(global_degrees_, global_degrees.data(),
                        global_degrees.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
}

void galois::graphs::GNNGraphGPUAllocations::SetGlobalTrainDegrees(
    const std::vector<uint32_t> global_train_degrees) {
  if (global_train_degree_size_ < global_train_degrees.size()) {
    if (global_train_degree_size_ > 0) {
      CUDA_CHECK(cudaFree(global_train_degrees_));
    }
    CUDA_CHECK(cudaMalloc((void**)(&global_train_degrees_),
                          global_train_degrees.size() * sizeof(uint32_t)));
    global_train_degree_size_ = global_train_degrees.size();
  }

  CUDA_CHECK(cudaMemcpy(global_train_degrees_, global_train_degrees.data(),
                        global_train_degrees.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
}

void galois::graphs::GNNGraphGPUAllocations::AllocAggregateBitset(size_t size) {
  cuda_bitset_graph_aggregate.alloc(1);
  cuda_bitset_graph_aggregate.cpu_wr_ptr()->alloc(size);
}

void galois::graphs::GNNGraphGPUAllocations::CopyToCPU(
    const PointerWithSize<GNNFloat>& input) {
  GNNFloat* cpu_input = (GNNFloat*)malloc(sizeof(GNNFloat) * input.size());
  cudaMemcpy(cpu_input, input.data(), sizeof(GNNFloat) * input.size(),
             cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < input.size(); i++)
    fprintf(stdout, "** %lu is %f\n", i, cpu_input[i]);
}

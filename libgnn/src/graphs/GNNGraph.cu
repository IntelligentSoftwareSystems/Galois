#include "galois/CUDAUtil.h"
#include "galois/graphs/GNNGraph.cuh"

galois::graphs::GNNGraphGPUAllocations::~GNNGraphGPUAllocations() {
  GALOIS_LOG_VERBOSE("Freeing GPU graph allocations");
  CUDA_FREE(num_features_);
  CUDA_FREE(feature_length_);
  CUDA_FREE(num_edges_);
  CUDA_FREE(edge_index_);
  CUDA_FREE(edge_destinations_);
  CUDA_FREE(feature_vector_);
  CUDA_FREE(ground_truth_);
  CUDA_FREE(norm_factors_);
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

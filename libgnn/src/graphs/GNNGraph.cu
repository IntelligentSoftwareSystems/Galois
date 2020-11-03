#include "galois/CUDAUtil.h"
#include "galois/graphs/GNNGraph.cuh"

void galois::graphs::GNNGraphGPUAllocations::SetFeatures(
    const std::vector<GNNFeature>& features) {
  CUDA_CHECK(cudaMalloc((void**)(&feature_vector_),
                        features.size() * sizeof(GNNFeature)));
  CUDA_CHECK(cudaMemcpy(feature_vector_, features.data(),
                        features.size() * sizeof(GNNFeature),
                        cudaMemcpyHostToDevice));
}

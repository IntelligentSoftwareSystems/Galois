#include "galois/CUDAUtil.h"
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

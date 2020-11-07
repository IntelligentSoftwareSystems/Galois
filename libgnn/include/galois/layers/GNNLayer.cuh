#pragma once
#include "galois/GNNTypes.h"

namespace galois {

//! Holds pointers to GNN layer weights/gradient on GPU
class GNNLayerGPUAllocations {
public:
  //! CUDA frees all allocated memory (i.e. non-nullptr)
  ~GNNLayerGPUAllocations();
  //! Initializes forward and backward output matrices of this layer on GPU
  void InitInOutMemory(size_t forward_size, size_t backward_size);
  //! Initializes memory for weight and weight gradients on GPU
  void InitWeightMemory(size_t num_weights);
  //! Copy provided data in vector to GPU weights
  void CopyToWeights(const std::vector<GNNFloat>& cpu_layer_weights);

private:
  size_t* num_weights_{nullptr};
  GNNFloat* forward_output_matrix_{nullptr};
  GNNFloat* backward_output_matrix_{nullptr};
  GNNFloat* layer_weights_{nullptr};
  GNNFloat* layer_weight_gradients_{nullptr};
};

} // namespace galois

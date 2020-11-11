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
  //! Copy GPU forward output to the provided vector (assumes vector is already
  //! correct size)
  void CopyForwardOutputToCPU(std::vector<GNNFloat>* cpu_forward_output);
  //! Prints forward output matrix on gpu
  void PrintForwardOutput(size_t num);

  GNNFloat* forward_output() { return forward_output_matrix_; }
  GNNFloat* backward_output() { return backward_output_matrix_; }
  GNNFloat* layer_weights() { return layer_weights_; }
  GNNFloat* layer_weight_gradients() { return layer_weight_gradients_; }

private:
  size_t* num_weights_{nullptr};
  GNNFloat* forward_output_matrix_{nullptr};
  GNNFloat* backward_output_matrix_{nullptr};
  GNNFloat* layer_weights_{nullptr};
  GNNFloat* layer_weight_gradients_{nullptr};
};

} // namespace galois

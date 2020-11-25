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
  //! Initializes memory for dropout
  void InitDropoutMemory(size_t dropout_size);
  //! Copy provided data in vector to GPU weights
  void CopyToWeights(const std::vector<GNNFloat>& cpu_layer_weights);
  //! Copy GPU forward output to the provided vector (assumes vector is already
  //! correct size)
  void CopyForwardOutputToCPU(std::vector<GNNFloat>* cpu_forward_output);
  //! Copy GPU backward output to the provided vector (assumes vector is already
  //! correct size)
  void CopyBackwardOutputToCPU(std::vector<GNNFloat>* cpu_backward_output);
  //! Copy GPU weight gradients to the provided vector (assumes vector is
  //! already correct size)
  void CopyWeightGradientsToCPU(std::vector<GNNFloat>* cpu_gradients);

  //! Prints forward output matrix on gpu
  void PrintForwardOutput(size_t num);

  //! Does dropout on the GPU; saves non-dropped weights to output
  void DoDropoutGPU(const PointerWithSize<GNNFloat> input_to_dropout,
                    PointerWithSize<GNNFloat> output, float dropout_rate);

  //! Helper function: give a vector which is copied over to the GPU (new
  //! memory is allocated as necessary)
  GNNFloat* Allocate(const std::vector<GNNFloat>& v);

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
  GNNFloat* rng_results_{nullptr};
  char* dropout_mask_{nullptr};
};

} // namespace galois

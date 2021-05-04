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
  //! Copy provided data in vector to GPU weight gradients
  void CopyToWeightGradients(const std::vector<GNNFloat>& cpu_gradients);
  //! Copy GPU forward output to the provided vector (assumes vector is already
  //! correct size)
  void CopyForwardOutputToCPU(GNNFloat* cpu_forward_output,
                              size_t forward_output_size);
  //! Copy GPU backward output to the provided vector (assumes vector is already
  //! correct size)
  void CopyBackwardOutputToCPU(GNNFloat* cpu_backward_output,
                               size_t backward_output_size);
  //! Copy GPU weight gradients to the provided vector (assumes vector is
  //! already correct size)
  void CopyWeightGradientsToCPU(std::vector<GNNFloat>* cpu_gradients);

  //! Prints forward output matrix on gpu
  void PrintForwardOutput(size_t num);

  //! Prints backward output matrix on gpu
  void PrintBackwardOutput(size_t num);

  //! Does dropout on the GPU; saves non-dropped weights to output
  void DoDropoutGPU(const PointerWithSize<GNNFloat> input_to_dropout,
                    PointerWithSize<GNNFloat> output, float dropout_rate);
  //! Does dropout derivative on the backward output matrix of the gpu
  void DoDropoutDerivativeGPU(size_t input_size, GNNFloat scale);

  //! Helper function: give a vector which is copied over to the GPU (new
  //! memory is allocated as necessary)
  GNNFloat* Allocate(const std::vector<GNNFloat>& v);

  //! Initializes vectors on GPU to 1
  void InitGPUVectorTo1(GNNFloat* vector, size_t vector_size);

  //! Apply an activation function
  void ActivationGPU(size_t num_forward_output_elements);
  //! Apply an activation function for derivative
  void ActivationDerivativeGPU(GNNFloat* gradients,
                               size_t num_gradients_elements);
  void
  ReconstructDropoutMatrixGPU(const PointerWithSize<GNNFloat> input_to_drouput,
                              PointerWithSize<GNNFloat>* output_matrix,
                              size_t num_elements, GNNFloat scale);

  void MaskNonMastersGPU(PointerWithSize<GNNFloat>* input, size_t start_node,
                         size_t end_node, size_t row_index);

  GNNFloat* forward_output() { return forward_output_matrix_; }
  GNNFloat* backward_output() { return backward_output_matrix_; }
  GNNFloat* layer_weights() { return layer_weights_; }
  GNNFloat* layer_weight_gradients() { return layer_weight_gradients_; }

  void CopyToCPU(PointerWithSize<GNNFloat>* input);
  void CopyToCPU(GNNFloat* input, size_t size);

private:
  size_t* num_weights_{nullptr};
  GNNFloat* forward_output_matrix_{nullptr};
  GNNFloat* backward_output_matrix_{nullptr};
  GNNFloat* layer_weights_{nullptr};
  GNNFloat* layer_weight_gradients_{nullptr};
  GNNFloat* rng_results_{nullptr};
  char* dropout_mask_{nullptr};
  uint8_t* activation_memo_{nullptr};
};

} // namespace galois

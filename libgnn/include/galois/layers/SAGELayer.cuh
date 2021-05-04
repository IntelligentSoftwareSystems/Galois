#pragma once
#include "galois/GNNTypes.h"
#include "galois/graphs/GNNGraph.cuh"

namespace galois {

//! Holds pointers for GPU memory for SAGE layer
class SAGEGPUAllocations {
public:
  // free memory
  ~SAGEGPUAllocations();

  // allocate the 3 temp arrays
  void AllocateInTemp1(const size_t size);
  void AllocateInTemp2(const size_t size);
  void AllocateOutTemp(const size_t size);

  GNNFloat* in_temp_1() { return in_temp_1_; }
  GNNFloat* in_temp_2() { return in_temp_2_; }
  GNNFloat* out_temp() { return out_temp_; }

  void AllocateWeight2(const size_t size);
  void AllocateWeightGradient2(const size_t size);

  GNNFloat* layer_weights_2() { return layer_weights_2_; }
  GNNFloat* layer_weight_gradients_2() { return layer_weight_gradients_2_; }

  void AggregateAllGPU(const graphs::GNNGraphGPUAllocations& gpu_graph,
                       size_t num_nodes, size_t column_length,
                       const GNNFloat* node_embeddings,
                       GNNFloat* aggregate_output, bool use_norm,
                       bool is_backward);

  void UpdateEmbeddingsGPU(size_t num_nodes, size_t input_columns,
                           size_t output_columns,
                           const GNNFloat* node_embeddings,
                           const GNNFloat* layer_weights, GNNFloat* output);
  void UpdateEmbeddingsDerivativeGPU(size_t num_nodes, size_t input_columns,
                                     size_t output_columns,
                                     const GNNFloat* node_embeddings,
                                     const GNNFloat* layer_weights,
                                     GNNFloat* output);

  void GetWeightGradientsGPU(size_t num_nodes, size_t input_columns,
                             size_t output_columns, const GNNFloat* prev_input,
                             const GNNFloat* gradients, GNNFloat* output);

  void SelfFeatureUpdateEmbeddingsGPU(size_t input_rows, size_t input_columns,
                                      size_t output_columns,
                                      const GNNFloat* node_embeddings,
                                      GNNFloat* output);

  void SelfFeatureUpdateEmbeddingsDerivativeGPU(size_t input_rows,
                                                size_t output_columns,
                                                size_t input_columns,
                                                const GNNFloat* gradients,
                                                GNNFloat* output);

  void UpdateWeight2DerivativeGPU(size_t input_columns, size_t input_rows,
                                  size_t output_columns,
                                  const GNNFloat* prev_layer_inputs,
                                  const GNNFloat* input_gradients,
                                  GNNFloat* output);

  //! Copy provided data in vector to GPU self weight
  void CopyToWeights2(const std::vector<GNNFloat>& cpu_layer_weights);
  //! Copy provided data in vector to GPU self weight gradients
  void CopyToWeight2Gradients(const std::vector<GNNFloat>& cpu_gradients);

  //! Copy GPU self weight gradients to the provided vector (assumes vector is
  //! already correct size)
  void CopyWeight2GradientsToCPU(std::vector<GNNFloat>* cpu_gradients);

private:
  GNNFloat* in_temp_1_{nullptr};
  GNNFloat* in_temp_2_{nullptr};
  GNNFloat* out_temp_{nullptr};
  GNNFloat* layer_weights_2_{nullptr};
  GNNFloat* layer_weight_gradients_2_{nullptr};
};

} // namespace galois

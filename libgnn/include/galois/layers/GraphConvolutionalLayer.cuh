#pragma once
#include "galois/GNNTypes.h"
#include "galois/graphs/GNNGraph.cuh"

namespace galois {

//! Holds pointers for GPU memory for GCN layer
class GCNGPUAllocations {
public:
  // free memory
  ~GCNGPUAllocations();

  void AllocateInTemp1(const size_t size);
  void AllocateInTemp2(const size_t size);
  void AllocateOutTemp(const size_t size);

  GNNFloat* in_temp_1() { return in_temp_1_; }
  GNNFloat* in_temp_2() { return in_temp_2_; }
  GNNFloat* out_temp() { return out_temp_; }

  void AggregateAllGPU(const graphs::GNNGraphGPUAllocations& gpu_graph,
                       size_t num_nodes, size_t column_length,
                       const GNNFloat* node_embeddings,
                       GNNFloat* aggregate_output, bool use_norm,
                       bool disable_self_aggregate, size_t last_master);

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

private:
  GNNFloat* in_temp_1_{nullptr};
  GNNFloat* in_temp_2_{nullptr};
  GNNFloat* out_temp_{nullptr};
};

} // namespace galois

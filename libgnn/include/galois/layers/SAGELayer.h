#pragma once
#include "galois/layers/GNNLayer.h"

#ifdef GALOIS_ENABLE_GPU
// TODO(loc/hochan)
#endif

namespace galois {

struct SAGELayerConfig {
  bool disable_concat{false};
};

// TODO(loc) move common functionality with GCN layer to common parent class
// (e.g. inits): cleans up Dense code a bit as well

//! Same as GCN layer except for the following:
//! - Mean aggregation; no symmetric norm with sqrts used (this
//! ends up performing better for some graphs)
//! - Concatination of the self: rather than aggregating self
//! feature it is concatinated (i.e. dimensions are doubled)
class SAGELayer : public GNNLayer {
public:
  //! Initializes the variables of the base class and also allocates additional
  //! memory for temporary matrices. Also initializes sync substrate for the
  //! weight matrix
  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
            const GNNLayerDimensions& dimensions, const GNNLayerConfig& config,
            const SAGELayerConfig& sage_config);

  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
            const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
      : SAGELayer(layer_num, graph, dimensions, config, SAGELayerConfig()) {}

  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
            const GNNLayerDimensions& dimensions)
      : SAGELayer(layer_num, graph, dimensions, GNNLayerConfig(),
                  SAGELayerConfig()) {}

  void InitSelfWeightsTo1() {
    if (layer_weights_2_.size()) {
      layer_weights_2_.assign(layer_weights_2_.size(), 1);
    }
  }

  //! Returns the 2nd set of weight gradients
  const PointerWithSize<GNNFloat> GetLayerWeightGradients2() {
    return p_layer_weight_gradients_2_;
  }

  // Parent functions
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final;

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(const PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) final;

private:
  //! CPU aggregation
  void AggregateAllCPU(
      size_t column_length, const GNNFloat* node_embeddings,
      GNNFloat* aggregate_output,
      galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts,
      bool is_backward);

  //! Performs aggregation for all nodes of the graph given the length of the
  //! vector to aggregate, the features themselves, an output array, and per
  //! thread storage for the intermediate scaling via norm factor
  void
  AggregateAll(size_t column_length, const GNNFloat* node_embeddings,
               GNNFloat* aggregate_output,
               galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts);
  void
  AggregateAll(size_t column_length, const GNNFloat* node_embeddings,
               GNNFloat* aggregate_output,
               galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts,
               bool is_backward);

  //! Do embedding update via mxm with this layer's weights (forward)
  void UpdateEmbeddings(const GNNFloat* node_embeddings, GNNFloat* output);
  //! Same as above but uses the second set of weights (self feature weights)
  void SelfFeatureUpdateEmbeddings(const GNNFloat* node_embeddings,
                                   GNNFloat* output);
  //! Calculate graident via mxm with last layer's gradients (backward)
  void UpdateEmbeddingsDerivative(const GNNFloat* gradients, GNNFloat* output);
  //! Same as above but uses the second set of weights (self feature weights)
  void SelfFeatureUpdateEmbeddingsDerivative(const GNNFloat* gradients,
                                             GNNFloat* output);

  //! override parent function: optimizes the second set of weights as well
  void OptimizeLayer(BaseOptimizer* optimizer, size_t trainable_layer_number);

  //! SAGE config params
  SAGELayerConfig sage_config_;
  //! Need own optimizer for the 2nd weight matrix
  std::unique_ptr<AdamOptimizer> second_weight_optimizer_;

  // second set of weights for the concat that may occur
  std::vector<GNNFloat> layer_weights_2_;
  std::vector<GNNFloat> layer_weight_gradients_2_;
  PointerWithSize<GNNFloat> p_layer_weights_2_;
  PointerWithSize<GNNFloat> p_layer_weight_gradients_2_;

  // 2 temporaries the size of the forward input; used for dropout and
  // aggregation (if either are required)
  std::vector<GNNFloat> in_temp_1_;
  std::vector<GNNFloat> in_temp_2_;
  // Temporary matrix the size of the output of the forward pass; used if
  // an intermediate op occurs before writing to the final output matrix
  std::vector<GNNFloat> out_temp_;

  // Pointer with size versions
  PointerWithSize<GNNFloat> p_in_temp_1_;
  PointerWithSize<GNNFloat> p_in_temp_2_;
  PointerWithSize<GNNFloat> p_out_temp_;

  // Each thread has a vector of size # input columns or # output columns for
  // storing intermediate results during aggregation.
  // The one used depeneds on if aggregation occurs before or after the mxm.
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      input_column_intermediates_;
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      output_column_intermediates_;

#ifdef GALOIS_ENABLE_GPU
  // TODO(loc/hochan)
  GCNGPUAllocations gpu_object_;
#endif
};

} // namespace galois

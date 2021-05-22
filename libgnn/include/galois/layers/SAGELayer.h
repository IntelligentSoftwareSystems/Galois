#pragma once
#include "galois/layers/GNNLayer.h"
#include "galois/layers/GradientSyncStructures.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/layers/SAGELayer.cuh"
#endif

namespace galois {

extern galois::DynamicBitSet graphs::bitset_graph_aggregate;

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
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions, const GNNLayerConfig& config,
            const SAGELayerConfig& sage_config);

  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
      : SAGELayer(layer_num, graph, backward_output_matrix, dimensions, config,
                  SAGELayerConfig()) {}

  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions)
      : SAGELayer(layer_num, graph, backward_output_matrix, dimensions,
                  GNNLayerConfig(), SAGELayerConfig()) {}

  void InitSelfWeightsTo1() {
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      size_t layer_weights_2_size = p_layer_weights_2_.size();
      if (layer_weights_2_size > 0) {
        base_gpu_object_.InitGPUVectorTo1(gpu_object_.layer_weights_2(),
                                          layer_weights_2_size);
      }
    } else {
#endif
      if (layer_weights_2_.size()) {
        layer_weights_2_.assign(layer_weights_2_.size(), 1);
      }
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  //! Returns the 2nd set of weight gradients
  const PointerWithSize<GNNFloat> GetLayerWeightGradients2() {
    return p_layer_weight_gradients_2_;
  }

  // Parent functions
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final;

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) final;

#ifdef GALOIS_ENABLE_GPU
  //! Copies over self weight gradients to CPU from GPU
  const std::vector<GNNFloat>& CopyWeight2GradientsFromGPU() {
    if (!layer_weight_gradients_2_.size()) {
      layer_weight_gradients_2_.resize(p_layer_weight_gradients_2_.size());
    }
    gpu_object_.CopyWeight2GradientsToCPU(&layer_weight_gradients_2_);
    return layer_weight_gradients_2_;
  }
#endif

private:
  static const constexpr char* kRegionName = "SAGELayer";

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
  void UpdateEmbeddings(const GNNFloat* node_embeddings, GNNFloat* output,
                        bool after);
  //! Same as above but uses the second set of weights (self feature weights)
  void SelfFeatureUpdateEmbeddings(const GNNFloat* node_embeddings,
                                   GNNFloat* output);
  //! Calculate graident via mxm with last layer's gradients (backward)
  void UpdateEmbeddingsDerivative(const GNNFloat* gradients, GNNFloat* output,
                                  bool after);
  //! Same as above but uses the second set of weights (self feature weights)
  void SelfFeatureUpdateEmbeddingsDerivative(const GNNFloat* gradients,
                                             GNNFloat* output);

  //! override parent function: optimizes the second set of weights as well
  void OptimizeLayer(BaseOptimizer* optimizer, size_t trainable_layer_number);

  //! Sync second set of weight gradients
  void WeightGradientSyncSum2();

  void ResizeRows(size_t new_row_count) {
    GNNLayer::ResizeRows(new_row_count);
    ResizeIntermediates(new_row_count, new_row_count);
  }

  void ResizeInputOutputRows(size_t input_row, size_t output_row) {
    GNNLayer::ResizeInputOutputRows(input_row, output_row);
    ResizeIntermediates(input_row, output_row);
  }

  void ResizeIntermediates(size_t new_input_rows, size_t new_output_rows);

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
  SAGEGPUAllocations gpu_object_;
#endif
};

} // namespace galois

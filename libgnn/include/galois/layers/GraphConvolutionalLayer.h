#pragma once
#include "galois/layers/GNNLayer.h"

namespace galois {

class GraphConvolutionalLayer : public GNNLayer {
public:
  //! Initializes the variables of the base class and also allocates additional
  //! memory for temporary matrices. Also initializes sync substrate for the
  //! weight matrix
  GraphConvolutionalLayer(size_t layer_num,
                          const galois::graphs::GNNGraph& graph,
                          const GNNLayerDimensions& dimensions,
                          const GNNLayerConfig& config);

  GraphConvolutionalLayer(size_t layer_num,
                          const galois::graphs::GNNGraph& graph,
                          const GNNLayerDimensions& dimensions)
      : GraphConvolutionalLayer(layer_num, graph, dimensions,
                                GNNLayerConfig()) {}

  // Parent functions
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final;

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(const PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) final;

private:
  // 2 temporaries the size of the forward input; used for dropout and
  // aggregation (if either are required)
  std::vector<GNNFloat> in_temp_1_;
  std::vector<GNNFloat> in_temp_2_;
  // Temporary matrix the size of the output of the forward pass; used if
  // an intermediate op occurs before writing to the final output matrix
  std::vector<GNNFloat> out_temp_;
  // Each thread has a vector of size # input columns or # output columns for
  // storing intermediate results during aggregation.
  // The one used depeneds on if aggregation occurs before or after the mxm.
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      input_column_intermediates_;
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      output_column_intermediates_;

  //! Performs aggregation for all nodes of the graph given the length of the
  //! vector to aggregate, the features themselves, an output array, and per
  //! thread storage for the intermediate scaling via norm factor
  void
  AggregateAll(size_t column_length, const GNNFloat* node_embeddings,
               GNNFloat* aggregate_output,
               galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts);

  //! Do embedding update via mxm with this layer's weights (forward)
  void UpdateEmbeddings(const GNNFloat* node_embeddings, GNNFloat* output);
  //! Calculate graident via mxm with last layer's gradients (backward)
  void UpdateEmbeddingsDerivative(const GNNFloat* gradients, GNNFloat* output);
};

} // namespace galois

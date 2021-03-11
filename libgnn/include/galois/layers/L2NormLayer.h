#pragma once
#include "galois/layers/GNNLayer.h"

#ifdef GALOIS_ENABLE_GPU
// TODO(loc/hochan)
#endif

namespace galois {

//! Applies L2 norm to rows of the input
class L2NormLayer : public GNNLayer {
public:
  L2NormLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,

              PointerWithSize<GNNFloat>* backward_output_matrix,
              const GNNLayerDimensions& dimensions)
      : L2NormLayer(layer_num, graph, backward_output_matrix, dimensions,
                    GNNLayerConfig{.allocate_weights = false}) {}
  L2NormLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
              PointerWithSize<GNNFloat>* backward_output_matrix,
              const GNNLayerDimensions& dimensions,
              const GNNLayerConfig& config)
      : GNNLayer(layer_num, graph, backward_output_matrix, dimensions, config) {
    layer_type_ = galois::GNNLayerType::kL2Norm;
    // input/output columns must be equivalent in a softmax
    GALOIS_LOG_ASSERT(dimensions.input_columns == dimensions.output_columns);
    GALOIS_LOG_VERBOSE("L2 norm initialized");
  }

  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings);

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient);

private:
  const PointerWithSize<galois::GNNFloat>
  ForwardPhaseCPU(const PointerWithSize<galois::GNNFloat> input_embeddings);

  PointerWithSize<galois::GNNFloat>
  BackwardPhaseCPU(PointerWithSize<galois::GNNFloat> prev_layer_input,
                   PointerWithSize<galois::GNNFloat>* input_gradient);

  //! No op
  void OptimizeLayer(BaseOptimizer*, size_t) { return; };

#ifdef GALOIS_ENABLE_GPU
    // TODO(loc/hochan)
#endif
};

} // namespace galois

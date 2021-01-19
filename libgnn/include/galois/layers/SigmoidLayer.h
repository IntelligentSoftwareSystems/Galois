#pragma once
#include "galois/layers/GNNLayer.h"

// TODO(loc) GPU support

namespace galois {

//! Sigmoid layer: applies sigmoid function element wise to each element of the
//! input.
//! Meant for use with *multi-class* labels.
class SigmoidLayer : public GNNLayer {
public:
  SigmoidLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
               const GNNLayerDimensions& dimensions)
      : GNNLayer(layer_num, graph, dimensions,
                 GNNLayerConfig{.allocate_weights = false}),
        input_loss_(dimensions.input_rows),
        norm_gradient_vectors_(dimensions.input_columns) {
    output_layer_type_ = galois::GNNOutputLayerType::kSigmoid;
    // input/output columns must be equivalent
    GALOIS_LOG_ASSERT(dimensions.input_columns == dimensions.output_columns);
    // output needs to match number of possible classes
    GALOIS_LOG_ASSERT(dimensions.input_columns == graph.GetNumLabelClasses());
  }

  //! Normalizes all elements by applying sigmoid to all of them
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final;

  //! Get gradients to fix distribution such that it leans more towards
  //! multiclass ground truth.
  PointerWithSize<galois::GNNFloat>
  BackwardPhase(const PointerWithSize<galois::GNNFloat>,
                PointerWithSize<galois::GNNFloat>*) final;

private:
  const PointerWithSize<galois::GNNFloat>
  ForwardPhaseCPU(const PointerWithSize<galois::GNNFloat> input_embeddings);
  PointerWithSize<galois::GNNFloat> BackwardPhaseCPU();

  //! Loss for each row of the input
  std::vector<GNNFloat> input_loss_;
  //! Each thread gets storage to allocate the gradients during backward
  //! prop; each is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      norm_gradient_vectors_;
};

} // namespace galois

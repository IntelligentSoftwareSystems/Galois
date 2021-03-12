#pragma once
#include "galois/layers/GNNLayer.h"
#ifdef GALOIS_ENABLE_GPU
#include "galois/layers/SoftmaxLayer.cuh"
#endif

namespace galois {

//! Softmax layer: takes each row of the input matrix and creates a probability
//! distribution based on the magnitude of elements in each row.
//! Currently this only works with **single class* labels and is coded as such.
class SoftmaxLayer : public GNNLayer {
public:
  SoftmaxLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,

               PointerWithSize<GNNFloat>* backward_output_matrix,
               const GNNLayerDimensions& dimensions)
      : GNNLayer(
            layer_num, graph, backward_output_matrix, dimensions,
            GNNLayerConfig{.allocate_weights = false, .disable_output = true}),
#ifdef GALOIS_ENABLE_GPU
        gpu_object_(graph.GetGPUGraph()),
#endif
        input_loss_(dimensions.input_rows),
        ground_truth_vectors_(dimensions.input_columns),
        norm_gradient_vectors_(dimensions.input_columns),
        softmax_temp_vectors_(dimensions.input_columns)

  {
    output_layer_type_ = galois::GNNOutputLayerType::kSoftmax;
    // input/output columns must be equivalent in a softmax
    GALOIS_LOG_ASSERT(dimensions.input_columns == dimensions.output_columns);
    // output needs to match number of possible classes
    GALOIS_LOG_ASSERT(dimensions.input_columns == graph.GetNumLabelClasses());
  }

  const PointerWithSize<galois::GNNFloat>
  ForwardPhaseCPU(const PointerWithSize<galois::GNNFloat> input_embeddings);
  //! Creates probability distribution of each row of input
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final;

  PointerWithSize<galois::GNNFloat> BackwardPhaseCPU();
  //! Get gradients to fix distribution such that it leans more towards single
  //! class ground truth.
  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> in_out,
                PointerWithSize<galois::GNNFloat>* input_gradient) final;

private:
#ifdef GALOIS_ENABLE_GPU
  SoftmaxLayerGPU gpu_object_;
#endif

  //! Loss for each row of the input
  std::vector<GNNFloat> input_loss_;
  //! Each thread gets storage to allocate the ground truth vector in during
  //! calculation; each vector is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      ground_truth_vectors_;
  //! Each thread gets storage to allocate the gradients during backward
  //! prop; each is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      norm_gradient_vectors_;
  //! Each thread gets storage for a temporary vector used during softmax
  //! derivative calculation; each is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      softmax_temp_vectors_;
};

} // namespace galois

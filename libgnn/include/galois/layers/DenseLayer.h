#pragma once
#include "galois/layers/GNNLayer.h"

namespace galois {

//! Just does a linear xform with no convolution over graph
class DenseLayer : public GNNLayer {
public:
  //! Initializes the variables of the base class and also allocates additional
  //! memory for temporary matrices. Also initializes sync substrate for the
  //! weight matrix
  DenseLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
             const GNNLayerDimensions& dimensions,
             const GNNLayerConfig& config);

  DenseLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
             const GNNLayerDimensions& dimensions)
      : DenseLayer(layer_num, graph, dimensions, GNNLayerConfig()) {}

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
  // Pointer with size versions
  PointerWithSize<GNNFloat> p_in_temp_1_;

  // Each thread has a vector of size # input columns or # output columns for
  // storing intermediate results during aggregation.
  // The one used depeneds on if aggregation occurs before or after the mxm.
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      input_column_intermediates_;
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      output_column_intermediates_;

  //! Do embedding update via mxm with this layer's weights (forward)
  void UpdateEmbeddings(const GNNFloat* node_embeddings, GNNFloat* output);
  //! Calculate graident via mxm with last layer's gradients (backward)
  void UpdateEmbeddingsDerivative(const GNNFloat* gradients, GNNFloat* output);

#ifdef GALOIS_ENABLE_GPU
  // TODO(hochan/loc) replace with dense gpu object
  GCNGPUAllocations gpu_object_;
#endif
};

} // namespace galois

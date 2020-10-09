#pragma once

#include "galois/PerThreadRNG.h"
#include "galois/GNNOptimizers.h"
#include "galois/graphs/GNNGraph.h"

namespace galois {

//! Supported layer types in the GNN
enum class GNNLayerType {
  //! Invalid placeholder
  kInvalid,
  //! GCN
  kGraphConvolutional
  // TODO SAGE and GAT
};

// TODO Sigmoid
//! Supported output layer types in the GNN
enum class GNNOutputLayerType { kInvalid, kSoftmax };

//! Struct holding the dimensions of a layer. Assumption is that a layer takes
//! a matrix and outputs another matrix with a different # of columns (e.g.
//! matrix multiply with a set of weights)
struct GNNLayerDimensions {
  //! Number of rows in input and output of this layer
  size_t input_rows;
  //! Number of columns in input of this layer
  size_t input_columns;
  //! Number of columns output of this layer
  size_t output_columns;
};

//! Config options for operations that can occur in a layer
struct GNNConfig {
  //! True if weights should be allocated
  bool allocate_weights{true};
  //! True if dropout is to be done at beginning of forward phase
  bool do_dropout{false};
  //! Rate at which to drop things if dropout is on
  float dropout_rate{0.5};
  //! True if some activation function is to be called done at end of forward
  //! phase
  bool do_activation{false};
  //! True if normalization is to occur during multiplies
  bool do_normalization{false};
  // TODO activation type; for now default is softmax
};

// Tried to avoid inheritance, but keeping track of heterogeneous layers
// becomes a mess if there isn't a base class I can create the container on.
//! Base class for layers in a graph neural network
class GNNLayer {
public:
  GNNLayer() = delete;
  //! Creation of a layer needs the # of the layer, the graph to train on, and
  //! the input/output dimensions of the MxM that occurs in the layer; config
  //! as well
  GNNLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
           const GNNLayerDimensions& dimensions, const GNNConfig& config);

  //! Uses a default config
  GNNLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
           const GNNLayerDimensions& dimensions)
      : GNNLayer(layer_num, graph, dimensions, GNNConfig()) {}

  GNNPhase layer_phase() { return layer_phase_; }
  //! Changes this layer's phase
  void SetLayerPhase(GNNPhase new_phase) { layer_phase_ = new_phase; }

  //! Initializes all layer weights to 1. This is used as a debug function for
  //! testing.
  void InitAllWeightsTo1() {
    if (layer_weights_.size()) {
      layer_weights_.assign(layer_weights_.size(), 1);
    }
  }

  const std::vector<GNNFloat>& GetForwardOutput() const {
    return forward_output_matrix_;
  }
  const std::vector<GNNFloat>& GetBackwardOutput() const {
    return backward_output_matrix_;
  }

  //! Returns the weight gradients
  const std::vector<GNNFloat>& GetLayerWeightGradients() const {
    return layer_weight_gradients_;
  }

  //! Returns dimensions of this layer
  const GNNLayerDimensions& GetLayerDimensions() const {
    return layer_dimensions_;
  }

  galois::GNNLayerType layer_type() const { return layer_type_; }
  galois::GNNOutputLayerType output_layer_type() const {
    return output_layer_type_;
  }

  //! Conducts the forward phase given the input to this layer which
  //! ultimately leads to an output (classfication of node labels) at the end
  //! of the GNN.
  //! @returns Output of the forward phase (i.e. input to next layer)
  virtual const std::vector<galois::GNNFloat>&
  ForwardPhase(const std::vector<galois::GNNFloat>& input_embeddings) = 0;
  //! Conducts the backward phase given the input to this layer; the backward
  //! phase calculates the gradients to update the weights of trainable
  //! parts of the layer (e.g., weights, trainable params for aggregate, etc.).
  //! @param prev_layer_input The input that was given to this layer in the
  //! forward phase
  //! @param input_gradient gradient from the backward phase layer before this
  //! one; takes a pointer to save space by writing intermediate results to it
  //! @returns Output of the backward phase (i.e. input to previous layer); note
  //! it's a pointer because layer can mess with it
  virtual std::vector<galois::GNNFloat>*
  BackwardPhase(const std::vector<galois::GNNFloat>& prev_layer_input,
                std::vector<galois::GNNFloat>* input_gradient) = 0;

  //! Given an optimizer, update the weights in this layer based on gradients
  //! stored in the layer
  void OptimizeLayer(BaseOptimizer* optimizer, size_t trainable_layer_number);

protected:
  //! Layer order (starts from 0); used in backward to shortcut output as layer
  //! 0 does not need to do some things that other layers need to do
  // XXX be more specific
  size_t layer_number_;
  //! Pointer to the graph being trained by this layer.
  //! This is owned by the creator of this layer, so no need to free it when
  //! this layer is destroyed.
  const galois::graphs::GNNGraph& graph_;
  //! Dimensions (input/output sizes) of this layer
  GNNLayerDimensions layer_dimensions_;
  //! Config object for certain parameters for layer
  GNNConfig config_;
  //! Weights used by this layer. Dimensions: input columns by output columns
  std::vector<GNNFloat> layer_weights_;
  //! Gradients used to update the weights of this layer
  std::vector<GNNFloat> layer_weight_gradients_;
  // There is a forward and a backward as their sizes will differ and we only
  // want to allocate memory once to avoid runtime memory allocation.
  //! The output of the forward phase for this layer.
  std::vector<GNNFloat> forward_output_matrix_;
  //! The output of the backward phase for this layer.
  std::vector<GNNFloat> backward_output_matrix_;
  //! RNG for matrix initialization
  PerThreadRNG random_init_rng_{-5.0, 5.0};
  //! RNG for dropout
  PerThreadRNG dropout_rng_;
  //! Indicates which fields of the weight matrix are dropped if dropout is
  //! used
  std::vector<bool> dropout_mask_;
  //! Phase of GNN computation that this layer is currently in
  galois::GNNPhase layer_phase_{galois::GNNPhase::kTrain};
  //! Layer type (invalid if output layer)
  galois::GNNLayerType layer_type_{galois::GNNLayerType::kInvalid};
  //! Output layer type (remains invalid if not an output layer)
  galois::GNNOutputLayerType output_layer_type_{
      galois::GNNOutputLayerType::kInvalid};

  //////////////////////////////////////////////////////////////////////////////

  //! Init based from following paper
  //! http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  //! Since it is unclear what j and j+1 refer to in that paper, the things
  //! used are the dimensions of this particular weight matrix
  //! TODO revisit paper and see what they really mean
  //! Code inspired DGL and TinyDNN
  void GlorotBengioInit(std::vector<GNNFloat>* vector_to_init);

  //! Randomly init a float vector using the class's random init RNG
  void RandomInitVector(std::vector<GNNFloat>* vector_to_init);

  //! Choose a set of weights from this layer's weights to keep and save to
  //! the output matrix + apply some scaling to the kept weights based on
  //! dropout rate
  void DoDropout(const std::vector<GNNFloat>& input_to_drop,
                 std::vector<GNNFloat>* output_matrix);
  //! Apply the derivative of dropout to the backward phase output
  void DoDropoutDerivative();

  //! Does some activation function based on configuration on forward output
  //! matrix
  void Activation();
  //! Calculate derivative of activation function based on config on the matrix
  void ActivationDerivative(std::vector<GNNFloat>* matrix);
};

} // namespace galois

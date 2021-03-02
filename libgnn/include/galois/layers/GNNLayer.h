#pragma once

#include "galois/PerThreadRNG.h"
#include "galois/GNNOptimizers.h"
#include "galois/graphs/GNNGraph.h"
#include "galois/layers/GluonGradientInterface.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/layers/GNNLayer.cuh"
#endif

namespace galois {

//! Supported layer types in the GNN
enum class GNNLayerType {
  //! Invalid placeholder
  kInvalid,
  //! GCN
  kGraphConvolutional,
  //! Sage layer: same as GCN except with mean aggregation and concat
  kSAGE,
  //! Dense linear xform layer
  kDense,
  //! L2 normalization layer
  kL2Norm
  // TODO GAT
};

//! Supported output layer types in the GNN
enum class GNNOutputLayerType { kInvalid, kSoftmax, kSigmoid };

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
struct GNNLayerConfig {
  //! True if weights should be allocated
  bool allocate_weights{true};
  //! Turns off dropout of weights if enabled
  bool disable_dropout{false};
  //! Rate at which to drop things if dropout is on
  float dropout_rate{0.5};
  //! True to disable activation function for intermediate layers
  bool disable_activation{false};
  //! True if normalization is disabled to occur during multiplies
  bool disable_normalization{false};
  //! If this is false, aggregate may occur after multiply if # of input columns
  //! is higher than output columns to do less work in aggregation
  bool disable_aggregate_after_update{false};
  //! On to not aggregate self vector during aggregation
  bool disable_self_aggregate{false};
  //! Graph sampling flag in use or not
  bool do_sampling{false};
  //! Inductive layer means for aggregation all non-training nodes are ignored
  bool inductive_training_{false};
  // TODO activation type; for now default is softmax

  //! Sets settings such that testing is easy
  void DebugConfig() {
    disable_activation     = true;
    disable_normalization  = true;
    disable_dropout        = true;
    disable_self_aggregate = true;
  }
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
           const GNNLayerDimensions& dimensions, const GNNLayerConfig& config);

  //! Uses a default config
  GNNLayer(size_t layer_num, const galois::graphs::GNNGraph& graph,
           const GNNLayerDimensions& dimensions)
      : GNNLayer(layer_num, graph, dimensions, GNNLayerConfig()) {}

  GNNPhase layer_phase() { return layer_phase_; }
  //! Changes this layer's phase
  void SetLayerPhase(GNNPhase new_phase) { layer_phase_ = new_phase; }

  void DisableActivation() { config_.disable_activation = true; }

  //! Initializes all layer weights to 1. This is used as a debug function for
  //! testing.
  void InitAllWeightsTo1() {
    if (layer_weights_.size()) {
      layer_weights_.assign(layer_weights_.size(), 1);
    }
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      CopyLayerWeightsToGPU();
    }
#endif
  }

  const PointerWithSize<GNNFloat> GetForwardOutput() {
    return p_forward_output_matrix_;
  }

  const PointerWithSize<GNNFloat> GetBackwardOutput() {
    return p_backward_output_matrix_;
  }

  //! Returns the weight gradients
  const PointerWithSize<GNNFloat> GetLayerWeightGradients() {
    return p_layer_weight_gradients_;
  }

  //! Returns dimensions of this layer
  const GNNLayerDimensions& GetLayerDimensions() const {
    return layer_dimensions_;
  }

  galois::GNNLayerType layer_type() const { return layer_type_; }
  galois::GNNOutputLayerType output_layer_type() const {
    return output_layer_type_;
  }
  size_t layer_number() const { return layer_number_; }

  //! Conducts the forward phase given the input to this layer which
  //! ultimately leads to an output (classfication of node labels) at the end
  //! of the GNN.
  //! @returns Output of the forward phase (i.e. input to next layer)
  // XXX size of embeddings
  virtual const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) = 0;
  //! Conducts the backward phase given the input to this layer; the backward
  //! phase calculates the gradients to update the weights of trainable
  //! parts of the layer (e.g., weights, trainable params for aggregate, etc.).
  //! @param prev_layer_input The input that was given to this layer in the
  //! forward phase
  //! @param input_gradient gradient from the backward phase layer before this
  //! one; takes a pointer to save space by writing intermediate results to it
  //! @returns Output of the backward phase (i.e. input to previous layer); note
  //! it's a pointer because layer can mess with it
  virtual PointerWithSize<galois::GNNFloat>
  BackwardPhase(const PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) = 0;

  //! Given an optimizer, update the weights in this layer based on gradients
  //! stored in the layer
  virtual void OptimizeLayer(BaseOptimizer* optimizer,
                             size_t trainable_layer_number) {
    optimizer->GradientDescent(p_layer_weight_gradients_, p_layer_weights_,
                               trainable_layer_number);
  }

  //! Flip sampling switch on
  void EnableSampling() { config_.do_sampling = true; }
  bool IsSampledLayer() const { return config_.do_sampling; }
  bool IsInductiveLayer() const { return config_.inductive_training_; }

#ifdef GALOIS_ENABLE_GPU
  //! Utility function for allocating
  PointerWithSize<GNNFloat> AllocateGPU(const std::vector<GNNFloat>& v) {
    return PointerWithSize<GNNFloat>(base_gpu_object_.Allocate(v), v.size());
  }
  //! Copies over forward output results to CPU from GPU
  const std::vector<GNNFloat>& CopyForwardOutputFromGPU() {
    base_gpu_object_.CopyForwardOutputToCPU(&forward_output_matrix_);
    return forward_output_matrix_;
  }
  //! Copies over backward output results to CPU from GPU
  const std::vector<GNNFloat>& CopyBackwardOutputFromGPU() {
    base_gpu_object_.CopyBackwardOutputToCPU(&backward_output_matrix_);
    return backward_output_matrix_;
  }
  //! Copies over weight gradients to CPU from GPU
  const std::vector<GNNFloat>& CopyWeightGradientsFromGPU() {
    base_gpu_object_.CopyWeightGradientsToCPU(&layer_weight_gradients_);
    return layer_weight_gradients_;
  }

  void PrintForwardOutputGPU() {
    base_gpu_object_.PrintForwardOutput(forward_output_matrix_.size());
  }
#endif

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
  GNNLayerConfig config_;

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

  // These are wrapper around the pointer for the data associated with
  // any GNN layer: takes a CPU or GPU pointer depending on configuration
  // Needed to allow both CPU/GPU runs with same code
  PointerWithSize<GNNFloat> p_layer_weights_;
  PointerWithSize<GNNFloat> p_layer_weight_gradients_;
  PointerWithSize<GNNFloat> p_forward_output_matrix_;
  PointerWithSize<GNNFloat> p_backward_output_matrix_;

  //! Synchronizes all weights (used in distributed setting)
  void SyncInitialWeights();

  //! Wrapper over gradient matrix to make it compatible with Gluon
  std::unique_ptr<GluonGradientInterface> gradient_sync_interface_;
  //! Synchronization substrate for the weight gradients
  std::unique_ptr<galois::graphs::GluonSubstrate<GluonGradientInterface>>
      gradient_sync_substrate_;

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

  // Used mainly for accuracy tracking
  galois::DGAccumulator<uint32_t> node_count_;
  galois::DGAccumulator<float> float_accumulator_;

  //////////////////////////////////////////////////////////////////////////////

  //! Init based from following paper
  //! http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  //! Since it is unclear what j and j+1 refer to in that paper, the things
  //! used are the dimensions of this particular weight matrix
  //! TODO revisit paper and see what they really mean
  //! Code inspired DGL and TinyDNN
  void GlorotBengioInit(std::vector<GNNFloat>* vector_to_init);

  //! Init 2 things as one unit; used for SAGE
  void PairGlorotBengioInit(std::vector<GNNFloat>* vector1,
                            std::vector<GNNFloat>* vector2);

  //! Randomly init a float vector using the class's random init RNG
  void RandomInitVector(std::vector<GNNFloat>* vector_to_init);

  //! CPU variant of dropout
  void DoDropoutCPU(const PointerWithSize<GNNFloat> input_to_drop,
                    PointerWithSize<GNNFloat>* output_matrix);

  //! Choose a set of weights from this layer's weights to keep and save to
  //! the output matrix + apply some scaling to the kept weights based on
  //! dropout rate
  void DoDropout(const PointerWithSize<GNNFloat> input_to_drop,
                 PointerWithSize<GNNFloat>* output_matrix);
  //! Apply the derivative of dropout to the backward phase output
  void DoDropoutDerivative();

  //! Does some activation function based on configuration on forward output
  //! matrix
  void Activation();
  //! Calculate derivative of activation function based on config on the matrix
  void ActivationDerivative(PointerWithSize<GNNFloat>* matrix);

  //! Synchronize weight gradients with a summation
  void WeightGradientSyncSum();
  //! Synchronize weight gradients with a summation, then locally divide all
  //! weights to get an average
  void WeightGradientSyncAverage();

#ifdef GALOIS_ENABLE_GPU
  //! Object that holds all GPU allocated pointers to memory related to layers
  GNNLayerGPUAllocations base_gpu_object_;
  //! Copies over layer weights to GPU
  void CopyLayerWeightsToGPU() {
    base_gpu_object_.CopyToWeights(layer_weights_);
  }
#endif
};

} // namespace galois

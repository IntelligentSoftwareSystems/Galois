#pragma once

#include "galois/PerThreadRNG.h"
#include "galois/GNNOptimizers.h"
#include "galois/graphs/GNNGraph.h"
#include "galois/Logging.h"
#include "galois/layers/GradientSyncStructures.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/layers/GNNLayer.cuh"
#endif

//#define PRINT_VEC_LOG_
//#define PRINT_GPU_VEC_

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
  kL2Norm,
  //! ReLU layer
  kReLU
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
  //! If rows change, this is set. Otherwise, ignored.
  size_t output_rows;
};

//! Config options for operations that can occur in a layer
struct GNNLayerConfig {
  //! True if weights should be allocated
  bool allocate_weights{true};
  //! If true, disable allocation of the output matrix (used for output layers
  //! which can overwrite the input, i.e. passthrough)
  bool disable_output{false};
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
template <typename VTy, typename ETy>
class GNNLayer {
public:
  //! Creation of a layer needs the # of the layer, the graph to train on, and
  //! the input/output dimensions of the MxM that occurs in the layer; config
  //! as well
  GNNLayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
           PointerWithSize<GNNFloat>* backward_output_matrix,
           const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
      : layer_number_(layer_num), graph_(graph), layer_dimensions_(dimensions),
        config_(config) {
    // TODO(loc)
    // this is currently a backward-compatibility hack, need to have caller
    // set output rows rather than created here
    layer_dimensions_.output_rows = layer_dimensions_.input_rows;

    if (config_.allocate_weights) {
      // dropout allocation; dropout is same as input
      if (!config_.disable_dropout) {
        dropout_mask_.resize(layer_dimensions_.input_rows *
                                 layer_dimensions_.input_columns,
                             false);
      }
      // allocate memory based on layer dimensions
      size_t num_weight_elements =
          layer_dimensions_.input_columns * layer_dimensions_.output_columns;
      galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                    ", layer weights ", num_weight_elements, " (",
                    FloatElementsToGB(num_weight_elements), " GB)");
      layer_weights_.resize(num_weight_elements);
      galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                    ", layer gradients ", num_weight_elements, " (",
                    FloatElementsToGB(num_weight_elements), " GB)");
      layer_weight_gradients_.resize(num_weight_elements, 0);
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        base_gpu_object_.InitWeightMemory(num_weight_elements);
        base_gpu_object_.InitDropoutMemory(layer_dimensions_.input_rows *
                                           layer_dimensions_.input_columns);
      }
#endif

      GlorotBengioInit(&layer_weights_);
    }

    // TODO(loc) optimize this and layer creation in general
    // this does not use output_rows and assumes the worst case where
    // all nodes are generated
    // for now it's kept as input_rows so as to not break things
    size_t num_output_elements =
        layer_dimensions_.input_rows * layer_dimensions_.output_columns;

    if (!config_.disable_output) {
      galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                    ", forward output matrix ", num_output_elements, " (",
                    FloatElementsToGB(num_output_elements), " GB)");
      forward_output_matrix_.resize(num_output_elements, 0);
    }

    if (layer_number_ != 0) {
      GALOIS_LOG_VASSERT(
          backward_output_matrix->size() ==
              layer_dimensions_.input_rows * layer_dimensions_.input_columns,
          "backward output size {} should equal input size {}",
          backward_output_matrix->size(),
          layer_dimensions_.input_rows * layer_dimensions_.input_columns);
    } else {
      GALOIS_LOG_VASSERT(backward_output_matrix->data() == nullptr,
                         "layer 0 should null ptr backward output");
      GALOIS_LOG_VASSERT(backward_output_matrix->size() == 0,
                         "layer 0 should size 0 backward output");
    }

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.InitInOutMemory(num_output_elements,
                                       layer_dimensions_.input_rows *
                                           layer_dimensions_.input_columns);

      // initialize the PointerWithSize wrappers
      p_layer_weights_ = PointerWithSize<GNNFloat>(
          base_gpu_object_.layer_weights(), layer_weights_.size());
      p_layer_weight_gradients_ =
          PointerWithSize<GNNFloat>(base_gpu_object_.layer_weight_gradients(),
                                    layer_weight_gradients_.size());
      p_forward_output_matrix_ = PointerWithSize<GNNFloat>(
          base_gpu_object_.forward_output(), forward_output_matrix_.size());
      p_backward_output_matrix_ = PointerWithSize<GNNFloat>(
          base_gpu_object_.backward_output(), backward_output_matrix->size());
      // TODO can clear the cpu side vectors/don't use .size() since optimally
      // they aren't initialized
    } else {
#endif
      // initialize the PointerWithSize wrappers
      p_layer_weights_ = PointerWithSize<GNNFloat>(layer_weights_);
      p_layer_weight_gradients_ =
          PointerWithSize<GNNFloat>(layer_weight_gradients_);
      p_forward_output_matrix_ =
          PointerWithSize<GNNFloat>(forward_output_matrix_);
      p_backward_output_matrix_ = *backward_output_matrix;
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  //! Uses a default config
  GNNLayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
           PointerWithSize<GNNFloat>* backward_output_matrix,
           const GNNLayerDimensions& dimensions)
      : GNNLayer(layer_num, graph, backward_output_matrix, dimensions,
                 GNNLayerConfig()) {}

  virtual void ResizeRows(size_t new_row_count) {
    layer_dimensions_.input_rows  = new_row_count;
    layer_dimensions_.output_rows = new_row_count;
    ResizeOutputMatrix(new_row_count);
  }

  virtual void ResizeInputOutputRows(size_t input_row, size_t output_row) {
    layer_dimensions_.input_rows  = input_row;
    layer_dimensions_.output_rows = output_row;
    ResizeOutputMatrix(output_row);
  }

  void ResizeOutputMatrix(size_t new_output_row) {
    size_t num_output_elements =
        new_output_row * layer_dimensions_.output_columns;

    if (!config_.disable_output &&
        (forward_output_matrix_.size() < num_output_elements)) {
      galois::gInfo(graph_.host_prefix(), "Resizing layer ", layer_number_,
                    ", forward output matrix to ", num_output_elements, " (",
                    FloatElementsToGB(num_output_elements), " GB)");
      // resize with a bit of a buffer to prevent possible future resizes
      size_t buffer_size = (num_output_elements * 0.02);
      forward_output_matrix_.resize(num_output_elements + buffer_size, 0);
    }

    // XXX(hochan) GPU end
#ifdef GALOIS_ENABLE_GPU
    // XXX(hochan)
#endif
    // reinitialize the PointerWithSize wrappers
    p_forward_output_matrix_ =
        PointerWithSize<GNNFloat>(forward_output_matrix_);
#ifdef GALOIS_ENABLE_GPU
    // XXX(hochan)
#endif
  }

  void UpdateBackwardOutput(PointerWithSize<GNNFloat>* backward_output_matrix) {
    // XXX(hochan) gpu
    if (layer_number_ != 0) {
      assert(backward_output_matrix->size() >=
             layer_dimensions_.input_rows * layer_dimensions_.input_columns);
    } else {
      GALOIS_LOG_FATAL("Layer 0 should not need to update backward output");
    }
    p_backward_output_matrix_ = *backward_output_matrix;
  }

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
  size_t graph_user_layer_number() const { return graph_user_layer_number_; }

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
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
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
  void DisableSampling() { config_.do_sampling = false; }
  bool IsSampledLayer() const { return config_.do_sampling; }
  //! Sets the graph user layer number; important for sampling as this index
  //! determines which index to use when checking for sampled edges
  void SetGraphUserLayerNumber(size_t num) { graph_user_layer_number_ = num; }

#ifdef GALOIS_ENABLE_GPU
  //! Utility function for allocating
  PointerWithSize<GNNFloat> AllocateGPU(const std::vector<GNNFloat>& v) {
    return PointerWithSize<GNNFloat>(base_gpu_object_.Allocate(v), v.size());
  }

  //! Copies over forward output results to CPU from GPU
  const std::vector<GNNFloat> CopyForwardOutputFromGPU() {
    size_t cpu_forward_output_size = p_forward_output_matrix_.size();
    GNNFloat* cpu_forward_output =
        (GNNFloat*)malloc(cpu_forward_output_size * sizeof(GNNFloat));
    base_gpu_object_.CopyForwardOutputToCPU(cpu_forward_output,
                                            cpu_forward_output_size);
    return std::vector<GNNFloat>(cpu_forward_output,
                                 cpu_forward_output + cpu_forward_output_size);
  }

  //! Copies over backward output results to CPU from GPU
  const PointerWithSize<GNNFloat> CopyBackwardOutputFromGPU() {
    size_t cpu_backward_output_size = p_backward_output_matrix_.size();
    GNNFloat* cpu_backward_output =
        (GNNFloat*)malloc(cpu_backward_output_size * sizeof(GNNFloat));
    base_gpu_object_.CopyBackwardOutputToCPU(cpu_backward_output,
                                             cpu_backward_output_size);
    return PointerWithSize<GNNFloat>(cpu_backward_output,
                                     cpu_backward_output_size);
  }

  //! Copies over weight gradients to CPU from GPU
  const std::vector<GNNFloat>& CopyWeightGradientsFromGPU() {
    base_gpu_object_.CopyWeightGradientsToCPU(&layer_weight_gradients_);
    return layer_weight_gradients_;
  }

  void PrintForwardOutputGPU() {
    base_gpu_object_.PrintForwardOutput(forward_output_matrix_.size());
  }

  void PrintBackwardOutputGPU() {
    base_gpu_object_.PrintBackwardOutput(p_backward_output_matrix_.size());
  }
#endif
  void EnableTimers() { use_timer_ = true; }
  void DisableTimers() { use_timer_ = false; }

protected:
  //! Layer order (starts from 0); used in backward to shortcut output as layer
  //! 0 does not need to do some things that other layers need to do
  // XXX be more specific
  size_t layer_number_;
  //! Graph layer number: only layers that use the graph are numbered
  size_t graph_user_layer_number_;
  //! Pointer to the graph being trained by this layer.
  //! This is owned by the creator of this layer, so no need to free it when
  //! this layer is destroyed.
  const galois::graphs::GNNGraph<VTy, ETy>& graph_;
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

  // These are wrapper around the pointer for the data associated with
  // any GNN layer: takes a CPU or GPU pointer depending on configuration
  // Needed to allow both CPU/GPU runs with same code
  PointerWithSize<GNNFloat> p_layer_weights_;
  PointerWithSize<GNNFloat> p_layer_weight_gradients_;
  PointerWithSize<GNNFloat> p_forward_output_matrix_;
  PointerWithSize<GNNFloat> p_backward_output_matrix_;
  galois::DynamicBitSet activation_memo_;

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

  bool use_timer_{true};
  void TimerStart(galois::StatTimer* t) {
    if (use_timer_)
      t->start();
  }
  void TimerStop(galois::StatTimer* t) {
    if (use_timer_)
      t->stop();
  }

  //! Init based from following paper
  //! http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  //! Since it is unclear what j and j+1 refer to in that paper, the things
  //! used are the dimensions of this particular weight matrix
  //! TODO revisit paper and see what they really mean
  //! Code inspired DGL and TinyDNN
  void GlorotBengioInit(std::vector<GNNFloat>* vector_to_init) {
    float max = std::sqrt(6.0) / std::sqrt(layer_dimensions_.output_columns +
                                           layer_dimensions_.input_columns);
    std::default_random_engine rng(1 + layer_number_);
    std::uniform_real_distribution<GNNFloat> dist(-max, max);

    for (size_t i = 0; i < vector_to_init->size(); i++) {
      (*vector_to_init)[i] = dist(rng);
    }
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      CopyLayerWeightsToGPU();
    }
#endif
  }

  //! Init 2 things as one unit; used for SAGE
  void PairGlorotBengioInit(std::vector<GNNFloat>* vector1,
                            std::vector<GNNFloat>* vector2) {
    // multiplied by 2 here because 2 pieces are 1 unit
    float max =
        std::sqrt(6.0) / std::sqrt((2 * layer_dimensions_.output_columns) +
                                   layer_dimensions_.input_columns);
    assert(vector1->size() == (layer_dimensions_.input_columns *
                               layer_dimensions_.output_columns));
    assert(vector2->size() == (layer_dimensions_.input_columns *
                               layer_dimensions_.output_columns));
    std::default_random_engine rng(1 + layer_number_);
    std::uniform_real_distribution<GNNFloat> dist(-max, max);

    for (size_t i = 0; i < vector1->size(); i++) {
      (*vector1)[i] = dist(rng);
    }
    for (size_t i = 0; i < vector2->size(); i++) {
      (*vector2)[i] = dist(rng);
    }

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      CopyLayerWeightsToGPU();
    }
#endif
  }

  //! Randomly init a float vector using the class's random init RNG
  void RandomInitVector(std::vector<GNNFloat>* vector_to_init) {
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), vector_to_init->size()),
        [&](size_t i) {
          // pull from the class's per thread RNG
          (*vector_to_init)[i] = random_init_rng_.GetRandomNumber();
        },
        galois::loopname("RandomInitVector"));
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      CopyLayerWeightsToGPU();
    }
#endif
  }

  //! CPU variant of dropout
  void DoDropoutCPU(const PointerWithSize<GNNFloat> input_to_dropout,
                    PointerWithSize<GNNFloat>* output_matrix) {
    // TODO This (and dropout in general) may not work in the sampling setting
    size_t num_elements =
        layer_dimensions_.input_rows * layer_dimensions_.input_columns;

    // determine which parts to drop
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), num_elements),
        [&](size_t i) {
          dropout_mask_[i] = dropout_rng_.DoBernoulli(config_.dropout_rate);
        },
        galois::loopname("LayerDropoutRNG"));

    // create new matrix with non-dropped input + some scaling
    // TODO save scaling elsewhere?
    GNNFloat scale = 1. / (1. - config_.dropout_rate);
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), num_elements),
        [&](size_t i) {
          (*output_matrix)[i] = input_to_dropout[i] *
                                static_cast<GNNFloat>(dropout_mask_[i]) * scale;
        },
        galois::loopname("LayerDropout"));
  }

  //! Choose a set of weights from this layer's weights to keep and save to
  //! the output matrix + apply some scaling to the kept weights based on
  //! dropout rate
  void DoDropout(const PointerWithSize<GNNFloat> input_to_dropout,
                 PointerWithSize<GNNFloat>* output_matrix) {
    galois::StatTimer timer("ForwardDropout", "GNNLayer");
    TimerStart(&timer);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.DoDropoutGPU(input_to_dropout, *output_matrix,
                                    config_.dropout_rate);
    } else {
#endif
      DoDropoutCPU(input_to_dropout, output_matrix);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    TimerStop(&timer);
  }

  //! Apply the derivative of dropout to the backward phase output
  void DoDropoutDerivative() {
    galois::StatTimer timer("BackwardDropout", "GNNLayer");
    TimerStart(&timer);
    assert(p_backward_output_matrix_.size() == dropout_mask_.size());
    GNNFloat scale = 1. / (1. - config_.dropout_rate);

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.DoDropoutDerivativeGPU(p_backward_output_matrix_.size(),
                                              scale);
    } else {
#endif
      // use dropout mask to figure out derivative
      galois::do_all(
          galois::iterate(static_cast<size_t>(0),
                          p_backward_output_matrix_.size()),
          [&](size_t i) {
            p_backward_output_matrix_[i] =
                p_backward_output_matrix_[i] *
                static_cast<GNNFloat>(dropout_mask_[i]) * scale;
          },
          galois::loopname("LayerDropoutDerivative"));
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    TimerStop(&timer);
  }

  void
  ReconstructDropoutMatrix(const PointerWithSize<GNNFloat> input_to_dropout,
                           PointerWithSize<GNNFloat>* output_matrix) {
    galois::StatTimer timer("ReconstructDropoutMatrix", "GNNLayer");
    TimerStart(&timer);
    // reuse the dropout mask from a previous dropout call
    size_t num_elements = output_matrix->size();
    GNNFloat scale      = 1. / (1. - config_.dropout_rate);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.ReconstructDropoutMatrixGPU(
          input_to_dropout, output_matrix, num_elements, scale);
    } else {
#endif
      galois::do_all(
          galois::iterate(static_cast<size_t>(0), num_elements),
          [&](size_t i) {
            (*output_matrix)[i] = input_to_dropout[i] *
                                  static_cast<GNNFloat>(dropout_mask_[i]) *
                                  scale;
          },
          galois::loopname("ReconstructDropout"));
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    TimerStop(&timer);
  }

  //! Does some activation function based on configuration on forward output
  //! matrix
  void Activation() {
    galois::StatTimer timer("ForwardActivation", "GNNLayer");
    TimerStart(&timer);

    // TODO only does relu at the moment; should check user specified activation
    // and act accordingly
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.ActivationGPU(p_forward_output_matrix_.size());
    } else {
#endif
      if (activation_memo_.size() != p_forward_output_matrix_.size()) {
        activation_memo_.resize(p_forward_output_matrix_.size());
      }
      activation_memo_.reset();
      assert(activation_memo_.size() == p_forward_output_matrix_.size());
      assert(layer_dimensions_.output_rows * layer_dimensions_.output_columns <=
             p_forward_output_matrix_.size());

      galois::do_all(galois::iterate(static_cast<size_t>(0),
                                     layer_dimensions_.output_rows *
                                         layer_dimensions_.output_columns),
                     [&](size_t i) {
                       if (p_forward_output_matrix_[i] > 0.0) {
                         // do nothing, keep value; set the memo though
                         activation_memo_.set(i);
                       } else {
                         p_forward_output_matrix_[i] = 0;
                       }
                     });
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    TimerStop(&timer);
  }

  void ActivationCPU();
  //! Calculate derivative of activation function based on config on the matrix
  void ActivationDerivative(PointerWithSize<GNNFloat>* gradient) {
    galois::StatTimer timer("BackwardActivation", "GNNLayer");
    TimerStart(&timer);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.ActivationDerivativeGPU(gradient->data(),
                                               gradient->size());
    } else {
#endif
      assert(gradient->size() >=
             layer_dimensions_.output_rows * layer_dimensions_.output_columns);
      // TODO only does relu at the moment; should check user specified
      // activation and act accordingly keep gradient if the original output was
      // greater than 0
      galois::do_all(
          galois::iterate(static_cast<size_t>(0),
                          layer_dimensions_.output_rows *
                              layer_dimensions_.output_columns),
          [&](size_t i) {
            // it was <= 0 before; set back to 0
            if (!activation_memo_.test(i)) {
              (*gradient)[i] = 0;
            }
          },
          galois::loopname("ReLU-Derivative"));
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    TimerStop(&timer);
  }

  //! Synchronize weight gradients with a summation
  void WeightGradientSyncSum() {
    galois::StatTimer clubbed_timer("Sync_BackwardSync", "Gluon");
    TimerStart(&clubbed_timer);
    galois::StatTimer t("Sync_WeightGradientsSum", "GNNLayer");
    TimerStart(&t);
    int weight_size = static_cast<int>(p_layer_weight_gradients_.size());

    // TODO(loc) remove this limitation later; can just do a loop over the
    // weight matrix
    if (p_layer_weight_gradients_.size() >
        size_t{std::numeric_limits<int>::max()}) {
      GALOIS_LOG_FATAL("Weight sync code does not handle size larger than max "
                       "int at the moment");
    }
#ifdef GALOIS_ENABLE_GPU
    // TODO(lhc) make this clang option later
    bool gpu_direct_enabled = false;
    if (device_personality == DevicePersonality::GPU_CUDA &&
        !gpu_direct_enabled) {
      base_gpu_object_.CopyWeightGradientsToCPU(&layer_weight_gradients_);
      MPI_Allreduce(MPI_IN_PLACE, layer_weight_gradients_.data(), weight_size,
                    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      base_gpu_object_.CopyToWeightGradients(layer_weight_gradients_);
    } else {
#endif
      MPI_Allreduce(MPI_IN_PLACE,
                    static_cast<void*>(p_layer_weight_gradients_.data()),
                    weight_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    TimerStop(&t);
    TimerStop(&clubbed_timer);
  }

#ifdef GALOIS_ENABLE_GPU
  //! Object that holds all GPU allocated pointers to memory related to layers
  GNNLayerGPUAllocations base_gpu_object_;
  //! Copies over layer weights to GPU
  void CopyLayerWeightsToGPU() {
    base_gpu_object_.CopyToWeights(layer_weights_);
  }
#endif

  void MaskInputNonMasters(PointerWithSize<GNNFloat>* input) {
    MaskInputNonMasters(input, std::numeric_limits<size_t>::max());
  }
  void MaskInputNonMasters(PointerWithSize<GNNFloat>* input, size_t max_rows) {
    assert(*(graph_.begin_owned()) == 0);
    size_t start_node = *(graph_.end_owned());
    size_t end_node   = graph_.active_size();

    if (start_node > max_rows) {
      start_node = max_rows;
    }
    if (end_node > max_rows) {
      end_node = max_rows;
    }

    size_t row_index = layer_dimensions_.input_columns;
    assert(start_node * row_index <= input->size());
    assert(end_node * row_index <= input->size());

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.MaskNonMastersGPU(input, start_node, end_node,
                                         row_index);
    } else {
#endif
      galois::do_all(
          galois::iterate(start_node, end_node),
          [&](size_t non_master) {
            // TODO(loc) use a std function for this for max efficiency
            for (size_t i = 0; i < row_index; i++) {
              (*input)[non_master * row_index + i] = 0;
            }
          },
          galois::loopname("MaskInputNonMasters"));
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  void MaskInputNonMasters(PointerWithSize<GNNFloat>* input, size_t max_rows,
                           const galois::DynamicBitSet& bs) {
    assert(*(graph_.begin_owned()) == 0);
    size_t start_node = *(graph_.end_owned());
    size_t end_node   = graph_.active_size();

    if (start_node > max_rows) {
      start_node = max_rows;
    }
    if (end_node > max_rows) {
      end_node = max_rows;
    }

    size_t row_index = layer_dimensions_.input_columns;
    assert(start_node * row_index <= input->size());
    assert(end_node * row_index <= input->size());

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.MaskNonMastersGPU(input, start_node, end_node,
                                         row_index);
    } else {
#endif
      galois::do_all(
          galois::iterate(start_node, end_node),
          [&](size_t non_master) {
            if (!bs.test(non_master)) {
              // TODO(loc) use a std function for this for max efficiency
              for (size_t i = 0; i < row_index; i++) {
                (*input)[non_master * row_index + i] = 0;
              }
            }
          },
          galois::loopname("MaskInputNonMasters"));
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  //! Mask a gradient size'd matrix's rows that correspond to mirrors
  void MaskGradientNonMasters(PointerWithSize<GNNFloat>* input) {
    MaskGradientNonMasters(input, std::numeric_limits<size_t>::max());
  }
  void MaskGradientNonMasters(PointerWithSize<GNNFloat>* gradients,
                              size_t max_rows) {
    assert(*(graph_.begin_owned()) == 0);
    size_t start_node = *(graph_.end_owned());
    size_t end_node   = graph_.active_size();

    if (start_node > max_rows) {
      start_node = max_rows;
    }
    if (end_node > max_rows) {
      end_node = max_rows;
    }

    size_t row_index = layer_dimensions_.output_columns;
    if (start_node > max_rows) {
      start_node = max_rows;
    }
    if (end_node > max_rows) {
      end_node = max_rows;
    }
    assert(start_node * row_index <= gradients->size());
    assert(end_node * row_index <= gradients->size());

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.MaskNonMastersGPU(gradients, start_node, end_node,
                                         row_index);
    } else {
#endif
      galois::do_all(
          galois::iterate(start_node, end_node),
          [&](size_t non_master) {
            // TODO(loc) use a std function for this for max efficiency
            for (size_t i = 0; i < row_index; i++) {
              (*gradients)[non_master * row_index + i] = 0;
            }
          },
          galois::loopname("MaskGradientNonMasters"));
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  void MaskGradientNonMasters(PointerWithSize<GNNFloat>* gradients,
                              size_t max_rows,
                              const galois::DynamicBitSet& bs) {
    assert(*(graph_.begin_owned()) == 0);
    size_t start_node = *(graph_.end_owned());
    size_t end_node   = graph_.active_size();

    if (start_node > max_rows) {
      start_node = max_rows;
    }
    if (end_node > max_rows) {
      end_node = max_rows;
    }

    size_t row_index = layer_dimensions_.output_columns;
    if (start_node > max_rows) {
      start_node = max_rows;
    }
    if (end_node > max_rows) {
      end_node = max_rows;
    }
    assert(start_node * row_index <= gradients->size());
    assert(end_node * row_index <= gradients->size());

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.MaskNonMastersGPU(gradients, start_node, end_node,
                                         row_index);
    } else {
#endif
      // galois::gInfo(start_node, " to ", end_node);
      galois::do_all(
          galois::iterate(start_node, end_node),
          [&](size_t non_master) {
            // if something is not a master, kill it
            if (!bs.test(non_master)) {
              // galois::gInfo("don't keep ", non_master);
              // TODO(loc) use a std function for this for max efficiency
              for (size_t i = 0; i < row_index; i++) {
                (*gradients)[non_master * row_index + i] = 0;
              }
            }
          },
          galois::loopname("MaskGradientNonMasters"));
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  //! Does some math to get GB used by some # of floats
  double FloatElementsToGB(size_t num_of_floats) const {
    return num_of_floats * double{4} / (1 << 30);
  }

  void MaskNonMastersGPU(PointerWithSize<GNNFloat>* input, size_t start_node,
                         size_t end_node, size_t row_index);
};

} // namespace galois

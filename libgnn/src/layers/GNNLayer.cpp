#include "galois/Logging.h"
#include "galois/layers/GNNLayer.h"
#include "galois/layers/GradientSyncStructures.h"

galois::GNNLayer::GNNLayer(size_t layer_num,
                           const galois::graphs::GNNGraph& graph,
                           PointerWithSize<GNNFloat>* backward_output_matrix,
                           const GNNLayerDimensions& dimensions,
                           const GNNLayerConfig& config)
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

void galois::GNNLayer::GlorotBengioInit(std::vector<GNNFloat>* vector_to_init) {
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

void galois::GNNLayer::PairGlorotBengioInit(std::vector<GNNFloat>* vector1,
                                            std::vector<GNNFloat>* vector2) {
  // multiplied by 2 here because 2 pieces are 1 unit
  float max =
      std::sqrt(6.0) / std::sqrt((2 * layer_dimensions_.output_columns) +
                                 layer_dimensions_.input_columns);
  assert(vector1->size() ==
         (layer_dimensions_.input_columns * layer_dimensions_.output_columns));
  assert(vector2->size() ==
         (layer_dimensions_.input_columns * layer_dimensions_.output_columns));
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

void galois::GNNLayer::RandomInitVector(std::vector<GNNFloat>* vector_to_init) {
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

void galois::GNNLayer::DoDropoutCPU(
    const PointerWithSize<GNNFloat> input_to_dropout,
    PointerWithSize<GNNFloat>* output_matrix) {
  // XXX(loc) check this to make sure it works in subgraph setting
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

void galois::GNNLayer::DoDropout(
    const PointerWithSize<GNNFloat> input_to_dropout,
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

void galois::GNNLayer::ReconstructDropoutMatrix(
    const PointerWithSize<GNNFloat> input_to_dropout,
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
                                static_cast<GNNFloat>(dropout_mask_[i]) * scale;
        },
        galois::loopname("ReconstructDropout"));
#ifdef GALOIS_ENABLE_GPU
  }
#endif
  TimerStop(&timer);
}

void galois::GNNLayer::DoDropoutDerivative() {
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

void galois::GNNLayer::Activation() {
  galois::StatTimer timer("ForwardActivation", "GNNLayer");
  TimerStart(&timer);

  // TODO only does relu at the moment; should check user specified activation
  // and act accordingly
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    base_gpu_object_.ActivationGPU(p_forward_output_matrix_.size());
  } else {
#endif
    if (activation_memo_.size() == 0) {
      activation_memo_.resize(forward_output_matrix_.size());
    }
    activation_memo_.reset();

    galois::do_all(galois::iterate(static_cast<size_t>(0),
                                   layer_dimensions_.output_rows *
                                       layer_dimensions_.output_columns),
                   [&](size_t i) {
                     if (forward_output_matrix_[i] > 0.0) {
                       // do nothing, keep value; set the memo though
                       activation_memo_.set(i);
                     } else {
                       forward_output_matrix_[i] = 0;
                     }
                   });
#ifdef GALOIS_ENABLE_GPU
  }
#endif
  TimerStop(&timer);
}

void galois::GNNLayer::ActivationDerivative(
    PointerWithSize<GNNFloat>* gradient) {
  galois::StatTimer timer("BackwardActivation", "GNNLayer");
  TimerStart(&timer);

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    base_gpu_object_.ActivationDerivativeGPU(gradient->data(),
                                             gradient->size());
  } else {
#endif
    // TODO only does relu at the moment; should check user specified activation
    // and act accordingly
    // keep gradient if the original output was greater than 0
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

void galois::GNNLayer::WeightGradientSyncSum() {
  galois::StatTimer t("Sync_WeightGradientsSum", "GNNLayer");
  TimerStart(&t);
  int weight_size = static_cast<int>(p_layer_weight_gradients_.size());

  // TODO(loc) remove this limitation later; can just do a loop over the weight
  // matrix
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
}

void galois::GNNLayer::MaskInputNonMasters(PointerWithSize<GNNFloat>* input,
                                           size_t max_rows) {
  assert(*(graph_.begin_owned()) == 0);
  size_t start_node = *(graph_.end_owned());
  size_t end_node   = graph_.active_size();
  size_t row_index  = layer_dimensions_.input_columns;
  assert((row_index * layer_dimensions_.input_rows) <= input->size());

  if (start_node > max_rows) {
    start_node = max_rows;
  }
  if (end_node > max_rows) {
    end_node = max_rows;
  }

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    base_gpu_object_.MaskNonMastersGPU(input, start_node, end_node, row_index);
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

void galois::GNNLayer::MaskGradientNonMasters(
    PointerWithSize<GNNFloat>* gradient, size_t max_rows) {
  assert(*(graph_.begin_owned()) == 0);
  size_t start_node = *(graph_.end_owned());
  size_t end_node   = graph_.active_size();
  size_t row_index  = layer_dimensions_.output_columns;

  if (start_node > max_rows) {
    start_node = max_rows;
  }
  if (end_node > max_rows) {
    end_node = max_rows;
  }

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    base_gpu_object_.MaskNonMastersGPU(gradient, start_node, end_node,
                                       row_index);
  } else {
#endif
    galois::do_all(
        galois::iterate(start_node, end_node),
        [&](size_t non_master) {
          // TODO(loc) use a std function for this for max efficiency
          for (size_t i = 0; i < row_index; i++) {
            (*gradient)[non_master * row_index + i] = 0;
          }
        },
        galois::loopname("MaskGradientNonMasters"));
#ifdef GALOIS_ENABLE_GPU
  }
#endif
}

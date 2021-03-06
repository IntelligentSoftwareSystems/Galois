#include "galois/Logging.h"
#include "galois/layers/GNNLayer.h"
#include "galois/layers/GradientSyncStructures.h"

galois::GNNLayer::GNNLayer(size_t layer_num,
                           const galois::graphs::GNNGraph& graph,
                           const GNNLayerDimensions& dimensions,
                           const GNNLayerConfig& config)
    : layer_number_(layer_num), graph_(graph), layer_dimensions_(dimensions),
      config_(config) {
  if (config_.allocate_weights) {
    // TODO some of this does not need alloc if not used
    // dropout allocation; dropout is same as input
    dropout_mask_.resize(
        layer_dimensions_.input_rows * layer_dimensions_.input_columns, false);
    // allocate memory based on layer dimensions
    size_t num_weight_elements =
        layer_dimensions_.input_columns * layer_dimensions_.output_columns;
    layer_weights_.resize(num_weight_elements);
    layer_weight_gradients_.resize(num_weight_elements, 0);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      base_gpu_object_.InitWeightMemory(num_weight_elements);
      base_gpu_object_.InitDropoutMemory(layer_dimensions_.input_rows *
                                         layer_dimensions_.input_columns);
    }
#endif

    GlorotBengioInit(&layer_weights_);

    // initialize sync substrate
    gradient_sync_interface_ =
        std::make_unique<GluonGradientInterface>(layer_weight_gradients_);
    gradient_sync_substrate_ = std::make_unique<
        galois::graphs::GluonSubstrate<GluonGradientInterface>>(
        *gradient_sync_interface_,
        galois::runtime::getSystemNetworkInterface().ID,
        galois::runtime::getSystemNetworkInterface().Num, false);
  }

  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;
  forward_output_matrix_.resize(num_output_elements, 0);
  backward_output_matrix_.resize(
      layer_dimensions_.input_rows * layer_dimensions_.input_columns, 0);
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
        base_gpu_object_.backward_output(), backward_output_matrix_.size());
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
    p_backward_output_matrix_ =
        PointerWithSize<GNNFloat>(backward_output_matrix_);
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
  // TODO
  GALOIS_LOG_FATAL("TODO: copy both not 1");
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
  size_t num_elements = output_matrix->size();
  assert(num_elements == dropout_mask_.size());
  assert(num_elements == input_to_dropout.size());

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
  timer.start();
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
  timer.stop();
}

void galois::GNNLayer::ReconstructDropoutMatrix(
    const PointerWithSize<GNNFloat> input_to_dropout,
    PointerWithSize<GNNFloat>* output_matrix) {
  galois::StatTimer timer("ReconstructDropoutMatrix", "GNNLayer");
  timer.start();
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    // TODO(hochan)
    GALOIS_LOG_FATAL("Implement me");
  } else {
#endif
    // reuse the dropout mask from a previous dropout call
    size_t num_elements = output_matrix->size();
    GNNFloat scale      = 1. / (1. - config_.dropout_rate);
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
  timer.stop();
}

void galois::GNNLayer::DoDropoutDerivative() {
  galois::StatTimer timer("BackwardDropout", "GNNLayer");
  timer.start();
  assert(backward_output_matrix_.size() == dropout_mask_.size());
  GNNFloat scale = 1. / (1. - config_.dropout_rate);

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    base_gpu_object_.DoDropoutDerivativeGPU(p_backward_output_matrix_.size(),
                                            scale);
  } else {
#endif
    // use dropout mask to figure out derivative
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), backward_output_matrix_.size()),
        [&](size_t i) {
          backward_output_matrix_[i] = backward_output_matrix_[i] *
                                       static_cast<GNNFloat>(dropout_mask_[i]) *
                                       scale;
        },
        galois::loopname("LayerDropoutDerivative"));
#ifdef GALOIS_ENABLE_GPU
  }
#endif
  timer.stop();
}

void galois::GNNLayer::Activation() {
  galois::StatTimer timer("ForwardActivation", "GNNLayer");
  timer.start();

  // TODO only does relu at the moment; should check user specified activation
  // and act accordingly
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), forward_output_matrix_.size()),
      [&](size_t i) {
        forward_output_matrix_[i] =
            std::max(forward_output_matrix_.at(i), static_cast<GNNFloat>(0));
      },
      galois::loopname("ReLU"));
  timer.stop();
}

void galois::GNNLayer::ActivationDerivative(
    PointerWithSize<GNNFloat>* gradient) {
  galois::StatTimer timer("BackwardActivation", "GNNLayer");
  timer.start();

  // TODO only does relu at the moment; should check user specified activation
  // and act accordingly
  // keep gradient if the original output is greater than 0
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), gradient->size()),
      [&](size_t i) {
        (*gradient)[i] =
            (forward_output_matrix_.at(i) > static_cast<GNNFloat>(0))
                ? (*gradient)[i]
                : static_cast<GNNFloat>(0);
      },
      galois::loopname("ReLU-Derivative"));
  timer.stop();
}

void galois::GNNLayer::WeightGradientSyncSum() {
  galois::StatTimer t("Sync_WeightGradientsSum", "GNNLayer");
  t.start();
#ifdef GALOIS_ENABLE_GPU
  // TODO(hochan) collectives here rather than gluon sync if possible like the
  // CPU code
  // preferably without needing to do a gpu->cpu copy
  galois::gWarn(
      "GPU still using inefficient point to point comms for weight sync");
  gradient_sync_substrate_->sync<writeAny, readAny, WeightGradientSummation>(
      "WeightGradientsSync");
#else
  // TODO(loc) remove this limitation later; can just do a loop over the weight
  // matrix
  if (p_layer_weight_gradients_.size() >
      size_t{std::numeric_limits<int>::max()}) {
    GALOIS_LOG_FATAL("Weight sync code does not handle size larger than max "
                     "int at the moment");
  }
  MPI_Allreduce(MPI_IN_PLACE,
                static_cast<void*>(p_layer_weight_gradients_.data()),
                static_cast<int>(p_layer_weight_gradients_.size()), MPI_FLOAT,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  t.stop();
}

void galois::GNNLayer::SyncInitialWeights() {
  if (galois::runtime::getSystemNetworkInterface().Num == 1) {
    return;
  }
#ifdef GALOIS_ENABLE_GPU
  // TODO(loc/hochan); not required at the moment however
  GALOIS_LOG_FATAL("Need to implement GPU version of this");
#endif
  // copy weights over to gradients
  for (size_t i = 0; i < layer_weights_.size(); i++) {
    layer_weight_gradients_[i] = layer_weights_[i];
  }
  // sync "gradients" with a set only (reduction ignored)
  gradient_sync_substrate_->sync<writeAny, readAny, WeightGradientSet>(
      "InitialSync");
  // copy "gradients" (actually weights) back to weight matrix
  for (size_t i = 0; i < layer_weights_.size(); i++) {
    layer_weights_[i]          = layer_weight_gradients_[i];
    layer_weight_gradients_[i] = 0;
  }
}

void galois::GNNLayer::MaskGradientNonMasters(
    PointerWithSize<GNNFloat>* gradient) {
#ifdef GALOIS_ENABLE_GPU
  // TODO(hochan) mask away the **non** masters on gpu
  GALOIS_LOG_FATAL("implement this");
#else
  assert(*(graph_.begin_owned()) == 0);
  size_t start_node = *(graph_.end_owned());
  size_t end_node   = graph_.size();
  size_t row_index  = layer_dimensions_.output_columns;
  galois::do_all(
      galois::iterate(start_node, end_node),
      [&](size_t non_master) {
        // TODO(loc) use a std function for this for max efficiency
        for (size_t i = 0; i < row_index; i++) {
          (*gradient)[non_master * row_index + i] = 0;
        }
      },
      galois::loopname("MaskGradientNonMasters"));
#endif
}

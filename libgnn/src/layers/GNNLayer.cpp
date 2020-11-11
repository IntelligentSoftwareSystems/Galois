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
    GlorotBengioInit(&layer_weights_);

    // initialize sync substrate
    gradient_sync_interface_ =
        std::make_unique<GluonGradientInterface>(layer_weight_gradients_);
    gradient_sync_substrate_ = std::make_unique<
        galois::graphs::GluonSubstrate<GluonGradientInterface>>(
        *gradient_sync_interface_,
        galois::runtime::getSystemNetworkInterface().ID,
        galois::runtime::getSystemNetworkInterface().Num, false);
#ifdef GALOIS_ENABLE_GPU
    base_gpu_object_.InitWeightMemory(num_weight_elements);
#endif
  }

  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;
  forward_output_matrix_.resize(num_output_elements, 0);
  backward_output_matrix_.resize(
      layer_dimensions_.input_rows * layer_dimensions_.input_columns, 0);
#ifdef GALOIS_ENABLE_GPU
  base_gpu_object_.InitInOutMemory(num_output_elements,
                                   layer_dimensions_.input_rows *
                                       layer_dimensions_.input_columns);
#endif

  // initialize the PointerWithSize wrappers
#ifndef GALOIS_ENABLE_GPU
  p_layer_weights_ = PointerWithSize<GNNFloat>(layer_weights_);
  p_layer_weight_gradients_ =
      PointerWithSize<GNNFloat>(layer_weight_gradients_);
  p_forward_output_matrix_ = PointerWithSize<GNNFloat>(forward_output_matrix_);
  p_backward_output_matrix_ =
      PointerWithSize<GNNFloat>(backward_output_matrix_);
#else
  p_layer_weights_ = PointerWithSize<GNNFloat>(base_gpu_object_.layer_weights(),
                                               layer_weights_.size());
  p_layer_weight_gradients_ =
      PointerWithSize<GNNFloat>(base_gpu_object_.layer_weight_gradients(),
                                layer_weight_gradients_.size());
  p_forward_output_matrix_ = PointerWithSize<GNNFloat>(
      base_gpu_object_.forward_output(), forward_output_matrix_.size());
  p_backward_output_matrix_ = PointerWithSize<GNNFloat>(
      base_gpu_object_.backward_output(), backward_output_matrix_.size());
  // TODO can clear the cpu side vectors/don't use .size() since optimally they
  // aren't initialized
#endif
}

void galois::GNNLayer::GlorotBengioInit(std::vector<GNNFloat>* vector_to_init) {
  float max = std::sqrt(6.0) / std::sqrt(layer_dimensions_.output_columns +
                                         layer_dimensions_.input_columns);
  // TODO this seed should be configurable
  std::default_random_engine rng(1);
  std::uniform_real_distribution<GNNFloat> dist(-max, max);

  for (size_t i = 0; i < vector_to_init->size(); i++) {
    (*vector_to_init)[i] = dist(rng);
  }
}

void galois::GNNLayer::RandomInitVector(std::vector<GNNFloat>* vector_to_init) {
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), vector_to_init->size()),
      [&](size_t i) {
        // pull from the class's per thread RNG
        (*vector_to_init)[i] = random_init_rng_.GetRandomNumber();
      },
      galois::loopname("RandomInitVector"));
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
  //#ifdef GALOIS_ENABLE_GPU
  //  // XXX
  //  DoDropoutGPU();
  //#else
  DoDropoutCPU(input_to_dropout, output_matrix);
  //#endif
}

void galois::GNNLayer::DoDropoutDerivative() {
  assert(backward_output_matrix_.size() == dropout_mask_.size());
  GNNFloat scale = 1. / (1. - config_.dropout_rate);

  // use dropout mask to figure out derivative
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), backward_output_matrix_.size()),
      [&](size_t i) {
        backward_output_matrix_[i] = backward_output_matrix_[i] *
                                     static_cast<GNNFloat>(dropout_mask_[i]) *
                                     scale;
      },
      galois::loopname("LayerDropoutDerivative"));
}

void galois::GNNLayer::Activation() {
  // TODO only does relu at the moment; should check user specified activation
  // and act accordingly
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), forward_output_matrix_.size()),
      [&](size_t i) {
        forward_output_matrix_[i] =
            std::max(forward_output_matrix_.at(i), static_cast<GNNFloat>(0));
      },
      galois::loopname("ReLU"));
}

void galois::GNNLayer::ActivationDerivative(
    PointerWithSize<GNNFloat>* gradient) {
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
}

void galois::GNNLayer::OptimizeLayer(BaseOptimizer* optimizer,
                                     size_t trainable_layer_number) {
  optimizer->GradientDescent(layer_weight_gradients_, &layer_weights_,
                             trainable_layer_number);
}

void galois::GNNLayer::WeightGradientSyncSum() {
  // XXX bitset
  gradient_sync_substrate_->sync<writeAny, readAny, WeightGradientSummation>(
      "WeightGradientsSync");
}

void galois::GNNLayer::WeightGradientSyncAverage() {
  size_t num_hosts = galois::runtime::getSystemNetworkInterface().Num;
  if (num_hosts > 1) {
    // XXX bitset
    // sum, then average by dividing all by num hosts (every host participates
    // in sync)
    gradient_sync_substrate_->sync<writeAny, readAny, WeightGradientSummation>(
        "WeightGradientsSyncAverage");
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), layer_weight_gradients_.size()),
        [&](size_t weight_index) {
          layer_weight_gradients_[weight_index] /= num_hosts;
        },
        galois::loopname("WeightGradientSyncAverageDivide"));
  }
}

#ifdef GALOIS_ENABLE_GPU
void galois::GNNLayer::CopyLayerWeightsToGPU() {
  base_gpu_object_.CopyToWeights(layer_weights_);
}
#endif

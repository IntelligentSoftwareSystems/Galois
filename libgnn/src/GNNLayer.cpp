#include "galois/layers/GNNLayer.h"

galois::GNNLayer::GNNLayer(size_t layer_num,
                           const galois::graphs::GNNGraph& graph,
                           const GNNLayerDimensions& dimensions,
                           const GNNConfig& config)
    : layer_number_(layer_num), graph_(graph), layer_dimensions_(dimensions),
      config_(config) {
  if (config_.allocate_weights) {
    // TODO some of this does not need alloc if not used
    // dropout allocation; dropout is same as input
    dropout_mask_.resize(layer_dimensions_.input_rows *
                         layer_dimensions_.input_columns);
    // allocate memory based on layer dimensions
    size_t num_weight_elements =
        layer_dimensions_.input_columns * layer_dimensions_.output_columns;
    layer_weights_.resize(num_weight_elements);
    layer_weight_gradients_.resize(num_weight_elements, 0);
    // init weights randomly with a parallel loop
    RandomInitVector(&layer_weights_);
  }

  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;
  forward_output_matrix_.resize(num_output_elements, 0);
  backward_output_matrix_.resize(
      layer_dimensions_.input_rows * layer_dimensions_.input_columns, 0);
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

void galois::GNNLayer::DoDropout(std::vector<GNNFloat>* output_matrix) {
  // XXX fix droptout, should use inputs not weights
  size_t num_weights = layer_weights_.size();
  // determine which weights to drop
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), num_weights),
      [&](size_t i) {
        dropout_mask_[i] = dropout_rng_.DoBernoulli(config_.dropout_rate);
      },
      galois::loopname("LayerDropoutRNG"));

  // create new matrix with non-dropped weights + some scaling
  // TODO scaling?
  GNNFloat scale = 1. / (1. - config_.dropout_rate);
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), num_weights),
      [&](size_t i) {
        (*output_matrix)[i] =
            layer_weights_[i] * static_cast<GNNFloat>(dropout_mask_[i]) * scale;
      },
      galois::loopname("LayerDropout"));
}

void galois::GNNLayer::DoDropoutDerivative() {
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

void galois::GNNLayer::ActivationDerivative(std::vector<GNNFloat>* gradient) {
  // TODO only does relu at the moment; should check user specified activation
  // and act accordingly
  // XXX
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

#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/SoftmaxLayer.h"

const std::vector<galois::GNNFloat>& galois::SoftmaxLayer::ForwardPhase(
    const std::vector<galois::GNNFloat>& input_embeddings) {
  input_loss_.assign(input_loss_.size(), 0.0);
  forward_output_matrix_.assign(forward_output_matrix_.size(), 0.0);
  const size_t feature_length = layer_dimensions_.input_columns;

  galois::do_all(
      galois::iterate(graph_.begin(), graph_.end()),
      [&](const unsigned i) {
        if (graph_.IsValidForPhase(i, layer_phase_)) {
          // do softmax
          GNNSoftmax(feature_length, &input_embeddings[feature_length * i],
                     &forward_output_matrix_[feature_length * i]);

          // create ground truth vector for this LID
          std::vector<GNNFloat>* ground_truth_vec =
              ground_truth_vectors_.getLocal();
          assert(ground_truth_vec->size() == feature_length);
          ground_truth_vec->assign(ground_truth_vec->size(), 0.0);
          // single class label is an index; set the correct one
          (*ground_truth_vec)[static_cast<size_t>(
              graph_.GetSingleClassLabel(i))] = 1.0;

          // calculate loss for this LID (note not all i will be filled)
          input_loss_[i] =
              GNNCrossEntropy(feature_length, ground_truth_vec->data(),
                              &forward_output_matrix_[feature_length * i]);
        }
      },
      // TODO chunk size?
      // steal on as some threads may have nothing to work on
      galois::steal(), galois::loopname("SoftmaxForward"));

  return forward_output_matrix_;
}

std::vector<galois::GNNFloat>*
galois::SoftmaxLayer::BackwardPhase(const std::vector<galois::GNNFloat>&,
                                    std::vector<galois::GNNFloat>*) {
  const size_t feature_length = layer_dimensions_.input_columns;

  galois::do_all(
      galois::iterate(graph_.begin(), graph_.end()),
      [&](const unsigned i) {
        if (graph_.IsValidForPhase(i, layer_phase_)) {
          // create ground truth vector for this LID
          // TODO maybe make this part of the graph class instead of recreating
          // every time
          std::vector<GNNFloat>* ground_truth_vec =
              ground_truth_vectors_.getLocal();
          assert(ground_truth_vec->size() == feature_length);
          ground_truth_vec->assign(ground_truth_vec->size(), 0.0);
          // single class label is an index; set the correct one
          (*ground_truth_vec)[static_cast<size_t>(
              graph_.GetSingleClassLabel(i))] = 1.0;

          // derivative cross entropy into norm grad
          std::vector<GNNFloat>* norm_gradient =
              norm_gradient_vectors_.getLocal();
          GNNCrossEntropyDerivative(
              feature_length, ground_truth_vec->data(),
              &(forward_output_matrix_[i * feature_length]),
              norm_gradient->data());

          // use norm grad with softmax deritave, save and return
          std::vector<GNNFloat>* softmax_temp =
              softmax_temp_vectors_.getLocal();
          GNNSoftmaxDerivative(feature_length,
                               &(forward_output_matrix_[i * feature_length]),
                               norm_gradient->data(), softmax_temp->data(),
                               &(backward_output_matrix_[i * feature_length]));
        }
      },
      // TODO chunk size?
      // steal on as some threads may have nothing to work on
      galois::steal(), galois::loopname("SoftmaxBackward"));

  return &backward_output_matrix_;
}

// TODO function for getting loss

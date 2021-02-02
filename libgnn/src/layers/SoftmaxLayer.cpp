#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/SoftmaxLayer.h"

const galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::ForwardPhaseCPU(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
  input_loss_.assign(input_loss_.size(), 0.0);
  forward_output_matrix_.assign(forward_output_matrix_.size(), 0.0);
  const size_t feature_length = layer_dimensions_.input_columns;
  // TODO(loc) once needed for accuracy debugging, print out loss

  galois::do_all(
      galois::iterate(graph_.begin_owned(), graph_.end_owned()),
      [&](const unsigned i) {
        if (IsSampledLayer()) {
          if (layer_phase_ == GNNPhase::kTrain && !graph_.IsInSampledGraph(i))
            return;
        }

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

const galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::ForwardPhase(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
#ifndef GALOIS_ENABLE_GPU
  return ForwardPhaseCPU(input_embeddings);
#else
  gpu_object_.ForwardPhaseGPU(
      layer_phase_, graph_.size(), layer_dimensions_.input_columns,
      input_embeddings.data(), p_forward_output_matrix_.data());
  return p_forward_output_matrix_;
#endif
}

galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::BackwardPhaseCPU() {
  const size_t feature_length = layer_dimensions_.input_columns;

  // zero out output
  backward_output_matrix_.assign(backward_output_matrix_.size(), 0);

  galois::do_all(
      galois::iterate(graph_.begin_owned(), graph_.end_owned()),
      [&](const unsigned i) {
        if (graph_.IsValidForPhase(i, layer_phase_)) {
          if (IsSampledLayer()) {
            if (layer_phase_ == GNNPhase::kTrain && !graph_.IsInSampledGraph(i))
              return;
          }

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

  return PointerWithSize(backward_output_matrix_);
}

galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::BackwardPhase(const PointerWithSize<galois::GNNFloat>,
                                    PointerWithSize<galois::GNNFloat>*) {
#ifndef GALOIS_ENABLE_GPU
  return BackwardPhaseCPU();
#else
  gpu_object_.BackwardPhaseGPU(
      layer_phase_, graph_.size(), layer_dimensions_.input_columns,
      p_forward_output_matrix_.data(), p_backward_output_matrix_.data());
  return p_backward_output_matrix_;
#endif
}

// TODO function for getting loss

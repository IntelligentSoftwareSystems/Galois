#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/SoftmaxLayer.h"

const galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::ForwardPhaseCPU(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
  galois::StatTimer timer("SoftmaxForward", "SoftmaxLayer");
  TimerStart(&timer);

  // note: p_backward == input_embeddings
  input_loss_.assign(input_loss_.size(), 0.0);
  const size_t feature_length = layer_dimensions_.input_columns;
#ifndef NDEBUG
  galois::DGAccumulator<GNNFloat> loss_accum;
  galois::DGAccumulator<size_t> handled;
  loss_accum.reset();
  handled.reset();
#endif

  galois::do_all(
      galois::iterate(graph_.begin(), graph_.end()),
      [&](const unsigned i) {
        if (IsSampledLayer()) {
          if (layer_phase_ == GNNPhase::kTrain && !graph_.IsInSampledGraph(i)) {
            // XXX
            VectorZero(feature_length,
                       &p_backward_output_matrix_[i * feature_length]);
            return;
          }
        }

        if (graph_.IsValidForPhase(i, layer_phase_)) {
          // do softmax
          GNNSoftmax(feature_length, &input_embeddings[feature_length * i],
                     &p_backward_output_matrix_[feature_length * i]);
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
                              &p_backward_output_matrix_[feature_length * i]);
#ifndef NDEBUG
          loss_accum += input_loss_[i];
          handled += 1;
#endif
        } else {
          VectorZero(feature_length,
                     &p_backward_output_matrix_[i * feature_length]);
        }
      },
      // TODO chunk size?
      // steal on as some threads may have nothing to work on
      galois::steal(), galois::loopname("SoftmaxForward"));
#ifndef NDEBUG
  GNNFloat reduced_loss = loss_accum.reduce();
  size_t t              = handled.reduce();
  galois::gPrint("Loss is ", reduced_loss / t, " ", reduced_loss, " ", t, "\n");
#endif

  TimerStop(&timer);
  return p_backward_output_matrix_;
}

const galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::ForwardPhase(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.ForwardPhaseGPU(
        layer_phase_, graph_.size(), layer_dimensions_.input_columns,
        input_embeddings.data(), p_backward_output_matrix_.data());
    return p_backward_output_matrix_;
  }
#endif
  return ForwardPhaseCPU(input_embeddings);
}

galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::BackwardPhaseCPU() {
  galois::StatTimer timer("SoftmaxForward", "SoftmaxLayer");
  TimerStart(&timer);

  const size_t feature_length = layer_dimensions_.input_columns;

  galois::do_all(
      galois::iterate(graph_.begin(), graph_.end()),
      [&](const unsigned node) {
        if (graph_.IsValidForPhase(node, layer_phase_)) {
          if (IsSampledLayer()) {
            if (layer_phase_ == GNNPhase::kTrain &&
                !graph_.IsInSampledGraph(node))
              return;
          }

          size_t correct = graph_.GetSingleClassLabel(node);
          // See here for explanation for why this works
          // https://gombru.github.io/2018/05/23/cross_entropy_loss/
          // Derivation of full combined derivative isn't there, but some
          // emperical inspection tells me this is likely correct
          // TODO(loc) work it out myself
          for (size_t idx = 0; idx < feature_length; idx++) {
            if (idx == correct) {
              // positive class
              p_backward_output_matrix_[node * feature_length + idx] =
                  p_backward_output_matrix_[node * feature_length + idx] - 1;
            } else {
              // negative class
              p_backward_output_matrix_[node * feature_length + idx] =
                  p_backward_output_matrix_[node * feature_length + idx];
            }
          }
        }
      },
      galois::steal(), galois::loopname("SoftmaxBackward"));

  TimerStop(&timer);

  return p_backward_output_matrix_;
}

galois::PointerWithSize<galois::GNNFloat>
galois::SoftmaxLayer::BackwardPhase(PointerWithSize<galois::GNNFloat>,
                                    PointerWithSize<galois::GNNFloat>*) {
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.BackwardPhaseGPU(
        layer_phase_, graph_.size(), layer_dimensions_.input_columns,
        p_backward_output_matrix_.data(), p_backward_output_matrix_.data());
    return p_backward_output_matrix_;
  }
#endif
  return BackwardPhaseCPU();
}

// TODO function for getting loss

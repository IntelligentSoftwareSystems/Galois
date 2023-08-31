#pragma once
#include "galois/layers/GNNLayer.h"
#include "galois/GNNMath.h"

#include <math.h>

// TODO(loc) GPU support

namespace galois {

//! Sigmoid layer: applies sigmoid function element wise to each element of the
//! input.
//! Meant for use with *multi-class* labels.
template <typename VTy, typename ETy>
class SigmoidLayer : public GNNLayer<VTy, ETy> {
public:
  SigmoidLayer(size_t layer_num,
               const galois::graphs::GNNGraph<VTy, ETy>& graph,
               PointerWithSize<GNNFloat>* backward_output_matrix,
               const GNNLayerDimensions& dimensions)
      : GNNLayer<VTy, ETy>(layer_num, graph, backward_output_matrix, dimensions,
                           GNNLayerConfig{.allocate_weights = false}),
        input_loss_(dimensions.input_rows),
        norm_gradient_vectors_(dimensions.input_columns) {
    this->output_layer_type_ = galois::GNNOutputLayerType::kSigmoid;
    // input/output columns must be equivalent
    GALOIS_LOG_ASSERT(dimensions.input_columns == dimensions.output_columns);
    // output needs to match number of possible classes
    GALOIS_LOG_ASSERT(dimensions.input_columns == graph.GetNumLabelClasses());
  }

  //! Normalizes all elements by applying sigmoid to all of them
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final {
#ifdef GALOIS_ENABLE_GPU
    // TODO(loc) when GPU needs it
    printf("%p\n", input_embeddings.data());
    return p_layer_weights_;
#else
    return ForwardPhaseCPU(input_embeddings);
#endif
  }

  //! Get gradients to fix distribution such that it leans more towards
  //! multiclass ground truth.
  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat>,
                PointerWithSize<galois::GNNFloat>*) final {
#ifdef GALOIS_ENABLE_GPU
    // TODO(loc) when GPU needs it
    return p_layer_weights_;
#else
    return BackwardPhaseCPU();
#endif
  }

private:
  const PointerWithSize<galois::GNNFloat>
  ForwardPhaseCPU(const PointerWithSize<galois::GNNFloat> input_embeddings) {
    galois::gWarn(
        "Sigmoid layer has not been kept up to date; do not use unless sure"
        " it works with new changes");

    input_loss_.assign(input_loss_.size(), 0.0);
    this->forward_output_matrix_.assign(this->forward_output_matrix_.size(),
                                        0.0);
    const size_t feature_length = this->layer_dimensions_.input_columns;
    this->node_count_.reset();
    this->float_accumulator_.reset();

    galois::do_all(
        galois::iterate(this->graph_.begin(), this->graph_.end()),
        [&](const unsigned local_node) {
          if (this->graph_.IsValidForPhase(local_node, this->layer_phase_)) {
            if (this->IsSampledLayer()) {
              if (this->layer_phase_ == GNNPhase::kTrain &&
                  !this->graph_.IsInSampledGraph(local_node))
                return;
            }

            this->node_count_ += 1;

            size_t node_offset = feature_length * local_node;
            // sigmoid the values for this node
            for (unsigned index = 0; index < feature_length; index++) {
              // splitting in half is done for numerical stability of log
              if (input_embeddings[node_offset + index] >= 0) {
                this->forward_output_matrix_[node_offset + index] =
                    1.0 / (1.0 + expf(-input_embeddings[node_offset + index]));
              } else {
                this->forward_output_matrix_[node_offset + index] =
                    expf(input_embeddings[node_offset + index]) /
                    (1.0 + expf(input_embeddings[node_offset + index]));
              }
            }

            input_loss_[local_node] = GNNCrossEntropy(
                feature_length, this->graph_.GetMultiClassLabel(local_node),
                &this->forward_output_matrix_[node_offset]);
            // TODO(loc) normalize the loss
            this->float_accumulator_ += input_loss_[local_node];
          }
        },
        galois::steal(), galois::loopname("SigmoidForward"));

    galois::gPrint(
        "Average loss is ",
        this->float_accumulator_.reduce() / this->node_count_.reduce(), "\n");
    return this->forward_output_matrix_;
  }

  PointerWithSize<galois::GNNFloat> BackwardPhaseCPU() {
    const size_t feature_length = this->layer_dimensions_.input_columns;
    galois::do_all(
        galois::iterate(size_t{0}, this->p_backward_output_matrix_.size()),
        [&](size_t i) { this->p_backward_output_matrix_[i] = 0; });

    galois::do_all(
        galois::iterate(this->graph_.begin(), this->graph_.end()),
        [&](const unsigned local_node) {
          if (this->graph_.IsValidForPhase(local_node, this->layer_phase_)) {
            if (this->IsSampledLayer()) {
              if (this->layer_phase_ == GNNPhase::kTrain &&
                  !this->graph_.IsInSampledGraph(local_node))
                return;
            }

            // derivative cross entropy into norm grad
            const GNNLabel* ground_truth =
                this->graph_.GetMultiClassLabel(local_node);
            size_t node_offset = feature_length * local_node;
            // sigmoid-cross-entropy derivative: turns out all it is is simple
            // subtraction
            for (unsigned index = 0; index < feature_length; index++) {
              this->p_backward_output_matrix_[node_offset + index] =
                  this->forward_output_matrix_[node_offset + index] -
                  ground_truth[index];
            }
          }
        },
        galois::steal(), galois::loopname("SigmoidBackward"));

    return this->p_backward_output_matrix_;
  }

  //! Loss for each row of the input
  std::vector<GNNFloat> input_loss_;
  //! Each thread gets storage to allocate the gradients during backward
  //! prop; each is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      norm_gradient_vectors_;
};

} // namespace galois

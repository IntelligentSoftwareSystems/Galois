#pragma once
#include "galois/layers/GNNLayer.h"
#include "galois/GNNMath.h"

// XXX(hc): We don't have GPU ReLU implementation.

// TODO(hc): All intermediate layers in Galois-GNN have internal ReLU
// layer. So, this is not yet being used.
// BUT, I would like to leave this for the future.

namespace galois {

//! ReLU layer: takes each row of the input matrix and sets 0 to elements < 0 in
//! a row. Currently this only works with **single class* labels and is coded as
//! such.
template <typename VTy, typename ETy>
class ReLULayer : public GNNLayer<VTy, ETy> {
public:
  ReLULayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions)
      : ReLULayer<VTy, ETy>(
            layer_num, graph, backward_output_matrix, dimensions,
            GNNLayerConfig{.allocate_weights = false, .disable_output = true}) {
  }

  ReLULayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
      : GNNLayer<VTy, ETy>(layer_num, graph, backward_output_matrix, dimensions,
                           config) {
    this->layer_type_ = galois::GNNLayerType::kReLU;
    GALOIS_LOG_ASSERT(dimensions.input_columns == dimensions.output_columns);
    GALOIS_LOG_VERBOSE("ReLU initialized");
  }

  //! Perform max(0, input) to each row of input
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final {
    return ForwardPhaseCPU(input_embeddings);
  }

  const PointerWithSize<galois::GNNFloat>
  ForwardPhaseCPU(const PointerWithSize<galois::GNNFloat> input_embeddings) {
    galois::StatTimer Timer("ReLULayer", "ReLULayer");
    this->TimerStart(&Timer);

    // note: p_backward == input_embeddings
    const size_t feature_length = this->layer_dimensions_.input_columns;

    galois::do_all(
        galois::iterate(size_t{0}, this->layer_dimensions_.input_rows),
        [&](const unsigned row) {
          if (this->IsSampledLayer()) {
            if ((this->layer_phase_ == GNNPhase::kTrain ||
                 this->layer_phase_ == GNNPhase::kBatch) &&
                !this->graph_.IsInSampledGraphSubgraph(row)) {
              return;
            }
          }

          if (this->graph_.IsValidForPhase(row, this->layer_phase_)) {
            size_t row_offset = row * feature_length;
            for (size_t row_index = row_offset;
                 row_index < (row_offset + feature_length); row_index++) {
              // TODO(hc): SHAD uses inplace update but Galois-GNN uses
              // separate vector for outputs.
              // Revisit this if there is performance differences.
              this->forward_output_matrix_[row_index] =
                  std::max(float{0}, input_embeddings[row_index]);
            }
          }
        },
        // TODO chunk size?
        // steal on as some threads may have nothing to work on
        // galois::steal(), galois::loopname("ReLUForward"));
        galois::steal());
    this->TimerStop(&Timer);
    return this->forward_output_matrix_;
  }

  PointerWithSize<galois::GNNFloat>
  BackwardPhaseCPU(PointerWithSize<galois::GNNFloat> prev_layer_input,
                   PointerWithSize<galois::GNNFloat>* input_gradients) {
    galois::StatTimer Timer("ReLUBackward", "ReLULayer");
    this->TimerStart(&Timer);

    const size_t feature_length = this->layer_dimensions_.input_columns;

    galois::do_all(
        galois::iterate(size_t{0}, this->layer_dimensions_.input_rows),
        [&](const unsigned row) {
          if (this->IsSampledLayer()) {
            if (this->layer_phase_ == GNNPhase::kTrain &&
                !this->graph_.IsInSampledGraphSubgraph(row))
              return;
          }
          // Even though ReLU is non-differentiable at 0,
          // PyTorch's ReLU returns 0 for the derivative of 0.
          if (this->graph_.IsValidForPhase(row, this->layer_phase_)) {
            size_t row_offset = row * feature_length;
            for (size_t row_index = row_offset;
                 row_index < (row_offset + feature_length); row_index++) {
              this->p_backward_output_matrix_[row_index] =
                  (prev_layer_input[row_index] > 0 ? 1 : 0) *
                  (*input_gradients)[row_index];
            }
          }
        },
        galois::steal(), galois::loopname("ReLUBackward"));

    this->TimerStop(&Timer);

    return this->p_backward_output_matrix_;
  }

  //! Get gradients to fix distribution such that it leans more towards single
  //! class ground truth.
  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradients) final {
    return BackwardPhaseCPU(prev_layer_input, input_gradients);
  }
};

} // namespace galois

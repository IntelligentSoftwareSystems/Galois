#pragma once
#include "galois/layers/GNNLayer.h"

#ifdef GALOIS_ENABLE_GPU
// TODO(loc/hochan)
#endif

namespace galois {

//! Applies L2 norm to rows of the input
template <typename VTy, typename ETy>
class L2NormLayer : public GNNLayer<VTy, ETy> {
public:
  L2NormLayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
              PointerWithSize<GNNFloat>* backward_output_matrix,
              const GNNLayerDimensions& dimensions)
      : L2NormLayer(layer_num, graph, backward_output_matrix, dimensions,
                    GNNLayerConfig{.allocate_weights = false}) {}
  L2NormLayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
              PointerWithSize<GNNFloat>* backward_output_matrix,
              const GNNLayerDimensions& dimensions,
              const GNNLayerConfig& config)
      : GNNLayer<VTy, ETy>(layer_num, graph, backward_output_matrix, dimensions,
                           config) {
    this->layer_type_ = galois::GNNLayerType::kL2Norm;
    // input/output columns must be equivalent in a softmax
    GALOIS_LOG_ASSERT(dimensions.input_columns == dimensions.output_columns);
    GALOIS_LOG_VERBOSE("L2 norm initialized");
  }

  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) {
#ifdef GALOIS_ENABLE_GPU
    // TODO
#endif
    GALOIS_LOG_FATAL(
        "L2 Layer has not been kept up to date for months; do not use");
    return ForwardPhaseCPU(input_embeddings);
  }

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) {
#ifdef GALOIS_ENABLE_GPU
    // TODO
#endif
    return BackwardPhaseCPU(prev_layer_input, input_gradient);
  }

private:
  const PointerWithSize<galois::GNNFloat>
  ForwardPhaseCPU(const PointerWithSize<galois::GNNFloat> input_embeddings) {
    this->forward_output_matrix_.assign(this->forward_output_matrix_.size(),
                                        0.0);
    // for each row, get square root of squared sums then normalize
    const size_t feature_length = this->layer_dimensions_.input_columns;
    // TODO(loc) make sure this works in distributed setting as well
    galois::do_all(
        galois::iterate(this->graph_.begin_owned(), this->graph_.end_owned()),
        [&](const unsigned row) {
          if (this->IsSampledLayer()) {
            if (this->layer_phase_ == GNNPhase::kTrain &&
                !this->graph_.IsInSampledGraph(row))
              return;
          }

          if (this->graph_.IsValidForPhase(row, this->layer_phase_)) {
            size_t row_offset        = row * feature_length;
            float running_square_sum = 0.0;
            // get square sums
            for (size_t row_index = row_offset;
                 row_index < (row_offset + feature_length); row_index++) {
              running_square_sum += std::pow(input_embeddings[row_index], 2);
            }

            // make sure running sum isn't too small
            running_square_sum =
                (running_square_sum < 1.0e-12) ? 10e-12 : running_square_sum;

            // sqrt of sums, then divide row by it
            float sqrt_squares = std::pow(running_square_sum, 0.5);
            for (size_t row_index = row_offset;
                 row_index < (row_offset + feature_length); row_index++) {
              this->forward_output_matrix_[row_index] =
                  input_embeddings[row_index] / sqrt_squares;
            }
          }
        },
        galois::loopname("L2ForwardNormalization"));

    return this->forward_output_matrix_;
  }

  PointerWithSize<galois::GNNFloat>
  BackwardPhaseCPU(PointerWithSize<galois::GNNFloat> prev_layer_input,
                   PointerWithSize<galois::GNNFloat>* input_gradient) {
    galois::do_all(
        galois::iterate(size_t{0}, this->p_backward_output_matrix_.size()),
        [&](size_t i) { this->p_backward_output_matrix_[i] = 0; });
    const size_t feature_length = this->layer_dimensions_.input_columns;

    // derivative of some x_1 is sum of gradient w.r.t. x_1 for all elements of
    // the row (since l2 norm affects entire row)
    // The math itself can be derived using quotient/chain rule on each element
    // of the normalized row
    galois::do_all(
        galois::iterate(this->graph_.begin_owned(), this->graph_.end_owned()),
        [&](const unsigned row) {
          if (this->IsSampledLayer()) {
            if (this->layer_phase_ == GNNPhase::kTrain &&
                !this->graph_.IsInSampledGraph(row))
              return;
          }

          if (this->graph_.IsValidForPhase(row, this->layer_phase_)) {
            size_t row_offset = row * feature_length;
            // note: if you work this out on paper it turns out that terms that
            // seem extra in the way this is calculated below simply get
            // canceled out, so this ends up working out This implementation is
            // taken from the IPDPS GraphSAINT implementation: I (loc) have
            // confirmed the math checks out
            float running_square_sum = 0.0;
            float mult_with_input    = 0.0;

            // get square sums
            for (size_t row_index = row_offset;
                 row_index < (row_offset + feature_length); row_index++) {
              running_square_sum += std::pow(prev_layer_input[row_index], 2);
              // gradient multiplied with corresponding input; subtraction
              // because derivative math ends up working out that way
              mult_with_input -=
                  prev_layer_input[row_index] * (*input_gradient)[row_index];
            }
            running_square_sum =
                (running_square_sum < 1.0e-12) ? 10e-12 : running_square_sum;
            assert(running_square_sum != 0.0);

            // denominator for all gradients is just the square sum to the
            // -3/2'd power since this is -, all we have to do is multiply it
            // later rather than divide
            float denominator = std::pow(running_square_sum, -1.5);
            assert(denominator != 0.0);

            for (size_t row_index = row_offset;
                 row_index < (row_offset + feature_length); row_index++) {
              this->p_backward_output_matrix_[row_index] =
                  denominator *
                  (prev_layer_input[row_index] * mult_with_input +
                   (*input_gradient)[row_index] * running_square_sum);
            }
          }
        },
        galois::loopname("L2Backward"));

    return this->p_backward_output_matrix_;
  }

  //! No op
  void OptimizeLayer(BaseOptimizer*, size_t) { return; };

#ifdef GALOIS_ENABLE_GPU
  // TODO(loc/hochan)
#endif
};

} // namespace galois

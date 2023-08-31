#pragma once
#include "galois/layers/GNNLayer.h"
#include "galois/GNNMath.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/layers/SoftmaxLayer.cuh"
#endif

namespace galois {

//! Softmax layer: takes each row of the input matrix and creates a probability
//! distribution based on the magnitude of elements in each row.
//! Currently this only works with **single class* labels and is coded as such.
template <typename VTy, typename ETy>
class SoftmaxLayer : public GNNLayer<VTy, ETy> {
public:
  SoftmaxLayer(size_t layer_num,
               const galois::graphs::GNNGraph<VTy, ETy>& graph,
               PointerWithSize<GNNFloat>* backward_output_matrix,
               const GNNLayerDimensions& dimensions)
      : GNNLayer<VTy, ETy>(
            layer_num, graph, backward_output_matrix, dimensions,
            GNNLayerConfig{.allocate_weights = false, .disable_output = true}),
#ifdef GALOIS_ENABLE_GPU
        gpu_object_(graph.GetGPUGraph()),
#endif
        input_loss_(dimensions.input_rows),
        ground_truth_vectors_(dimensions.input_columns),
        norm_gradient_vectors_(dimensions.input_columns),
        softmax_temp_vectors_(dimensions.input_columns)

  {
    this->output_layer_type_ = galois::GNNOutputLayerType::kSoftmax;
    // input/output columns must be equivalent in a softmax
    GALOIS_LOG_ASSERT(dimensions.input_columns == dimensions.output_columns);
    // output needs to match number of possible classes
    GALOIS_LOG_ASSERT(dimensions.input_columns == graph.GetNumLabelClasses());
  }

  const PointerWithSize<galois::GNNFloat>
  ForwardPhaseCPU(const PointerWithSize<galois::GNNFloat> input_embeddings) {
    galois::StatTimer Timer("SoftmaxForward", "SoftmaxLayer");
    this->TimerStart(&Timer);

    // note: p_backward == input_embeddings
    input_loss_.assign(input_loss_.size(), 0.0);
    const size_t feature_length = this->layer_dimensions_.input_columns;
#ifndef NDEBUG
    galois::DGAccumulator<GNNFloat> loss_accum;
    galois::DGAccumulator<size_t> handled;
    loss_accum.reset();
    handled.reset();
#endif

    galois::do_all(
        galois::iterate(size_t{0}, this->layer_dimensions_.input_rows),
        [&](const unsigned i) {
          if (this->IsSampledLayer()) {
            if ((this->layer_phase_ == GNNPhase::kTrain ||
                 this->layer_phase_ == GNNPhase::kBatch) &&
                !this->graph_.IsInSampledGraphSubgraph(i)) {
              // XXX
              VectorZero(feature_length,
                         &this->p_backward_output_matrix_[i * feature_length]);
              return;
            }
          }

          // do softmax
          GNNSoftmax(feature_length, &input_embeddings[feature_length * i],
                     &this->p_backward_output_matrix_[feature_length * i]);
          // create ground truth vector for this LID
          std::vector<GNNFloat>* ground_truth_vec =
              ground_truth_vectors_.getLocal();
          assert(ground_truth_vec->size() == feature_length);
          ground_truth_vec->assign(ground_truth_vec->size(), 0.0);
          // single class label is an index; set the correct one
          (*ground_truth_vec)[static_cast<size_t>(
              this->graph_.GetSingleClassLabel(i))] = 1.0;

          // calculate loss for this LID (note not all i will be filled)
          input_loss_[i] = GNNCrossEntropy(
              feature_length, ground_truth_vec->data(),
              &this->p_backward_output_matrix_[feature_length * i]);
#ifndef NDEBUG
          loss_accum += input_loss_[i];
          handled += 1;
#endif
        },
        // TODO chunk size?
        // steal on as some threads may have nothing to work on
        // galois::steal(), galois::loopname("SoftmaxForward"));
        galois::steal());
#ifndef NDEBUG
    GNNFloat reduced_loss = loss_accum.reduce();
    size_t t              = handled.reduce();
    galois::gPrint("Loss is ", reduced_loss / t, " ", reduced_loss, " ", t,
                   "\n");
#endif

    this->TimerStop(&Timer);
    return this->p_backward_output_matrix_;
  }

  //! Creates probability distribution of each row of input
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final {
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.ForwardPhaseGPU(this->layer_phase_, this->graph_.size(),
                                  this->layer_dimensions_.input_columns,
                                  input_embeddings.data(),
                                  this->p_backward_output_matrix_.data());
      return this->p_backward_output_matrix_;
    }
#endif
    return ForwardPhaseCPU(input_embeddings);
  }

  PointerWithSize<galois::GNNFloat> BackwardPhaseCPU() {
    galois::StatTimer Timer("SoftmaxBackward", "SoftmaxLayer");
    this->TimerStart(&Timer);

    const size_t feature_length = this->layer_dimensions_.input_columns;

    galois::do_all(
        galois::iterate(size_t{0}, this->layer_dimensions_.input_rows),
        [&](const unsigned node) {
          if (this->IsSampledLayer()) {
            if (this->layer_phase_ == GNNPhase::kTrain &&
                !this->graph_.IsInSampledGraphSubgraph(node))
              return;
          }

          size_t correct = this->graph_.GetSingleClassLabel(node);
          // See here for explanation for why this works
          // https://gombru.github.io/2018/05/23/cross_entropy_loss/
          // Derivation of full combined derivative isn't there, but some
          // emperical inspection tells me this is likely correct
          // TODO(loc) work it out myself
          for (size_t idx = 0; idx < feature_length; idx++) {
            if (idx == correct) {
              // positive class
              this->p_backward_output_matrix_[node * feature_length + idx] =
                  this->p_backward_output_matrix_[node * feature_length + idx] -
                  1;
            } else {
              // negative class
              this->p_backward_output_matrix_[node * feature_length + idx] =
                  this->p_backward_output_matrix_[node * feature_length + idx];
            }
          }
        },
        galois::steal(), galois::loopname("SoftmaxBackward"));

    this->TimerStop(&Timer);

    return this->p_backward_output_matrix_;
  }

  //! Get gradients to fix distribution such that it leans more towards single
  //! class ground truth.
  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat>,
                PointerWithSize<galois::GNNFloat>*) final {
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.BackwardPhaseGPU(this->layer_phase_, this->graph_.size(),
                                   this->layer_dimensions_.input_columns,
                                   this->p_backward_output_matrix_.data(),
                                   this->p_backward_output_matrix_.data());
      return this->p_backward_output_matrix_;
    }
#endif
    return BackwardPhaseCPU();
  }

  void ResizeRows(size_t new_row_count) {
    this->layer_dimensions_.input_rows  = new_row_count;
    this->layer_dimensions_.output_rows = new_row_count;
    // no output resize
    if (input_loss_.size() < new_row_count) {
      input_loss_.resize(new_row_count * 1.02);
    }
  }

  void ResizeInputOutputRows(size_t in, size_t out) {
    assert(in == out);
    this->layer_dimensions_.input_rows  = in;
    this->layer_dimensions_.output_rows = out;
    // no output resize
    if (input_loss_.size() < in) {
      input_loss_.resize(in * 1.02);
    }
  }

private:
#ifdef GALOIS_ENABLE_GPU
  SoftmaxLayerGPU gpu_object_;
#endif

  //! Loss for each row of the input
  std::vector<GNNFloat> input_loss_;
  //! Each thread gets storage to allocate the ground truth vector in during
  //! calculation; each vector is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      ground_truth_vectors_;
  //! Each thread gets storage to allocate the gradients during backward
  //! prop; each is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      norm_gradient_vectors_;
  //! Each thread gets storage for a temporary vector used during softmax
  //! derivative calculation; each is the size of a feature vector
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      softmax_temp_vectors_;
};

} // namespace galois

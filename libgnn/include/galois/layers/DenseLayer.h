
#pragma once
#include "galois/layers/GNNLayer.h"
#include "galois/Logging.h"
#include "galois/GNNMath.h"

namespace galois {

//! Just does a linear xform with no convolution over graph
template <typename VTy, typename ETy>
class DenseLayer : public GNNLayer<VTy, ETy> {
public:
  //! Initializes the variables of the base class and also allocates additional
  //! memory for temporary matrices. Also initializes sync substrate for the
  //! weight matrix
  DenseLayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
             PointerWithSize<GNNFloat>* backward_output_matrix,
             const GNNLayerDimensions& layer_dimensions,
             const GNNLayerConfig& config)
      : GNNLayer<VTy, ETy>(layer_num, graph, backward_output_matrix,
                           layer_dimensions, config),
        input_column_intermediates_(layer_dimensions.input_columns),
        output_column_intermediates_(layer_dimensions.output_columns) {
    // TODO Need to make sure that layer knows about forward/backward matrix
    // sharing (e.g., overwriting previously used input to save space)
    GALOIS_LOG_FATAL(
        "This layer has not been kept up to date; do not use until "
        "sure it's been updated");
    size_t num_input_elements = this->layer_dimensions_.input_rows *
                                this->layer_dimensions_.input_columns;
    in_temp_1_.resize(num_input_elements, 0);
    size_t num_output_elements = this->layer_dimensions_.input_rows *
                                 this->layer_dimensions_.output_columns;
    GALOIS_LOG_VERBOSE("Output elements {}", num_output_elements);
    this->layer_type_  = galois::GNNLayerType::kDense;
    this->p_in_temp_1_ = PointerWithSize<GNNFloat>(in_temp_1_);
    GALOIS_LOG_VERBOSE("Dense initialized");
  }

  DenseLayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
             PointerWithSize<GNNFloat>* backward_output_matrix,
             const GNNLayerDimensions& dimensions)
      : DenseLayer(layer_num, graph, backward_output_matrix, dimensions,
                   GNNLayerConfig()) {}

  // Parent functions
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final {
    GALOIS_LOG_VERBOSE("Calling forward phase");
    assert(input_embeddings.size() == (this->layer_dimensions_.input_rows *
                                       this->layer_dimensions_.input_columns));
    assert(this->p_in_temp_1_.size() == input_embeddings.size());
    assert(this->p_forward_output_matrix_.size() ==
           (this->layer_dimensions_.input_rows *
            this->layer_dimensions_.output_columns));
    // pointer to input to operate on
    const GNNFloat* input_data = input_embeddings.data();
    // first, dropout
    if (!this->config_.disable_dropout &&
        (this->layer_phase_ == GNNPhase::kTrain)) {
      this->DoDropout(input_embeddings, &this->p_in_temp_1_);
      input_data = this->p_in_temp_1_.data();
    }

    // FW
    UpdateEmbeddings(input_data, this->p_forward_output_matrix_.data());

    if (!this->config_.disable_activation) {
      GALOIS_LOG_VERBOSE("Doing activation");
      this->Activation();
    }

    assert(this->p_forward_output_matrix_.size() ==
           (this->layer_dimensions_.input_rows *
            this->layer_dimensions_.output_columns));
    return this->p_forward_output_matrix_;
  }

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) final {
    assert(this->layer_phase_ == GNNPhase::kTrain);

    // derivative of activation
    if (!this->config_.disable_activation) {
      this->ActivationDerivative(input_gradient);
    }

    if (this->layer_number_ != 0) {
      // derivative for update
      // backout = F'
      UpdateEmbeddingsDerivative(input_gradient->data(),
                                 this->p_backward_output_matrix_.data());
    }

    galois::PointerWithSize<galois::GNNFloat> input_data;
    if (!this->config_.disable_dropout) {
      // dropout result is currently stored in temp 1
      // needs to be used before it gets overwritten
      input_data = this->p_in_temp_1_;
    } else {
      // no dropout = use vanilla input
      input_data = prev_layer_input;
    }

    // W' = F^T (FW)'
    galois::CBlasSGEMM(
        CblasTrans, CblasNoTrans, this->layer_dimensions_.input_columns,
        this->layer_dimensions_.input_rows,
        this->layer_dimensions_.output_columns, input_data.data(),
        input_gradient->data(), this->p_layer_weight_gradients_.data());
    // sync weight gradients; note aggregation sync occurs in the function call
    // already
    this->WeightGradientSyncSum();

    if (!this->config_.disable_dropout && this->layer_number_ != 0) {
      this->DoDropoutDerivative();
    }

    return this->p_backward_output_matrix_;
  }

private:
  // 2 temporaries the size of the forward input; used for dropout and
  // aggregation (if either are required)
  std::vector<GNNFloat> in_temp_1_;
  // Pointer with size versions
  PointerWithSize<GNNFloat> p_in_temp_1_;

  // Each thread has a vector of size # input columns or # output columns for
  // storing intermediate results during aggregation.
  // The one used depeneds on if aggregation occurs before or after the mxm.
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      input_column_intermediates_;
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      output_column_intermediates_;

  //! Do embedding update via mxm with this layer's weights (forward)
  void UpdateEmbeddings(const GNNFloat* node_embeddings, GNNFloat* output) {
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      /* TODO(lhc) implement this
      gpu_object_.UpdateEmbeddingsGPU(
          layer_dimensions_.input_rows, layer_dimensions_.input_columns,
          layer_dimensions_.output_columns, node_embeddings,
          base_gpu_object_.layer_weights(), output);
          */
    } else {
#endif
      // CPU version is just a call into CBlas
      galois::CBlasSGEMM(CblasNoTrans, CblasNoTrans,
                         this->layer_dimensions_.input_rows,
                         this->layer_dimensions_.input_columns,
                         this->layer_dimensions_.output_columns,
                         node_embeddings, this->layer_weights_.data(), output);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  //! Calculate graident via mxm with last layer's gradients (backward)
  void UpdateEmbeddingsDerivative(const GNNFloat* gradients, GNNFloat* output) {
    assert(this->p_layer_weights_.size() ==
           this->layer_dimensions_.input_columns *
               this->layer_dimensions_.output_columns);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      /* TODO(lhc) implement this
      gpu_object_.UpdateEmbeddingsDerivativeGPU(
          layer_dimensions_.input_rows, layer_dimensions_.input_columns,
          layer_dimensions_.output_columns, gradients,
          base_gpu_object_.layer_weights(), output);
          */
    } else {
#endif
      // difference is Trans for B matrix (data) to get z by y (weights is y by
      // z normally); result is x by y
      galois::CBlasSGEMM(CblasNoTrans, CblasTrans,
                         this->layer_dimensions_.input_rows,
                         this->layer_dimensions_.output_columns,
                         this->layer_dimensions_.input_columns, gradients,
                         this->layer_weights_.data(), output);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

#ifdef GALOIS_ENABLE_GPU
  // TODO(hochan/loc) replace with dense gpu object
  // GCNGPUAllocations gpu_object_;
#endif
};

} // namespace galois

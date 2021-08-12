#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/DenseLayer.h"

galois::DenseLayer::DenseLayer(
    size_t layer_num, const galois::graphs::GNNGraph& graph,
    PointerWithSize<GNNFloat>* backward_output_matrix,
    const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
    : GNNLayer(layer_num, graph, backward_output_matrix, dimensions, config),
      input_column_intermediates_(dimensions.input_columns),
      output_column_intermediates_(dimensions.output_columns) {
  // TODO Need to make sure that layer knows about forward/backward matrix
  // sharing (e.g., overwriting previously used input to save space)
  GALOIS_LOG_FATAL("This layer has not been kept up to date; do not use until "
                   "sure it's been updated");
  size_t num_input_elements =
      layer_dimensions_.input_rows * layer_dimensions_.input_columns;
  in_temp_1_.resize(num_input_elements, 0);
  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;
  GALOIS_LOG_VERBOSE("Output elements {}", num_output_elements);
  layer_type_  = galois::GNNLayerType::kDense;
  p_in_temp_1_ = PointerWithSize<GNNFloat>(in_temp_1_);
  GALOIS_LOG_VERBOSE("Dense initialized");
}

const galois::PointerWithSize<galois::GNNFloat>
galois::DenseLayer::ForwardPhase(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
  GALOIS_LOG_VERBOSE("Calling forward phase");
  assert(input_embeddings.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.input_columns));
  assert(p_in_temp_1_.size() == input_embeddings.size());
  assert(p_forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));
  // pointer to input to operate on
  const GNNFloat* input_data = input_embeddings.data();
  // first, dropout
  if (!config_.disable_dropout && (layer_phase_ == GNNPhase::kTrain)) {
    DoDropout(input_embeddings, &p_in_temp_1_);
    input_data = p_in_temp_1_.data();
  }

  // FW
  UpdateEmbeddings(input_data, p_forward_output_matrix_.data());

  if (!config_.disable_activation) {
    GALOIS_LOG_VERBOSE("Doing activation");
    Activation();
  }

  assert(p_forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));
  return p_forward_output_matrix_;
}

galois::PointerWithSize<galois::GNNFloat> galois::DenseLayer::BackwardPhase(
    galois::PointerWithSize<galois::GNNFloat> prev_layer_input,
    galois::PointerWithSize<galois::GNNFloat>* input_gradient) {
  assert(layer_phase_ == GNNPhase::kTrain);

  // derivative of activation
  if (!config_.disable_activation) {
    ActivationDerivative(input_gradient);
  }

  if (layer_number_ != 0) {
    // derivative for update
    // backout = F'
    UpdateEmbeddingsDerivative(input_gradient->data(),
                               p_backward_output_matrix_.data());
  }

  galois::PointerWithSize<galois::GNNFloat> input_data;
  if (!config_.disable_dropout) {
    // dropout result is currently stored in temp 1
    // needs to be used before it gets overwritten
    input_data = p_in_temp_1_;
  } else {
    // no dropout = use vanilla input
    input_data = prev_layer_input;
  }

  // W' = F^T (FW)'
  galois::CBlasSGEMM(CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
                     layer_dimensions_.input_rows,
                     layer_dimensions_.output_columns, input_data.data(),
                     input_gradient->data(), p_layer_weight_gradients_.data());
  // sync weight gradients; note aggregation sync occurs in the function call
  // already
  WeightGradientSyncSum();

  if (!config_.disable_dropout && layer_number_ != 0) {
    DoDropoutDerivative();
  }

  return p_backward_output_matrix_;
}

void galois::DenseLayer::UpdateEmbeddings(const GNNFloat* node_embeddings,
                                          GNNFloat* output) {
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
    galois::CBlasSGEMM(CblasNoTrans, CblasNoTrans, layer_dimensions_.input_rows,
                       layer_dimensions_.input_columns,
                       layer_dimensions_.output_columns, node_embeddings,
                       layer_weights_.data(), output);
#ifdef GALOIS_ENABLE_GPU
  }
#endif
}

void galois::DenseLayer::UpdateEmbeddingsDerivative(const GNNFloat* gradients,
                                                    GNNFloat* output) {
  assert(p_layer_weights_.size() ==
         layer_dimensions_.input_columns * layer_dimensions_.output_columns);
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
    // difference is Trans for B matrix (data) to get z by y (weights is y by z
    // normally); result is x by y
    galois::CBlasSGEMM(CblasNoTrans, CblasTrans, layer_dimensions_.input_rows,
                       layer_dimensions_.output_columns,
                       layer_dimensions_.input_columns, gradients,
                       layer_weights_.data(), output);
#ifdef GALOIS_ENABLE_GPU
  }
#endif
}

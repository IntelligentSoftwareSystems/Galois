#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/GraphConvolutionalLayer.h"

galois::GraphConvolutionalLayer::GraphConvolutionalLayer(
    size_t layer_num, const galois::graphs::GNNGraph& graph,
    const GNNLayerDimensions& dimensions, const GNNConfig& config)
    : GNNLayer(layer_num, graph, dimensions, config),
      input_column_intermediates_(dimensions.input_columns),
      output_column_intermediates_(dimensions.output_columns) {
  size_t num_input_elements =
      layer_dimensions_.input_rows * layer_dimensions_.input_columns;
  in_temp_1_.resize(num_input_elements, 0);
  // TODO temp2 does not need to be initialized in all circumstances
  in_temp_2_.resize(num_input_elements, 0);

  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;
  GALOIS_LOG_VERBOSE("Output elements {}", num_output_elements);
  out_temp_.resize(num_output_elements, 0);
  layer_type_ = galois::GNNLayerType::kGraphConvolutional;
}

const std::vector<galois::GNNFloat>&
galois::GraphConvolutionalLayer::ForwardPhase(
    const std::vector<galois::GNNFloat>& input_embeddings) {
  assert(input_embeddings.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.input_columns));
  assert(in_temp_1_.size() == input_embeddings.size());
  assert(in_temp_2_.size() == input_embeddings.size());
  assert(forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));
  // pointer to input to operate on
  const GNNFloat* input_data = input_embeddings.data();
  // first, dropout
  if (config_.do_dropout && (layer_phase_ == GNNPhase::kTrain)) {
    DoDropout(&in_temp_1_);
    input_data = in_temp_1_.data();
  }

  // flip aggregate/update if dimensions favor it (do less work)
  if (layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    // aggregation and update
    AggregateAll(layer_dimensions_.input_columns, input_data, in_temp_2_.data(),
                 &input_column_intermediates_);
    UpdateEmbeddings(in_temp_2_.data(), forward_output_matrix_.data());
  } else {
    // update to aggregate
    UpdateEmbeddings(input_data, out_temp_.data());
    AggregateAll(layer_dimensions_.output_columns, out_temp_.data(),
                 forward_output_matrix_.data(), &output_column_intermediates_);
  }

  // TODO synchronization of aggregation functions

  // TODO if input columns > output columns do update first then aggregate for
  // efficiency

  if (config_.do_activation) {
    GALOIS_LOG_VERBOSE("Doing activation");
    Activation();
  }

  assert(forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));
  return forward_output_matrix_;
}

std::vector<galois::GNNFloat>* galois::GraphConvolutionalLayer::BackwardPhase(
    const std::vector<galois::GNNFloat>& prev_layer_input,
    std::vector<galois::GNNFloat>* input_gradient) {
  assert(layer_phase_ == GNNPhase::kTrain);
  // derivative of activation
  if (config_.do_activation) {
    ActivationDerivative(input_gradient);
  }

  // derivative of aggregation/update
  // TODO clean up logic here to reduce nesting
  if (layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    if (layer_number_ != 0) {
      // transposed sgemm for derivative; in_temp is output
      assert(input_gradient->size() ==
             layer_dimensions_.input_rows * layer_dimensions_.output_columns);
      assert(in_temp_1_.size() ==
             layer_dimensions_.input_columns * layer_dimensions_.input_rows);
      UpdateEmbeddingsDerivative(input_gradient->data(), in_temp_1_.data());
      // derivative of aggregate is the same due to symmetric graph
      AggregateAll(layer_dimensions_.input_columns, in_temp_1_.data(),
                   backward_output_matrix_.data(),
                   &input_column_intermediates_);
    }
    // weight gradient calculation
    galois::CBlasSGEMM(
        CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
        layer_dimensions_.input_rows, layer_dimensions_.output_columns,
        prev_layer_input.data(), input_gradient->data(),
        layer_weight_gradients_.data());
  } else {
    // aggregate occurs regardless of layer being equal to 0 because it is
    // required in this case for the weight gradient calculation
    AggregateAll(layer_dimensions_.output_columns, input_gradient->data(),
                 out_temp_.data(), &output_column_intermediates_);
    if (layer_number_ != 0) {
      // derivative for update
      UpdateEmbeddingsDerivative(out_temp_.data(),
                                 backward_output_matrix_.data());
    }
    // weight gradient; note the use of the aggregated gradient in out_temp
    galois::CBlasSGEMM(
        CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
        layer_dimensions_.input_rows, layer_dimensions_.output_columns,
        prev_layer_input.data(), out_temp_.data(),
        layer_weight_gradients_.data());
  }

  // TODO sync agg/update

  // TODO sync weights

  if (config_.do_dropout && layer_number_ != 0) {
    DoDropoutDerivative();
  }

  return &backward_output_matrix_;
}

void galois::GraphConvolutionalLayer::AggregateAll(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts) {
  size_t num_nodes = graph_.size();

  galois::do_all(
      galois::iterate(static_cast<size_t>(0), num_nodes),
      [&](size_t src) {
        size_t index_to_src_feature = src * column_length;
        // zero out src feature first
        // TODO can init to self as well
        for (size_t i = 0; i < column_length; i++) {
          aggregate_output[index_to_src_feature + i] = 0;
        }

        GNNFloat source_norm = 0.0;
        if (config_.do_normalization) {
          source_norm = graph_.NormFactor(src);
        }

        // loop through all destinations to grab the feature to aggregate
        for (auto e = graph_.EdgeBegin(src); e != graph_.EdgeEnd(src); e++) {
          size_t dst                  = graph_.EdgeDestination(e);
          size_t index_to_dst_feature = dst * column_length;

          if (config_.do_normalization) {
            GNNFloat norm_scale = source_norm * graph_.NormFactor(dst);
            // scale the value on the destination by the combined norm term
            assert(pts->getLocal()->size() == column_length);
            GNNFloat* intermediate = pts->getLocal()->data();
            for (size_t i = 0; i < column_length; i++) {
              intermediate[i] =
                  norm_scale * node_embeddings[index_to_dst_feature + i];
            }
            // add intermediate instead of original feature
            galois::VectorAdd(
                column_length, &aggregate_output[index_to_src_feature],
                intermediate, &aggregate_output[index_to_src_feature]);
          } else {
            // add dst feature to aggregate output
            galois::VectorAdd(column_length,
                              &aggregate_output[index_to_src_feature],
                              &node_embeddings[index_to_dst_feature],
                              &aggregate_output[index_to_src_feature]);
          }
        }
      },
      galois::steal(), galois::loopname("ConvolutionalAggregateAll"));
}

void galois::GraphConvolutionalLayer::UpdateEmbeddings(
    const GNNFloat* node_embeddings, GNNFloat* output) {
  galois::CBlasSGEMM(CblasNoTrans, CblasNoTrans, layer_dimensions_.input_rows,
                     layer_dimensions_.input_columns,
                     layer_dimensions_.output_columns, node_embeddings,
                     layer_weights_.data(), output);
}

void galois::GraphConvolutionalLayer::UpdateEmbeddingsDerivative(
    const GNNFloat* gradients, GNNFloat* output) {
  assert(layer_weights_.size() ==
         layer_dimensions_.input_columns * layer_dimensions_.output_columns);
  // difference is Trans for B matrix (data) to get z by y (weights is y by z
  // normally); result is x by y
  galois::CBlasSGEMM(CblasNoTrans, CblasTrans, layer_dimensions_.input_rows,
                     layer_dimensions_.output_columns,
                     layer_dimensions_.input_columns, gradients,
                     layer_weights_.data(), output);
}

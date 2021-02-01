#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/GraphConvolutionalLayer.h"

galois::GraphConvolutionalLayer::GraphConvolutionalLayer(
    size_t layer_num, const galois::graphs::GNNGraph& graph,
    const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
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
#ifdef GALOIS_ENABLE_GPU
  gpu_object_.Allocate(num_input_elements, num_output_elements);
#endif

  // init pointers with size
#ifndef GALOIS_ENABLE_GPU
  p_in_temp_1_ = PointerWithSize<GNNFloat>(in_temp_1_);
  p_in_temp_2_ = PointerWithSize<GNNFloat>(in_temp_2_);
  p_out_temp_  = PointerWithSize<GNNFloat>(out_temp_);
#else
  p_in_temp_1_ =
      PointerWithSize<GNNFloat>(gpu_object_.in_temp_1(), in_temp_1_.size());
  p_in_temp_2_ =
      PointerWithSize<GNNFloat>(gpu_object_.in_temp_2(), in_temp_2_.size());
  p_out_temp_ =
      PointerWithSize<GNNFloat>(gpu_object_.out_temp(), out_temp_.size());
#endif
  GALOIS_LOG_VERBOSE("Conv layer initialized");
}

const galois::PointerWithSize<galois::GNNFloat>
galois::GraphConvolutionalLayer::ForwardPhase(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
  GALOIS_LOG_VERBOSE("Calling forward phase");
  assert(input_embeddings.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.input_columns));
  assert(p_in_temp_1_.size() == input_embeddings.size());
  assert(p_in_temp_2_.size() == input_embeddings.size());
  assert(p_forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));
  // pointer to input to operate on
  const GNNFloat* input_data = input_embeddings.data();
  // first, dropout
  if (config_.do_dropout && (layer_phase_ == GNNPhase::kTrain)) {
    DoDropout(input_embeddings, &p_in_temp_1_);
    input_data = p_in_temp_1_.data();
  }

  // flip aggregate/update if dimensions favor it (do less work)
  if (config_.disable_aggregate_after_update ||
      layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    // aggregation and update
    AggregateAll(layer_dimensions_.input_columns, input_data,
                 p_in_temp_2_.data(), &input_column_intermediates_);
    UpdateEmbeddings(p_in_temp_2_.data(), p_forward_output_matrix_.data());
  } else {
    // update to aggregate
    // FW
    UpdateEmbeddings(input_data, p_out_temp_.data());
    // A(FW)
    AggregateAll(layer_dimensions_.output_columns, p_out_temp_.data(),
                 p_forward_output_matrix_.data(),
                 &output_column_intermediates_);
  }

  // TODO synchronization of aggregation functions

  if (config_.do_activation) {
    GALOIS_LOG_VERBOSE("Doing activation");
    Activation();
  }

  assert(p_forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));
  return p_forward_output_matrix_;
}

galois::PointerWithSize<galois::GNNFloat>
galois::GraphConvolutionalLayer::BackwardPhase(
    galois::PointerWithSize<galois::GNNFloat> prev_layer_input,
    galois::PointerWithSize<galois::GNNFloat>* input_gradient) {
  assert(layer_phase_ == GNNPhase::kTrain);

  // derivative of activation
  if (config_.do_activation) {
    ActivationDerivative(input_gradient);
  }

  // AFW = O

  // derivative of aggregation/update
  // TODO clean up logic here to reduce nesting
  if (config_.disable_aggregate_after_update ||
      layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    if (layer_number_ != 0) {
      // transposed sgemm for derivative; in_temp is output
      assert(input_gradient->size() ==
             layer_dimensions_.input_rows * layer_dimensions_.output_columns);
      assert(p_in_temp_1_.size() ==
             layer_dimensions_.input_columns * layer_dimensions_.input_rows);
      // pintemp1 contains (AF)'
      UpdateEmbeddingsDerivative(input_gradient->data(), p_in_temp_1_.data());
      // pback contains F'
      // derivative of aggregate is the same due to symmetric graph
      AggregateAll(layer_dimensions_.input_columns, p_in_temp_1_.data(),
                   p_backward_output_matrix_.data(),
                   &input_column_intermediates_);
      // TODO if training A, then A' compute here if layer # is 0
      // dot product of edges that exist in A
    }
    // weight gradient calculation
    // TODO(loc) put this in a function to put the ifdef in there
#ifndef GALOIS_ENABLE_GPU
    // temp 2 holds aggregated feature vectors from forward phase
    galois::CBlasSGEMM(
        CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
        layer_dimensions_.input_rows, layer_dimensions_.output_columns,
        p_in_temp_2_.data(), input_gradient->data(),
        p_layer_weight_gradients_.data());
#else
    gpu_object_.GetWeightGradientsGPU(
        layer_dimensions_.input_rows, layer_dimensions_.input_columns,
        layer_dimensions_.output_columns, p_in_temp_2_.data(),
        input_gradient->data(), p_layer_weight_gradients_.data());
#endif
  } else {
    // TODO at this point, out_temp contains memoized FW
    // can use it to get A' = O' (FW)^T
    // aggregate occurs regardless of layer being equal to 0 because it is
    // required in this case for the weight gradient calculation
    // this is (FW)'
    AggregateAll(layer_dimensions_.output_columns, input_gradient->data(),
                 p_out_temp_.data(), &output_column_intermediates_);
    if (layer_number_ != 0) {
      // derivative for update
      // backout = F'
      UpdateEmbeddingsDerivative(p_out_temp_.data(),
                                 p_backward_output_matrix_.data());
    }
    // TODO put this in a function
    // W' = F^T (FW)'
#ifndef GALOIS_ENABLE_GPU
    // weight gradient; note the use of the aggregated gradient in out_temp
    galois::CBlasSGEMM(
        CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
        layer_dimensions_.input_rows, layer_dimensions_.output_columns,
        prev_layer_input.data(), p_out_temp_.data(),
        p_layer_weight_gradients_.data());
#else
    gpu_object_.GetWeightGradientsGPU(
        layer_dimensions_.input_rows, layer_dimensions_.input_columns,
        layer_dimensions_.output_columns, prev_layer_input.data(),
        p_out_temp_.data(), p_layer_weight_gradients_.data());
#endif
  }

  // sync weight gradients; note aggregation sync occurs in the function call
  // already
  // TODO figure out how to do this with GPUs
  // WeightGradientSyncAverage();
  WeightGradientSyncSum();

  if (config_.do_dropout && layer_number_ != 0) {
    DoDropoutDerivative();
  }

  return p_backward_output_matrix_;
}

void galois::GraphConvolutionalLayer::AggregateAll(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    [[maybe_unused]] galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*
        pts) {
#ifndef GALOIS_ENABLE_GPU
  AggregateAllCPU(column_length, node_embeddings, aggregate_output, pts);
#else
  gpu_object_.AggregateAllGPU(graph_.GetGPUGraph(), graph_.size(),
                              column_length, node_embeddings, aggregate_output,
                              config_.do_normalization);
#endif
}

void galois::GraphConvolutionalLayer::AggregateAllCPU(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts) {
  size_t num_nodes = graph_.size();

  galois::do_all(
      galois::iterate(static_cast<size_t>(0), num_nodes),
      [&](size_t src) {
        size_t index_to_src_feature = src * column_length;
        // zero out src feature first
        // TODO(loc) can init to self as well to add to self
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

        // GNNFloat* intermediate = pts->getLocal()->data();
        // GNNFloat norm_scale = source_norm * source_norm;
        // for (size_t i = 0; i < column_length; i++) {
        //  intermediate[i] =
        //      norm_scale * node_embeddings[index_to_src_feature + i];
        //}
        //// add self
        // galois::VectorAdd(column_length,
        //                  &aggregate_output[index_to_src_feature],
        //                  intermediate,
        //                  &aggregate_output[index_to_src_feature]);
      },
      galois::steal(), galois::loopname("ConvolutionalAggregateAll"));

  // aggregate sync
  graph_.AggregateSync(aggregate_output, column_length);
}

void galois::GraphConvolutionalLayer::UpdateEmbeddings(
    const GNNFloat* node_embeddings, GNNFloat* output) {
#ifndef GALOIS_ENABLE_GPU
  // CPU version is just a call into CBlas
  galois::CBlasSGEMM(CblasNoTrans, CblasNoTrans, layer_dimensions_.input_rows,
                     layer_dimensions_.input_columns,
                     layer_dimensions_.output_columns, node_embeddings,
                     layer_weights_.data(), output);
#else
  gpu_object_.UpdateEmbeddingsGPU(
      layer_dimensions_.input_rows, layer_dimensions_.input_columns,
      layer_dimensions_.output_columns, node_embeddings,
      base_gpu_object_.layer_weights(), output);
#endif
}

void galois::GraphConvolutionalLayer::UpdateEmbeddingsDerivative(
    const GNNFloat* gradients, GNNFloat* output) {
  assert(p_layer_weights_.size() ==
         layer_dimensions_.input_columns * layer_dimensions_.output_columns);
#ifndef GALOIS_ENABLE_GPU
  // difference is Trans for B matrix (data) to get z by y (weights is y by z
  // normally); result is x by y
  galois::CBlasSGEMM(CblasNoTrans, CblasTrans, layer_dimensions_.input_rows,
                     layer_dimensions_.output_columns,
                     layer_dimensions_.input_columns, gradients,
                     layer_weights_.data(), output);
#else
  gpu_object_.UpdateEmbeddingsDerivativeGPU(
      layer_dimensions_.input_rows, layer_dimensions_.input_columns,
      layer_dimensions_.output_columns, gradients,
      base_gpu_object_.layer_weights(), output);

#endif
}

#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/SAGELayer.h"

galois::SAGELayer::SAGELayer(size_t layer_num,
                             const galois::graphs::GNNGraph& graph,
                             const GNNLayerDimensions& dimensions,
                             const GNNLayerConfig& config,
                             const SAGELayerConfig& sage_config)
    : GNNLayer(layer_num, graph, dimensions, config), sage_config_(sage_config),
      input_column_intermediates_(dimensions.input_columns),
      output_column_intermediates_(dimensions.output_columns) {
  if (!sage_config_.disable_concat) {
    // there are now 2 weight matrices used: one for self, one for aggregation
    // abstractly it's one matrix: W = W1 | W2
    size_t num_weight_elements =
        layer_dimensions_.input_columns * layer_dimensions_.output_columns;
    layer_weights_2_.resize(num_weight_elements);
    layer_weight_gradients_2_.resize(num_weight_elements, 0);
    GlorotBengioInit(&layer_weights_2_);
    // update the pointers to them as well as realloc will require it
    p_layer_weights_2_ = PointerWithSize<GNNFloat>(layer_weights_2_);
    p_layer_weight_gradients_2_ =
        PointerWithSize<GNNFloat>(layer_weight_gradients_2_);
    // initialize the optimizer
    std::vector<size_t> weight_size = {num_weight_elements};
    second_weight_optimizer_ = std::make_unique<AdamOptimizer>(weight_size, 1);
  }

  size_t num_input_elements =
      layer_dimensions_.input_rows * layer_dimensions_.input_columns;
  in_temp_1_.resize(num_input_elements, 0);
  // TODO temp2 does not need to be initialized in all circumstances
  in_temp_2_.resize(num_input_elements, 0);

  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;
  GALOIS_LOG_VERBOSE("Output elements {}", num_output_elements);
  out_temp_.resize(num_output_elements, 0);
  layer_type_ = galois::GNNLayerType::kSAGE;
#ifdef GALOIS_ENABLE_GPU
  // TODO(loc/hochan) GPU SAGE
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.Allocate(num_input_elements, num_output_elements);
    // init pointers with size
    p_in_temp_1_ =
        PointerWithSize<GNNFloat>(gpu_object_.in_temp_1(), in_temp_1_.size());
    p_in_temp_2_ =
        PointerWithSize<GNNFloat>(gpu_object_.in_temp_2(), in_temp_2_.size());
    p_out_temp_ =
        PointerWithSize<GNNFloat>(gpu_object_.out_temp(), out_temp_.size());
  } else {
#endif
    p_in_temp_1_ = PointerWithSize<GNNFloat>(in_temp_1_);
    p_in_temp_2_ = PointerWithSize<GNNFloat>(in_temp_2_);
    p_out_temp_  = PointerWithSize<GNNFloat>(out_temp_);
#ifdef GALOIS_ENABLE_GPU
    // TODO concat parameters
  }
#endif

  GALOIS_LOG_VERBOSE("SAGE layer initialized");
}

void MatrixAdd(size_t num_nodes, galois::PointerWithSize<galois::GNNFloat> in,
               galois::PointerWithSize<galois::GNNFloat>* out) {
  assert(in.size() == out->size());
  assert((in.size() % num_nodes) == 0);
  size_t column_size = in.size() / num_nodes;
  // split matrix to threads
  galois::do_all(galois::iterate(size_t{0}, num_nodes), [&](size_t node) {
    size_t my_offset = node * column_size;
    galois::VectorAdd(column_size, &(in[my_offset]),
                      &((out->data())[my_offset]), &(out->data()[my_offset]));
  });
}

const galois::PointerWithSize<galois::GNNFloat> galois::SAGELayer::ForwardPhase(
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
  if (!config_.disable_dropout && (layer_phase_ == GNNPhase::kTrain)) {
    DoDropout(input_embeddings, &p_in_temp_1_);
    input_data = p_in_temp_1_.data();
  }

  // O = FW1 + AFW2 is what is done if concat is on: below is the AFW2 part
  // which is done regardless

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

  if (!sage_config_.disable_concat) {
    // FW1 is unaffected by the agg/update flip, so can to it
    // separately
    SelfFeatureUpdateEmbeddings(input_data, p_forward_output_matrix_.data());
  }

  if (!config_.disable_activation) {
    GALOIS_LOG_VERBOSE("Doing activation");
    Activation();
  }

  assert(p_forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));

  return p_forward_output_matrix_;
}

galois::PointerWithSize<galois::GNNFloat> galois::SAGELayer::BackwardPhase(
    galois::PointerWithSize<galois::GNNFloat> prev_layer_input,
    galois::PointerWithSize<galois::GNNFloat>* input_gradient) {
  assert(layer_phase_ == GNNPhase::kTrain);

  // derivative of activation
  if (!config_.disable_activation) {
    ActivationDerivative(input_gradient);
  }

  // if dropout was used, use the dropout matrix for the input
  galois::PointerWithSize<galois::GNNFloat> input_to_use;
  if (!config_.disable_dropout) {
    // dropout result is currently stored in temp 1
    // needs to be used before it gets overwritten
    input_to_use = p_in_temp_1_;
  } else {
    // no dropout = use vanilla input
    input_to_use = prev_layer_input;
  }

  // AFW = O
  if (!sage_config_.disable_concat) {
    // Fw1 + AFW2 = O; self feature has own weight matrix and makes own
    // contribution to gradients which is handled in this block
    // !!!! do this early because p_in_temp may get overwritten later
    // if update occurs before aggregate !!!
    galois::CBlasSGEMM(
        CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
        layer_dimensions_.input_rows, layer_dimensions_.output_columns,
        input_to_use.data(), input_gradient->data(),
        p_layer_weight_gradients_2_.data());
  }

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
                   &input_column_intermediates_, true);
    }
    // weight gradient calculation
    // TODO(loc) put this in a function to put the ifdef in there
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.GetWeightGradientsGPU(
          layer_dimensions_.input_rows, layer_dimensions_.input_columns,
          layer_dimensions_.output_columns, p_in_temp_2_.data(),
          input_gradient->data(), p_layer_weight_gradients_.data());
    } else {
#endif
      // temp 2 holds aggregated feature vectors from forward phase
      galois::CBlasSGEMM(
          CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
          layer_dimensions_.input_rows, layer_dimensions_.output_columns,
          p_in_temp_2_.data(), input_gradient->data(),
          p_layer_weight_gradients_.data());
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  } else {
    // aggregate occurs regardless of layer being equal to 0 because it is
    // required in this case for the weight gradient calculation
    // this is (FW)'
    AggregateAll(layer_dimensions_.output_columns, input_gradient->data(),
                 p_out_temp_.data(), &output_column_intermediates_, true);
    if (layer_number_ != 0) {
      // derivative for update
      // backout = F'
      UpdateEmbeddingsDerivative(p_out_temp_.data(),
                                 p_backward_output_matrix_.data());
    }
    // TODO put this in a function
    // W' = F^T (FW)'
    // input to use is not overwritten in this branch so it's safe to use
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.GetWeightGradientsGPU(
          layer_dimensions_.input_rows, layer_dimensions_.input_columns,
          layer_dimensions_.output_columns, input_to_use.data(),
          p_out_temp_.data(), p_layer_weight_gradients_.data());
    } else {
#endif
      galois::CBlasSGEMM(CblasTrans, CblasNoTrans,
                         layer_dimensions_.input_columns,
                         layer_dimensions_.input_rows,
                         layer_dimensions_.output_columns, input_to_use.data(),
                         p_out_temp_.data(), p_layer_weight_gradients_.data());
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  if (!sage_config_.disable_concat) {
    if (layer_number_ != 0) {
      // deal with feature gradients for the self feature here
      // this function will sum directly into the backward matrix
      SelfFeatureUpdateEmbeddingsDerivative(input_gradient->data(),
                                            p_backward_output_matrix_.data());
    }
  }

  // TODO(loc) sync both weight matrices
  WeightGradientSyncSum();

  if (!config_.disable_dropout && layer_number_ != 0) {
    DoDropoutDerivative();
  }

  return p_backward_output_matrix_;
}

void galois::SAGELayer::AggregateAll(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    [[maybe_unused]] galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*
        pts) {
  AggregateAll(column_length, node_embeddings, aggregate_output, pts, false);
}

void galois::SAGELayer::AggregateAll(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    [[maybe_unused]] galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*
        pts,
    bool is_backward) {
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.AggregateAllGPU(
        graph_.GetGPUGraph(), graph_.size(), column_length, node_embeddings,
        aggregate_output, !config_.disable_normalization);
    graph_.AggregateSync(aggregate_output, column_length, layer_number_);
  } else {
#endif
    AggregateAllCPU(column_length, node_embeddings, aggregate_output, pts,
                    is_backward);
#ifdef GALOIS_ENABLE_GPU
  }
#endif
}

void galois::SAGELayer::AggregateAllCPU(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*,
    bool is_backward) {
  size_t num_nodes = graph_.size();

  galois::do_all(
      galois::iterate(static_cast<size_t>(0), num_nodes),
      [&](size_t src) {
        size_t index_to_src_feature = src * column_length;
        // zero out src feature first
        for (size_t i = 0; i < column_length; i++) {
          aggregate_output[index_to_src_feature + i] = 0;
        }

        if (layer_phase_ == GNNPhase::kTrain) {
          if (IsInductiveLayer()) {
            // if inductive, all non-training nodes do not exist
            if (!graph_.IsValidForPhase(src, GNNPhase::kTrain))
              return;
          }

          if (IsSampledLayer()) {
            // check if node is part of sampled graph; ignore after 0'ing if not
            // sampled
            if (!graph_.IsInSampledGraph(src))
              return;
          }
        }

        GNNFloat source_norm = 0.0;
        if (!config_.disable_normalization) {
          source_norm = graph_.DegreeNorm(src);
        }

        // loop through all destinations to grab the feature to aggregate
        for (auto e = graph_.EdgeBegin(src); e != graph_.EdgeEnd(src); e++) {
          size_t dst = graph_.EdgeDestination(e);

          if (layer_phase_ == GNNPhase::kTrain) {
            if (IsInductiveLayer()) {
              // if inductive, all non-training nodes do not exist
              if (!graph_.IsValidForPhase(dst, GNNPhase::kTrain))
                return;
            }

            if (IsSampledLayer()) {
              // ignore non-sampled nodes
              if (layer_phase_ == GNNPhase::kTrain &&
                  !graph_.IsInSampledGraph(dst))
                continue;
            }
          }

          size_t index_to_dst_feature = dst * column_length;

          if (!config_.disable_normalization) {
            GNNFloat norm_scale;
            if (!is_backward) {
              norm_scale = source_norm;
            } else {
              norm_scale = graph_.DegreeNorm(dst);
            }

            galois::VectorMulAdd(
                column_length, &aggregate_output[index_to_src_feature],
                &node_embeddings[index_to_dst_feature], norm_scale,
                &aggregate_output[index_to_src_feature]);
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
  // aggregate sync
  graph_.AggregateSync(aggregate_output, column_length);
}

void galois::SAGELayer::UpdateEmbeddings(const GNNFloat* node_embeddings,
                                         GNNFloat* output) {
#ifdef GALOIS_ENABLE_GPU
  // TODO self change
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.UpdateEmbeddingsGPU(
        layer_dimensions_.input_rows, layer_dimensions_.input_columns,
        layer_dimensions_.output_columns, node_embeddings,
        base_gpu_object_.layer_weights(), output);
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

void galois::SAGELayer::SelfFeatureUpdateEmbeddings(
    const GNNFloat* node_embeddings, GNNFloat* output) {
#ifdef GALOIS_ENABLE_GPU
  // TODO self change
#endif
  // note use of layer weights 2 differentiates this from above
  galois::CBlasSGEMM(CblasNoTrans, CblasNoTrans, layer_dimensions_.input_rows,
                     layer_dimensions_.input_columns,
                     layer_dimensions_.output_columns, node_embeddings,
                     layer_weights_2_.data(), output, true);
#ifdef GALOIS_ENABLE_GPU
}
#endif
}

void galois::SAGELayer::UpdateEmbeddingsDerivative(const GNNFloat* gradients,
                                                   GNNFloat* output) {
  assert(p_layer_weights_.size() ==
         layer_dimensions_.input_columns * layer_dimensions_.output_columns);
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.UpdateEmbeddingsDerivativeGPU(
        layer_dimensions_.input_rows, layer_dimensions_.input_columns,
        layer_dimensions_.output_columns, gradients,
        base_gpu_object_.layer_weights(), output);
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

void galois::SAGELayer::SelfFeatureUpdateEmbeddingsDerivative(
    const GNNFloat* gradients, GNNFloat* output) {
  assert(p_layer_weights_.size() ==
         layer_dimensions_.input_columns * layer_dimensions_.output_columns);
#ifdef GALOIS_ENABLE_GPU
  // TODO gpu self
#endif
  // difference is Trans for B matrix (data) to get z by y (weights is y by z
  // normally); result is x by y
  // true at end -> accumulate
  galois::CBlasSGEMM(CblasNoTrans, CblasTrans, layer_dimensions_.input_rows,
                     layer_dimensions_.output_columns,
                     layer_dimensions_.input_columns, gradients,
                     layer_weights_2_.data(), output, true);
#ifdef GALOIS_ENABLE_GPU
#endif
}

void galois::SAGELayer::OptimizeLayer(BaseOptimizer* optimizer,
                                      size_t trainable_layer_number) {
  optimizer->GradientDescent(p_layer_weight_gradients_, p_layer_weights_,
                             trainable_layer_number);
  if (!sage_config_.disable_concat) {
    second_weight_optimizer_->GradientDescent(p_layer_weight_gradients_2_,
                                              p_layer_weights_2_, 0);
  }
}

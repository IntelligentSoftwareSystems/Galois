#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/SAGELayer.h"

galois::SAGELayer::SAGELayer(size_t layer_num,
                             const galois::graphs::GNNGraph& graph,
                             PointerWithSize<GNNFloat>* backward_output_matrix,
                             const GNNLayerDimensions& dimensions,
                             const GNNLayerConfig& config,
                             const SAGELayerConfig& sage_config)
    : GNNLayer(layer_num, graph, backward_output_matrix, dimensions, config),
      sage_config_(sage_config),
      input_column_intermediates_(dimensions.input_columns),
      output_column_intermediates_(dimensions.output_columns) {
  if (!sage_config_.disable_concat) {
    // there are now 2 weight matrices used: one for self, one for aggregation
    // abstractly it's one matrix: W = W1 | W2
    size_t num_weight_elements =
        layer_dimensions_.input_columns * layer_dimensions_.output_columns;
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", SAGE second layer weights ", num_weight_elements, " (",
                  FloatElementsToGB(num_weight_elements), " GB)");
    layer_weights_2_.resize(num_weight_elements);
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", SAGE second layer gradients ", num_weight_elements, " (",
                  FloatElementsToGB(num_weight_elements), " GB)");
    layer_weight_gradients_2_.resize(num_weight_elements, 0);

    // reinit both weight matrices as one unit
    PairGlorotBengioInit(&layer_weights_, &layer_weights_2_);

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

  // if in temp is smaller than out temp, or if dropout exists
  if (!config_.disable_dropout || config_.disable_aggregate_after_update ||
      layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", SAGE input temp var 1 ", num_input_elements, " (",
                  FloatElementsToGB(num_input_elements), " GB)");
    in_temp_1_.resize(num_input_elements, 0);
  }

  // only on in dropout case + if in temp is smaller than out temp
  if (!config_.disable_dropout &&
      (config_.disable_aggregate_after_update ||
       layer_dimensions_.input_columns <= layer_dimensions_.output_columns)) {
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", SAGE input temp var 2 ", num_input_elements, " (",
                  FloatElementsToGB(num_input_elements), " GB)");
    in_temp_2_.resize(num_input_elements, 0);
  }

  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;

  // only needed if out temp would be smaller than intemp
  if (!config_.disable_aggregate_after_update &&
      layer_dimensions_.input_columns > layer_dimensions_.output_columns) {
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", SAGE output temp var ", num_output_elements, " (",
                  FloatElementsToGB(num_output_elements), " GB)");
    out_temp_.resize(num_output_elements, 0);
  }

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

void galois::SAGELayer::WeightGradientSyncSum2() {
  galois::StatTimer t("Sync_WeightGradientsSum2", kRegionName);
  t.start();
#ifdef GALOIS_ENABLE_GPU
  // TODO(hochan) collectives here rather than gluon sync if possible like the
  // CPU code
  GALOIS_LOG_FATAL("implement me");
#else
  // TODO(loc) remove this limitation later; can just do a loop over the weight
  // matrix
  if (p_layer_weight_gradients_2_.size() >
      size_t{std::numeric_limits<int>::max()}) {
    GALOIS_LOG_FATAL("Weight sync code does not handle size larger than max "
                     "int at the moment");
  }
  MPI_Allreduce(MPI_IN_PLACE,
                static_cast<void*>(p_layer_weight_gradients_2_.data()),
                static_cast<int>(p_layer_weight_gradients_2_.size()), MPI_FLOAT,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  t.stop();
}

const galois::PointerWithSize<galois::GNNFloat> galois::SAGELayer::ForwardPhase(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
  galois::StatTimer timer("ForwardPhase", kRegionName);
  timer.start();

  assert(input_embeddings.size() >=
         (layer_dimensions_.input_rows * layer_dimensions_.input_columns));
  assert(p_forward_output_matrix_.size() >=
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));

  // pointer to input to operate on
  const GNNFloat* input_data = input_embeddings.data();
  GNNFloat* agg_data;
  // first, dropout
  if (!config_.disable_dropout && (layer_phase_ == GNNPhase::kTrain)) {
    DoDropout(input_embeddings, &p_in_temp_1_);
    input_data = p_in_temp_1_.data();
    agg_data   = p_in_temp_2_.data();
  } else {
    agg_data = p_in_temp_1_.data();
  }

  // O = FW1 + AFW2 is what is done if concat is on: below is the AFW2 part
  // which is done regardless

  // flip aggregate/update if dimensions favor it (do less work)
  if (config_.disable_aggregate_after_update ||
      layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    // aggregation and update
    AggregateAll(layer_dimensions_.input_columns, input_data, agg_data,
                 &input_column_intermediates_);
    UpdateEmbeddings(agg_data, p_forward_output_matrix_.data());
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

  assert(p_forward_output_matrix_.size() >=
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));

  timer.stop();

  return p_forward_output_matrix_;
}

galois::PointerWithSize<galois::GNNFloat> galois::SAGELayer::BackwardPhase(
    galois::PointerWithSize<galois::GNNFloat> prev_layer_input,
    galois::PointerWithSize<galois::GNNFloat>* input_gradient) {
  galois::StatTimer timer("BackwardPhase", kRegionName);
  timer.start();

  assert(layer_phase_ == GNNPhase::kTrain || layer_phase_ == GNNPhase::kBatch);

  // derivative of activation
  if (!config_.disable_activation) {
    ActivationDerivative(input_gradient);
  }

  // if dropout was used, use the dropout matrix for the input
  galois::PointerWithSize<galois::GNNFloat> input_data;
  galois::PointerWithSize<galois::GNNFloat> agg_data;
  if (!config_.disable_dropout) {
    // dropout result is currently stored in temp 1
    // needs to be used before it gets overwritten
    input_data = p_in_temp_1_;
    agg_data   = p_in_temp_2_;
  } else {
    // no dropout = use vanilla input
    input_data = prev_layer_input;
    agg_data   = p_in_temp_1_;
  }

  // aggregate this here before gradient starts to get overwritten
  if (!config_.disable_aggregate_after_update &&
      layer_dimensions_.input_columns > layer_dimensions_.output_columns) {
    // aggregate occurs regardless of layer being equal to 0 because it is
    // required in this case for the weight gradient calculation
    // this is (FW)'
    AggregateAll(layer_dimensions_.output_columns, input_gradient->data(),
                 p_out_temp_.data(), &output_column_intermediates_, true);
  }

  if (!sage_config_.disable_concat) {
    if (layer_number_ != 0) {
      MaskInputNonMasters(&input_data);
    } else {
      // if 0 then no input to mask: mask the gradient
      // this is fine because gradient won't be used to get feature gradients
      MaskGradientNonMasters(input_gradient);
    }
    // input data (prev layer input or temp1) or gradient need mask
    // can mask gradient if layer == 0
    // otherwise must mask other
    galois::CBlasSGEMM(
        CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
        layer_dimensions_.input_rows, layer_dimensions_.output_columns,
        input_data.data(), input_gradient->data(),
        p_layer_weight_gradients_2_.data());
  }
  WeightGradientSyncSum2();

  // AFW = O

  // derivative of aggregation/update
  // TODO clean up logic here to reduce nesting
  if (config_.disable_aggregate_after_update ||
      layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    // aggdata can == p_intemp1; in other words, need to use before overwrite
    // mask it, then use it
    if (layer_number_ != 0 || sage_config_.disable_concat) {
      MaskInputNonMasters(&agg_data);
    }
    // if concat is disabled, then input grad isn't masked; therefore, mask
    // this to get the same effect

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.GetWeightGradientsGPU(
          layer_dimensions_.input_rows, layer_dimensions_.input_columns,
          layer_dimensions_.output_columns, agg_data.data(),
          input_gradient->data(), p_layer_weight_gradients_.data());
    } else {
#endif
      // temp 2 holds aggregated feature vectors from forward phase
      galois::CBlasSGEMM(
          CblasTrans, CblasNoTrans, layer_dimensions_.input_columns,
          layer_dimensions_.input_rows, layer_dimensions_.output_columns,
          agg_data.data(), input_gradient->data(),
          p_layer_weight_gradients_.data());
#ifdef GALOIS_ENABLE_GPU
    }
#endif

    // 0 means input gradient shouldn't get masked
    if (layer_number_ != 0) {
      // ---unmasked---
      // transposed sgemm for derivative; in_temp is output
      assert(input_gradient->size() >=
             layer_dimensions_.input_rows * layer_dimensions_.output_columns);
      // pintemp1 contains (AF)'
      // overwrites the dropout matrix that was in ptemp1 (needed for second
      // weight matrix)
      UpdateEmbeddingsDerivative(input_gradient->data(), p_in_temp_1_.data());
      // pback contains F'
      // derivative of aggregate is the same due to symmetric graph
      AggregateAll(layer_dimensions_.input_columns, p_in_temp_1_.data(),
                   p_backward_output_matrix_.data(),
                   &input_column_intermediates_, true);
    }
  } else {
    // --unmasked--
    // disable concat part is here because otherwise it would get done elsewhere
    if (layer_number_ != 0 && sage_config_.disable_concat) {
      MaskInputNonMasters(&input_data);
    } else {
      // if 0 then no input to mask: mask the gradient
      // this is fine because gradient won't be used to get feature gradients
      MaskGradientNonMasters(&p_out_temp_);
    }

    // W' = F^T (FW)'
    // TODO put this in a function
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.GetWeightGradientsGPU(
          layer_dimensions_.input_rows, layer_dimensions_.input_columns,
          layer_dimensions_.output_columns, input_data.data(),
          p_out_temp_.data(), p_layer_weight_gradients_.data());
    } else {
#endif
      galois::CBlasSGEMM(CblasTrans, CblasNoTrans,
                         layer_dimensions_.input_columns,
                         layer_dimensions_.input_rows,
                         layer_dimensions_.output_columns, input_data.data(),
                         p_out_temp_.data(), p_layer_weight_gradients_.data());
#ifdef GALOIS_ENABLE_GPU
    }
#endif

    if (layer_number_ != 0) {
      // derivative for update
      // backout = F'
      UpdateEmbeddingsDerivative(p_out_temp_.data(),
                                 p_backward_output_matrix_.data());
    }
  }
  WeightGradientSyncSum();

  // full gradient needed here; should occur after all updates
  if (layer_number_ != 0) {
    // deal with feature gradients for the self feature here
    // this function will sum directly into the backward matrix
    // input gradient never gets masked if layer number != 0
    SelfFeatureUpdateEmbeddingsDerivative(input_gradient->data(),
                                          p_backward_output_matrix_.data());
  }

  if (!config_.disable_dropout && layer_number_ != 0) {
    DoDropoutDerivative();
  }

  timer.stop();
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
  std::string agg_timer_name = "Aggregate";
  if (!is_backward) {
    agg_timer_name += "Forward";
  } else {
    agg_timer_name += "Backward";
  }
  galois::StatTimer timer(agg_timer_name.c_str(), kRegionName);
  timer.start();

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    if (!IsSampledLayer()) {
      gpu_object_.AggregateAllGPU(
          graph_.GetGPUGraph(), graph_.size(), column_length, node_embeddings,
          aggregate_output, !config_.disable_normalization);
    } else {
      // TODO(hochan)
      GALOIS_LOG_FATAL("SAMPLING IMPLEMENTATION");
    }
    graph_.AggregateSync(aggregate_output, column_length, layer_number_);
  } else {
#endif
    AggregateAllCPU(column_length, node_embeddings, aggregate_output, pts,
                    is_backward);
#ifdef GALOIS_ENABLE_GPU
  }
#endif
  timer.stop();
}

void galois::SAGELayer::AggregateAllCPU(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*,
    bool is_backward) {
  galois::do_all(
      galois::iterate(graph_.begin(), graph_.end()),
      [&](size_t src) {
        size_t index_to_src_feature = src * column_length;
        // zero out src feature first
        for (size_t i = 0; i < column_length; i++) {
          aggregate_output[index_to_src_feature + i] = 0;
        }

        GNNFloat source_norm = 0.0;
        if (!config_.disable_normalization) {
          source_norm = graph_.GetDegreeNorm(src, graph_user_layer_number_);
        }

        if (!is_backward) {
          // loop through all destinations to grab the feature to aggregate
          for (auto e = graph_.edge_begin(src); e != graph_.edge_end(src);
               e++) {
            graphs::bitset_graph_aggregate.set(graph_.ConvertToLID(src));
            size_t dst = graph_.GetEdgeDest(e);

            if (layer_phase_ == GNNPhase::kTrain) {
              // XXX
              if (IsSampledLayer()) {
                if (!graph_.IsEdgeSampled(e, layer_number_)) {
                  continue;
                }
              }
            }

            size_t index_to_dst_feature = dst * column_length;

            if (!config_.disable_normalization) {
              GNNFloat norm_scale;
              if (!is_backward) {
                norm_scale = source_norm;
              } else {
                norm_scale =
                    graph_.GetDegreeNorm(dst, graph_user_layer_number_);
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
        } else {
          // loop through all destinations to grab the feature to aggregate
          for (auto e = graph_.in_edge_begin(src); e != graph_.in_edge_end(src);
               e++) {
            graphs::bitset_graph_aggregate.set(graph_.ConvertToLID(src));
            size_t dst = graph_.GetInEdgeDest(e);

            if (layer_phase_ == GNNPhase::kTrain) {
              // XXX
              if (IsSampledLayer()) {
                if (!graph_.IsInEdgeSampled(e, layer_number_)) {
                  continue;
                }
              }
            }

            size_t index_to_dst_feature = dst * column_length;

            if (!config_.disable_normalization) {
              GNNFloat norm_scale =
                  graph_.GetDegreeNorm(dst, graph_user_layer_number_);

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
        }
      },
      galois::chunk_size<1>(), galois::steal(),
      galois::loopname("ConvolutionalAggregateAll"));

  // aggregate sync
  graph_.AggregateSync(aggregate_output, column_length, is_backward);
}

void galois::SAGELayer::UpdateEmbeddings(const GNNFloat* node_embeddings,
                                         GNNFloat* output) {
  galois::StatTimer timer("ForwardXForm", kRegionName);
  timer.start();
#ifdef GALOIS_ENABLE_GPU
  // TODO self change
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.UpdateEmbeddingsGPU(
        layer_dimensions_.input_rows, layer_dimensions_.input_columns,
        layer_dimensions_.output_columns, node_embeddings,
        base_gpu_object_.layer_weights(), output);
  } else {
#endif
    galois::gDebug("Layer ", graph_user_layer_number_, " ",
                   layer_dimensions_.input_rows, " ",
                   layer_dimensions_.input_columns, " ",
                   layer_dimensions_.output_columns);
    // CPU version is just a call into CBlas
    galois::CBlasSGEMM(CblasNoTrans, CblasNoTrans, layer_dimensions_.input_rows,
                       layer_dimensions_.input_columns,
                       layer_dimensions_.output_columns, node_embeddings,
                       layer_weights_.data(), output);
#ifdef GALOIS_ENABLE_GPU
  }
#endif
  timer.stop();
}

void galois::SAGELayer::SelfFeatureUpdateEmbeddings(
    const GNNFloat* node_embeddings, GNNFloat* output) {
  galois::StatTimer timer("SelfForwardXForm", kRegionName);
  timer.start();
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
timer.stop();
}

void galois::SAGELayer::UpdateEmbeddingsDerivative(const GNNFloat* gradients,
                                                   GNNFloat* output) {
  galois::StatTimer timer("BackwardXForm", kRegionName);
  timer.start();

  assert(p_layer_weights_.size() >=
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
  timer.stop();
}

void galois::SAGELayer::SelfFeatureUpdateEmbeddingsDerivative(
    const GNNFloat* gradients, GNNFloat* output) {
  galois::StatTimer timer("SelfBackwardXForm", kRegionName);
  timer.start();

  assert(p_layer_weights_.size() >=
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
  timer.stop();
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

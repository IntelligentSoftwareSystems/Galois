#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/GraphConvolutionalLayer.h"

galois::GraphConvolutionalLayer::GraphConvolutionalLayer(
    size_t layer_num, const galois::graphs::GNNGraph& graph,
    PointerWithSize<GNNFloat>* backward_output_matrix,
    const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
    : GNNLayer(layer_num, graph, backward_output_matrix, dimensions, config),
      input_column_intermediates_(dimensions.input_columns),
      output_column_intermediates_(dimensions.output_columns) {
  size_t num_input_elements =
      layer_dimensions_.input_rows * layer_dimensions_.input_columns;
  if (!config_.disable_dropout || config_.disable_aggregate_after_update ||
      layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", GCN input temp var 1 ", num_input_elements, " (",
                  FloatElementsToGB(num_input_elements), " GB)");
    in_temp_1_.resize(num_input_elements, 0);
  }

  // only on in dropout case + if in temp is smaller than out temp
  if (!config_.disable_dropout &&
      (config_.disable_aggregate_after_update ||
       layer_dimensions_.input_columns <= layer_dimensions_.output_columns)) {
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", GCN input temp var 2 ", num_input_elements, " (",
                  FloatElementsToGB(num_input_elements), " GB)");
    in_temp_2_.resize(num_input_elements, 0);
  }

  size_t num_output_elements =
      layer_dimensions_.input_rows * layer_dimensions_.output_columns;

  // only needed if out temp would be smaller than intemp
  if (!config_.disable_aggregate_after_update &&
      layer_dimensions_.input_columns > layer_dimensions_.output_columns) {
    // xform matrix first to work with a smaller output size
    galois::gInfo(graph_.host_prefix(), "Creating layer ", layer_number_,
                  ", GCN output temp var ", num_output_elements, " (",
                  FloatElementsToGB(num_output_elements), " GB)");
    out_temp_.resize(num_output_elements, 0);
  }

  layer_type_ = galois::GNNLayerType::kGraphConvolutional;
#ifdef GALOIS_ENABLE_GPU
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
  }
#endif

  GALOIS_LOG_VERBOSE("Conv layer initialized");
}

const galois::PointerWithSize<galois::GNNFloat>
galois::GraphConvolutionalLayer::ForwardPhase(
    const galois::PointerWithSize<galois::GNNFloat> input_embeddings) {
  galois::StatTimer timer("ForwardPhase", kRegionName);
  timer.start();
  GALOIS_LOG_VERBOSE("Calling forward phase");
  assert(input_embeddings.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.input_columns));
  assert(p_forward_output_matrix_.size() ==
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

  if (!config_.disable_activation) {
    GALOIS_LOG_VERBOSE("Doing activation");
    Activation();
  }

  assert(p_forward_output_matrix_.size() ==
         (layer_dimensions_.input_rows * layer_dimensions_.output_columns));
  timer.stop();

  return p_forward_output_matrix_;
}

galois::PointerWithSize<galois::GNNFloat>
galois::GraphConvolutionalLayer::BackwardPhase(
    galois::PointerWithSize<galois::GNNFloat> prev_layer_input,
    galois::PointerWithSize<galois::GNNFloat>* input_gradient) {
  galois::StatTimer timer("BackwardPhase", kRegionName);
  timer.start();

  assert(layer_phase_ == GNNPhase::kTrain);

  // derivative of activation
  if (!config_.disable_activation) {
    ActivationDerivative(input_gradient);
  }

  // AFW = O
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

  // NOTE: PREV LAYER INPUT AND BACKWARDOUTPUT ARE THE SAME MEMORY LOCATION;
  // BEWARE OF DEPENDENCIES

  // derivative of aggregation/update
  // TODO clean up logic here to reduce nesting
  if (config_.disable_aggregate_after_update ||
      layer_dimensions_.input_columns <= layer_dimensions_.output_columns) {
    // aggdata can == p_intemp1; in other words, need to use before overwrite
    // mask it, then use it
    MaskInputNonMasters(&agg_data);

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

    // gradient isn't masked here; only temp1, which has already been
    // overwritten = fine
    if (layer_number_ != 0) {
      // transposed sgemm for derivative; in_temp is output
      assert(input_gradient->size() ==
             layer_dimensions_.input_rows * layer_dimensions_.output_columns);
      // pintemp1 contains (AF)'
      UpdateEmbeddingsDerivative(input_gradient->data(), p_in_temp_1_.data());
      // pback contains F'
      // derivative of aggregate is the same due to symmetric graph
      AggregateAll(layer_dimensions_.input_columns, p_in_temp_1_.data(),
                   p_backward_output_matrix_.data(),
                   &input_column_intermediates_, true);
    }
  } else {
    // TODO at this point, out_temp contains memoized FW
    // can use it to get A' = O' (FW)^T
    // aggregate occurs regardless of layer being equal to 0 because it is
    // required in this case for the weight gradient calculation
    // this is (FW)'
    AggregateAll(layer_dimensions_.output_columns, input_gradient->data(),
                 p_out_temp_.data(), &output_column_intermediates_, true);

    // done after above because input_data = p_backward_output_matrix in some
    // cases; use first before overwriting here if layer # doesn't = 0, it means
    // I can mess with the input data itself instad of masking the gradients I
    // can mask the input
    if (layer_number_ != 0) {
      MaskInputNonMasters(&input_data);
    } else {
      // if 0 then no input to mask: mask the gradient
      // this is fine because gradient won't be used to get feature gradients
      MaskGradientNonMasters(&p_out_temp_);
    }

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
      // can now overwrite p_backward without issue; since input gradient
      // is untouched if layer number isn't 0 this will be correct
      UpdateEmbeddingsDerivative(p_out_temp_.data(),
                                 p_backward_output_matrix_.data());
    }
  }

  // sync weight gradients; note aggregation sync occurs in the function call
  // already
  WeightGradientSyncSum();

  if (!config_.disable_dropout && layer_number_ != 0) {
    DoDropoutDerivative();
  }

  timer.stop();
  return p_backward_output_matrix_;
}

void galois::GraphConvolutionalLayer::AggregateAll(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    [[maybe_unused]] galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*
        pts) {
  AggregateAll(column_length, node_embeddings, aggregate_output, pts, false);
}

void galois::GraphConvolutionalLayer::AggregateAll(
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
    gpu_object_.AggregateAllGPU(
        graph_.GetGPUGraph(), graph_.size(), column_length, node_embeddings,
        aggregate_output, !config_.disable_normalization);
    graph_.AggregateSync(aggregate_output, column_length, layer_number_);
  } else {
#endif
    AggregateAllCPU(column_length, node_embeddings, aggregate_output, pts);
#ifdef GALOIS_ENABLE_GPU
  }
#endif
  timer.stop();
}

void galois::GraphConvolutionalLayer::AggregateAllCPU(
    size_t column_length, const GNNFloat* node_embeddings,
    GNNFloat* aggregate_output,
    galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*) {
  size_t num_nodes   = graph_.size();
  size_t last_master = *(graph_.end_owned());
  assert(0 == *(graph_.begin_owned()));

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
            // XXX(loc)
            GALOIS_LOG_WARN(
                "Edge sampling not yet implemented for GCN; only SAGE");
            // check if node is part of sampled graph; ignore after 0'ing if not
            // sampled
            if (!graph_.IsInSampledGraph(src))
              return;
          }
        }

        GNNFloat source_norm = 0.0;
        if (!config_.disable_normalization) {
          source_norm = graph_.GetGCNNormFactor(src);
        }

        // init to self
        if (!config_.disable_self_aggregate) {
          graphs::bitset_graph_aggregate.set(src);
          // only aggregate self once on master
          if (src < last_master) {
            for (size_t i = 0; i < column_length; i++) {
              aggregate_output[index_to_src_feature + i] =
                  node_embeddings[index_to_src_feature + i] * source_norm *
                  source_norm;
            }
          }
        }

        // loop through all destinations to grab the feature to aggregate
        for (auto e = graph_.edge_begin(src); e != graph_.edge_end(src); e++) {
          size_t dst = graph_.GetEdgeDest(e);
          graphs::bitset_graph_aggregate.set(src);

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
            GNNFloat norm_scale = source_norm * graph_.GetGCNNormFactor(dst);
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
      galois::chunk_size<1>(), galois::steal(),
      galois::loopname("ConvolutionalAggregateAll"));
  // aggregate sync
  graph_.AggregateSync(aggregate_output, column_length);
}

void galois::GraphConvolutionalLayer::UpdateEmbeddings(
    const GNNFloat* node_embeddings, GNNFloat* output) {
  galois::StatTimer timer("ForwardXform", kRegionName);
  timer.start();

#ifdef GALOIS_ENABLE_GPU
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
  timer.stop();
}

void galois::GraphConvolutionalLayer::UpdateEmbeddingsDerivative(
    const GNNFloat* gradients, GNNFloat* output) {
  galois::StatTimer timer("BackwardXform", kRegionName);
  timer.start();

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
  timer.stop();
}

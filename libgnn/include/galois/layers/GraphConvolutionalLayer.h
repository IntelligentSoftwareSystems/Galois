#pragma once
#include "galois/layers/GNNLayer.h"
#include "galois/Logging.h"
#include "galois/GNNMath.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/layers/GraphConvolutionalLayer.cuh"
#endif

namespace galois {

extern galois::DynamicBitSet graphs::bitset_graph_aggregate;

template <typename VTy, typename ETy>
class GraphConvolutionalLayer : public GNNLayer<VTy, ETy> {
public:
  //! Initializes the variables of the base class and also allocates additional
  //! memory for temporary matrices. Also initializes sync substrate for the
  //! weight matrix
  GraphConvolutionalLayer(size_t layer_num,
                          const galois::graphs::GNNGraph<VTy, ETy>& graph,
                          PointerWithSize<GNNFloat>* backward_output_matrix,
                          const GNNLayerDimensions& dimensions,
                          const GNNLayerConfig& config)
      : GNNLayer<VTy, ETy>(layer_num, graph, backward_output_matrix, dimensions,
                           config),
        input_column_intermediates_(dimensions.input_columns),
        output_column_intermediates_(dimensions.output_columns) {
    galois::gWarn(
        "GCN layer not up to date with new subgraph/sampling changes; "
        "do not use until updated to reflect changes (see GraphSAGE layer)");

    size_t num_input_elements = this->layer_dimensions_.input_rows *
                                this->layer_dimensions_.input_columns;
    if (!this->config_.disable_dropout ||
        this->config_.disable_aggregate_after_update ||
        this->layer_dimensions_.input_columns <=
            this->layer_dimensions_.output_columns) {
      galois::gInfo(this->graph_.host_prefix(), "Creating layer ",
                    this->layer_number_, ", GCN input temp var 1 ",
                    num_input_elements, " (",
                    this->FloatElementsToGB(num_input_elements), " GB)");
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateInTemp1(num_input_elements);
      } else {
#endif
        in_temp_1_.resize(num_input_elements, 0);
#ifdef GALOIS_ENABLE_GPU
      }
#endif
    }

    // only on in dropout case + if in temp is smaller than out temp
    if (!this->config_.disable_dropout &&
        (this->config_.disable_aggregate_after_update ||
         this->layer_dimensions_.input_columns <=
             this->layer_dimensions_.output_columns)) {
      galois::gInfo(this->graph_.host_prefix(), "Creating layer ",
                    this->layer_number_, ", GCN input temp var 2 ",
                    num_input_elements, " (",
                    this->FloatElementsToGB(num_input_elements), " GB)");
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateInTemp2(num_input_elements);
      } else {
#endif
        in_temp_2_.resize(num_input_elements, 0);
#ifdef GALOIS_ENABLE_GPU
      }
#endif
    }

    size_t num_output_elements = this->layer_dimensions_.input_rows *
                                 this->layer_dimensions_.output_columns;

    // only needed if out temp would be smaller than intemp
    if (!this->config_.disable_aggregate_after_update &&
        this->layer_dimensions_.input_columns >
            this->layer_dimensions_.output_columns) {
      // xform matrix first to work with a smaller output size
      galois::gInfo(this->graph_.host_prefix(), "Creating layer ",
                    this->layer_number_, ", GCN output temp var ",
                    num_output_elements, " (",
                    this->FloatElementsToGB(num_output_elements), " GB)");
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateOutTemp(num_output_elements);
      } else {
#endif
        out_temp_.resize(num_output_elements, 0);
#ifdef GALOIS_ENABLE_GPU
      }
#endif
    }

    this->layer_type_ = galois::GNNLayerType::kGraphConvolutional;
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      // init pointers with size
      p_in_temp_1_ = PointerWithSize<GNNFloat>(gpu_object_.in_temp_1(),
                                               num_input_elements);
      p_in_temp_2_ = PointerWithSize<GNNFloat>(gpu_object_.in_temp_2(),
                                               num_input_elements);
      p_out_temp_  = PointerWithSize<GNNFloat>(gpu_object_.out_temp(),
                                              num_output_elements);
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

  GraphConvolutionalLayer(size_t layer_num,
                          const galois::graphs::GNNGraph<VTy, ETy>& graph,
                          PointerWithSize<GNNFloat>* backward_output_matrix,
                          const GNNLayerDimensions& dimensions)
      : GraphConvolutionalLayer(layer_num, graph, backward_output_matrix,
                                dimensions, GNNLayerConfig()) {}

  // Parent functions
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final {
    galois::StatTimer timer("ForwardPhase", kRegionName);
    timer.start();
    GALOIS_LOG_VERBOSE("Calling forward phase");
    assert(input_embeddings.size() == (this->layer_dimensions_.input_rows *
                                       this->layer_dimensions_.input_columns));
    assert(this->p_forward_output_matrix_.size() ==
           (this->layer_dimensions_.input_rows *
            this->layer_dimensions_.output_columns));
    // pointer to input to operate on
    const GNNFloat* input_data = input_embeddings.data();
    GNNFloat* agg_data;
    // first, dropout
    if (!this->config_.disable_dropout &&
        (this->layer_phase_ == GNNPhase::kTrain ||
         this->layer_phase_ == GNNPhase::kBatch)) {
      this->DoDropout(input_embeddings, &p_in_temp_1_);
      input_data = p_in_temp_1_.data();
      agg_data   = p_in_temp_2_.data();
    } else {
      agg_data = p_in_temp_1_.data();
    }

    // flip aggregate/update if dimensions favor it (do less work)
    if (this->config_.disable_aggregate_after_update ||
        this->layer_dimensions_.input_columns <=
            this->layer_dimensions_.output_columns) {
      // aggregation and update
      AggregateAll(this->layer_dimensions_.input_columns, input_data, agg_data,
                   &input_column_intermediates_);
      UpdateEmbeddings(agg_data, this->p_forward_output_matrix_.data());
    } else {
      // update to aggregate
      // FW
      UpdateEmbeddings(input_data, p_out_temp_.data());
      // A(FW)
      AggregateAll(this->layer_dimensions_.output_columns, p_out_temp_.data(),
                   this->p_forward_output_matrix_.data(),
                   &output_column_intermediates_);
    }

    if (!this->config_.disable_activation) {
      GALOIS_LOG_VERBOSE("Doing activation");
      this->Activation();
    }

    assert(this->p_forward_output_matrix_.size() ==
           (this->layer_dimensions_.input_rows *
            this->layer_dimensions_.output_columns));
    timer.stop();

    return this->p_forward_output_matrix_;
  }

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) final {
    galois::StatTimer timer("BackwardPhase", kRegionName);
    galois::StatTimer weight_gradient_timer("BackwardPhaseWeight", kRegionName);
    galois::StatTimer weight_gradient_sync_timer("BackwardPhaseWeightSync",
                                                 kRegionName);
    timer.start();

    assert(this->layer_phase_ == GNNPhase::kTrain ||
           this->layer_phase_ == GNNPhase::kBatch);

    // derivative of activation
    if (!this->config_.disable_activation) {
      this->ActivationDerivative(input_gradient);
    }

    // AFW = O
    galois::PointerWithSize<galois::GNNFloat> input_data;
    galois::PointerWithSize<galois::GNNFloat> agg_data;
    if (!this->config_.disable_dropout) {
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
    if (this->config_.disable_aggregate_after_update ||
        this->layer_dimensions_.input_columns <=
            this->layer_dimensions_.output_columns) {
      // aggdata can == p_intemp1; in other words, need to use before overwrite
      // mask it, then use it
      if (this->layer_number_ != 0) {
        if (this->graph_.IsSubgraphOn()) {
          this->MaskInputNonMasters(&agg_data,
                                    this->layer_dimensions_.output_rows,
                                    this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskInputNonMasters(&agg_data,
                                    this->layer_dimensions_.output_rows);
        }
      } else {
        if (this->graph_.IsSubgraphOn()) {
          this->MaskGradientNonMasters(input_gradient,
                                       this->layer_dimensions_.output_rows,
                                       this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskGradientNonMasters(input_gradient,
                                       this->layer_dimensions_.output_rows);
        }
      }

#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.GetWeightGradientsGPU(
            this->layer_dimensions_.input_rows,
            this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_columns, agg_data.data(),
            input_gradient->data(), this->p_layer_weight_gradients.data());
      } else {
#endif
        weight_gradient_timer.start();
        // temp 2 holds aggregated feature vectors from forward phase
        // use output rows since gcn can use subgraphs
        galois::CBlasSGEMM(
            CblasTrans, CblasNoTrans, this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_rows,
            this->layer_dimensions_.output_columns, agg_data.data(),
            input_gradient->data(), this->p_layer_weight_gradients_.data());
        weight_gradient_timer.stop();
#ifdef GALOIS_ENABLE_GPU
      }
#endif

      // gradient isn't masked here; only temp1, which has already been
      // overwritten = fine
      if (this->layer_number_ != 0) {
        // transposed sgemm for derivative; in_temp is output
        assert(input_gradient->size() ==
               this->layer_dimensions_.input_rows *
                   this->layer_dimensions_.output_columns);
        // pintemp1 contains (AF)'
        UpdateEmbeddingsDerivative(input_gradient->data(), p_in_temp_1_.data());
        // pback contains F'
        // derivative of aggregate is the same due to symmetric graph
        AggregateAll(this->layer_dimensions_.input_columns, p_in_temp_1_.data(),
                     this->p_backward_output_matrix_.data(),
                     &input_column_intermediates_, true);
      }
    } else {
      // TODO at this point, out_temp contains memoized FW
      // can use it to get A' = O' (FW)^T
      // aggregate occurs regardless of layer being equal to 0 because it is
      // required in this case for the weight gradient calculation
      // this is (FW)'
      AggregateAll(this->layer_dimensions_.output_columns,
                   input_gradient->data(), p_out_temp_.data(),
                   &output_column_intermediates_, true);

      if (this->layer_number_ != 0) {
        if (this->graph_.IsSubgraphOn()) {
          // Gradients for mirror nodes should be updated by their owner
          // hosts. In case of graph sampling, we should let this know whether
          // a node is a sampled master or not.
          this->MaskInputNonMasters(&input_data,
                                    this->layer_dimensions_.input_rows,
                                    this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskInputNonMasters(&input_data,
                                    this->layer_dimensions_.input_rows);
        }
      } else {
        // The first layer can zerofy non-master nodes' gradients since
        // it is the last gradient aggregation.
        // if 0 then no input to mask: mask the gradient
        // this is fine because gradient won't be used to get feature gradients
        if (this->graph_.IsSubgraphOn()) {
          this->MaskGradientNonMasters(&p_out_temp_,
                                       this->layer_dimensions_.input_rows,
                                       this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskGradientNonMasters(&p_out_temp_);
        }
      }

#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.GetWeightGradientsGPU(
            this->layer_dimensions_.input_rows,
            this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_columns, input_data.data(),
            p_out_temp_.data(), this->p_layer_weight_gradients.data());
      } else {
#endif
        weight_gradient_timer.start();
        // p_out_temp aggregated gradients from the next layer.
        // The weight gradients for this layer is calculated by
        // (The current vertex embedding x p_out_temp).
        // Vertex embedding dimension is (input row x input column),
        // p_out_temp dimension is (input row x output column),
        // and weight is (input column x output column).
        galois::CBlasSGEMM(
            CblasTrans, CblasNoTrans, this->layer_dimensions_.input_columns,
            this->layer_dimensions_.input_rows,
            this->layer_dimensions_.output_columns, input_data.data(),
            p_out_temp_.data(), this->p_layer_weight_gradients_.data());
        weight_gradient_timer.stop();
#ifdef GALOIS_ENABLE_GPU
      }
#endif

      if (this->layer_number_ != 0) {
        // can now overwrite p_backward without issue; since input gradient
        // is untouched if layer number isn't 0 this will be correct
        UpdateEmbeddingsDerivative(p_out_temp_.data(),
                                   this->p_backward_output_matrix_.data());
      }
    }

    // sync weight gradients; note aggregation sync occurs in the function call
    // already
    weight_gradient_sync_timer.start();
    this->WeightGradientSyncSum();
    weight_gradient_sync_timer.stop();

    if (!this->config_.disable_dropout && this->layer_number_ != 0) {
      this->DoDropoutDerivative();
    }

    timer.stop();
    return this->p_backward_output_matrix_;
  }

private:
  static const constexpr char* kRegionName = "GCNLayer";
  // 2 temporaries the size of the forward input; used for dropout and
  // aggregation (if either are required)
  std::vector<GNNFloat> in_temp_1_;
  std::vector<GNNFloat> in_temp_2_;
  // Temporary matrix the size of the output of the forward pass; used if
  // an intermediate op occurs before writing to the final output matrix
  std::vector<GNNFloat> out_temp_;

  // Pointer with size versions
  PointerWithSize<GNNFloat> p_in_temp_1_;
  PointerWithSize<GNNFloat> p_in_temp_2_;
  PointerWithSize<GNNFloat> p_out_temp_;

  // Each thread has a vector of size # input columns or # output columns for
  // storing intermediate results during aggregation.
  // The one used depeneds on if aggregation occurs before or after the mxm.
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      input_column_intermediates_;
  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>
      output_column_intermediates_;

  //! CPU aggregation
  void
  AggregateAllCPU(size_t column_length, const GNNFloat* node_embeddings,
                  GNNFloat* aggregate_output,
                  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*,
                  bool is_backward) {
    galois::StatTimer aggregate_all_sync_timer("AggregateSync", kRegionName);
    size_t num_nodes   = (is_backward)
                             ? this->layer_dimensions_.input_rows
                           // In case of minibatching or graph sampling,
                           // the outut row must be the samped graph's number of
                           // nodes of that layer.
                             : this->layer_dimensions_.output_rows;
    size_t last_master = *(this->graph_.end_owned());

    assert(0 == *(this->graph_.begin_owned()));

    galois::do_all(
        /* Either an original or a sampled graph iterator is used */
        galois::iterate(*(this->graph_.begin()), num_nodes),
        [&](size_t src) {
          size_t index_to_src_feature = src * column_length;
          // zero out src feature first
          for (size_t i = 0; i < column_length; i++) {
            aggregate_output[index_to_src_feature + i] = 0;
          }

          if (this->layer_phase_ == GNNPhase::kTrain ||
              this->layer_phase_ == GNNPhase::kBatch) {
            if (this->IsSampledLayer()) {
              // Check if node is part of sampled graph; ignore after
              // 0'ing if it is not sampled.
              // TODO(hc): check if SAGE also checks this
              if (!this->graph_.IsInSampledGraphSubgraph(src)) {
                return;
              }
            }
          }

          GNNFloat source_norm = 1.0;
          if (!this->config_.disable_normalization) {
            if (this->graph_.IsSubgraphOn() ||
                this->graph_.IsSubgraphViewOn()) {
              source_norm = this->graph_.GetDegreeNorm(
                  src, this->graph_user_layer_number_);
            } else {
              source_norm = this->graph_.GetGCNNormFactor(src);
            }
          }

          // init to self
          if (!this->config_.disable_self_aggregate) {
            graphs::bitset_graph_aggregate.set(this->graph_.ConvertToLID(src));
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
          auto e_beg = (is_backward) ? this->graph_.in_edge_begin(src)
                                     : this->graph_.edge_begin(src);
          auto e_end = (is_backward) ? this->graph_.in_edge_end(src)
                                     : this->graph_.edge_end(src);
          for (auto e = e_beg; e != e_end; e++) {
            if (this->layer_phase_ == GNNPhase::kTrain ||
                this->layer_phase_ == GNNPhase::kBatch) {
              if (this->IsSampledLayer()) {
                bool is_sampled = (is_backward)
                                      ? this->graph_.IsInEdgeSampled(
                                            e, this->graph_user_layer_number_)
                                      : this->graph_.IsEdgeSampled(
                                            e, this->graph_user_layer_number_);
                // ignore non-sampled nodes and edges
                if (!is_sampled) {
                  continue;
                }
              }
            }
            size_t dst = (is_backward) ? this->graph_.GetInEdgeDest(e)
                                       : this->graph_.GetEdgeDest(e);
            graphs::bitset_graph_aggregate.set(this->graph_.ConvertToLID(src));
            size_t index_to_dst_feature = dst * column_length;

            if (!this->config_.disable_normalization) {
              GNNFloat norm_scale;
              if (this->graph_.IsSubgraphOn() ||
                  this->graph_.IsSubgraphViewOn()) {
                norm_scale = (is_backward)
                                 ? this->graph_.GetDegreeNorm(
                                       dst, this->graph_user_layer_number_)
                                 : source_norm;
              } else {
                norm_scale = source_norm * this->graph_.GetGCNNormFactor(dst);
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
        galois::chunk_size<1>(), galois::steal(),
        galois::loopname("ConvolutionalAggregateAll"));
    // aggregate sync
    aggregate_all_sync_timer.start();
    this->graph_.AggregateSync(aggregate_output, column_length, is_backward,
                               num_nodes);
    aggregate_all_sync_timer.stop();
  }

  //! Performs aggregation for all nodes of the graph given the length of the
  //! vector to aggregate, the features themselves, an output array, and per
  //! thread storage for the intermediate scaling via norm factor
  void AggregateAll(
      size_t column_length, const GNNFloat* node_embeddings,
      GNNFloat* aggregate_output,
      galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts) {
    AggregateAll(column_length, node_embeddings, aggregate_output, pts, false);
  }

  void
  AggregateAll(size_t column_length, const GNNFloat* node_embeddings,
               GNNFloat* aggregate_output,
               galois::substrate::PerThreadStorage<std::vector<GNNFloat>>* pts,
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
      size_t last_master = *(this->graph_.end_owned());
      gpu_object_.AggregateAllGPU(
          this->graph_.GetGPUGraph(), this->graph_.size(), column_length,
          node_embeddings, aggregate_output,
          !this->config_.disable_normalization,
          this->config_.disable_self_aggregate, last_master);
      this->graph_.AggregateSyncGPU(aggregate_output, column_length,
                                    this->layer_number_);
    } else {
#endif
      AggregateAllCPU(column_length, node_embeddings, aggregate_output, pts,
                      is_backward);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    timer.stop();
  }

  //! Do embedding update via mxm with this layer's weights (forward)
  void UpdateEmbeddings(const GNNFloat* node_embeddings, GNNFloat* output) {
    galois::StatTimer timer("ForwardXform", kRegionName);
    timer.start();

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.UpdateEmbeddingsGPU(this->layer_dimensions_.input_rows,
                                      this->layer_dimensions_.input_columns,
                                      this->layer_dimensions_.output_columns,
                                      node_embeddings,
                                      base_gpu_object_.layer_weights(), output);
    } else {
#endif
      // CPU version is just a call into CBlas
      galois::CBlasSGEMM(
          CblasNoTrans, CblasNoTrans,
          this->layer_dimensions_.input_rows /* Graph or sampled graph nodes */,
          this->layer_dimensions_.input_columns,
          this->layer_dimensions_.output_columns,
          node_embeddings /* input row x input columns */,
          this->layer_weights_.data() /* input column x output column */,
          output);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    timer.stop();
  }

  //! Calculate graident via mxm with last layer's gradients (backward)
  void UpdateEmbeddingsDerivative(const GNNFloat* gradients, GNNFloat* output) {
    galois::StatTimer timer("BackwardXform", kRegionName);
    timer.start();

    assert(this->p_layer_weights_.size() ==
           this->layer_dimensions_.input_columns *
               this->layer_dimensions_.output_columns);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.UpdateEmbeddingsDerivativeGPU(
          this->layer_dimensions_.input_rows,
          this->layer_dimensions_.input_columns,
          this->layer_dimensions_.output_columns, gradients,
          base_gpu_object_.layer_weights(), output);
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
    timer.stop();
  }

#ifdef GALOIS_ENABLE_GPU
  GCNGPUAllocations gpu_object_;
#endif
};

} // namespace galois

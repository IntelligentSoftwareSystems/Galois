#pragma once
#include "galois/layers/GNNLayer.h"
#include "galois/layers/GradientSyncStructures.h"
#include "galois/GNNMath.h"
#include "galois/Logging.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/layers/SAGELayer.cuh"
#endif

namespace galois {

extern galois::DynamicBitSet graphs::bitset_graph_aggregate;

struct SAGELayerConfig {
  bool disable_concat{false};
};

// TODO(loc) move common functionality with GCN layer to common parent class
// (e.g. inits): cleans up Dense code a bit as well

//! Same as GCN layer except for the following:
//! - Mean aggregation; no symmetric norm with sqrts used (this
//! ends up performing better for some graphs)
//! - Concatination of the self: rather than aggregating self
//! feature it is concatinated (i.e. dimensions are doubled)
template <typename VTy, typename ETy>
class SAGELayer : public GNNLayer<VTy, ETy> {
public:
  //! Initializes the variables of the base class and also allocates additional
  //! memory for temporary matrices. Also initializes sync substrate for the
  //! weight matrix
  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions, const GNNLayerConfig& config,
            const SAGELayerConfig& sage_config)
      : GNNLayer<VTy, ETy>(layer_num, graph, backward_output_matrix, dimensions,
                           config),
        sage_config_(sage_config),
        input_column_intermediates_(dimensions.input_columns),
        output_column_intermediates_(dimensions.output_columns) {
    if (!sage_config_.disable_concat) {
      // there are now 2 weight matrices used: one for self, one for aggregation
      // abstractly it's one matrix: W = W1 | W2
      size_t num_weight_elements = this->layer_dimensions_.input_columns *
                                   this->layer_dimensions_.output_columns;
      galois::gInfo(this->graph_.host_prefix(), "Creating layer ",
                    this->layer_number_, ", SAGE second layer weights ",
                    num_weight_elements, " (",
                    this->FloatElementsToGB(num_weight_elements), " GB)");
      // TODO(lhc) for now, allocate dummy cpu weight2 for copying to GPU
      layer_weights_2_.resize(num_weight_elements);
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateWeight2(num_weight_elements);
      }
#endif
      galois::gInfo(this->graph_.host_prefix(), "Creating layer ",
                    this->layer_number_, ", SAGE second layer gradients ",
                    num_weight_elements, " (",
                    this->FloatElementsToGB(num_weight_elements), " GB)");
      layer_weight_gradients_2_.resize(num_weight_elements, 0);
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateWeightGradient2(num_weight_elements);
      }
#endif

      // reinit both weight matrices as one unit
      this->PairGlorotBengioInit(&this->layer_weights_, &layer_weights_2_);
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        // copy weight2 to GPU
        gpu_object_.CopyToWeights2(layer_weights_2_);
        p_layer_weights_2_ = PointerWithSize<GNNFloat>(
            gpu_object_.layer_weights_2(), num_weight_elements);
        p_layer_weight_gradients_2_ = PointerWithSize<GNNFloat>(
            gpu_object_.layer_weight_gradients_2(), num_weight_elements);
      } else {
#endif
        // update the pointers to them as well as realloc will require it
        p_layer_weights_2_ = PointerWithSize<GNNFloat>(layer_weights_2_);
        p_layer_weight_gradients_2_ =
            PointerWithSize<GNNFloat>(layer_weight_gradients_2_);
#ifdef GALOIS_ENABLE_GPU
      }
#endif
      std::vector<size_t> weight_size = {num_weight_elements};
      // initialize the optimizer
      second_weight_optimizer_ =
          std::make_unique<AdamOptimizer>(weight_size, 1);
    }

    // TODO(loc) dropout uses input rows; this won't work if dropout is enabled
    size_t num_in_temp_elements = this->layer_dimensions_.output_rows *
                                  this->layer_dimensions_.input_columns;

    // if (this->layer_number_ == 0) {
    //   // set this to true for layer 0; it avoids aggregation completely
    //   // in the last layer for the backward phase
    //   config_.disable_aggregate_after_update = true;
    //   // TODO this *will* hurt test evaluation because test eval has no
    //   // backward phase, so the end-to-end benefits do not exist there
    //   // Solution to this is to allocate all intermediate structures for both
    //   // cases + make sure resize handles both cases
    // }

    // if in temp is smaller than out temp, or if dropout exists
    if (!this->config_.disable_dropout ||
        this->config_.disable_aggregate_after_update ||
        this->layer_dimensions_.input_columns <=
            this->layer_dimensions_.output_columns) {
      galois::gInfo(this->graph_.host_prefix(), "Creating layer ",
                    this->layer_number_, ", SAGE input temp var 1 ",
                    num_in_temp_elements, " (",
                    this->FloatElementsToGB(num_in_temp_elements), " GB)");
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateInTemp1(num_in_temp_elements);
      } else {
#endif
        in_temp_1_.resize(num_in_temp_elements, 0);
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
                    this->layer_number_, ", SAGE input temp var 2 ",
                    num_in_temp_elements, " (",
                    this->FloatElementsToGB(num_in_temp_elements), " GB)");
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateInTemp2(num_in_temp_elements);
      } else {
#endif
        in_temp_2_.resize(num_in_temp_elements, 0);
#ifdef GALOIS_ENABLE_GPU
      }
#endif
    }

    size_t num_out_temp = this->layer_dimensions_.input_rows *
                          this->layer_dimensions_.output_columns;
    // only needed if out temp would be smaller than intemp
    if (!this->config_.disable_aggregate_after_update &&
        this->layer_dimensions_.input_columns >
            this->layer_dimensions_.output_columns) {
      galois::gInfo(this->graph_.host_prefix(), "Creating layer ",
                    this->layer_number_, ", SAGE output temp var ",
                    num_out_temp, " (", this->FloatElementsToGB(num_out_temp),
                    " GB)");
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.AllocateOutTemp(num_out_temp);
      } else {
#endif
        out_temp_.resize(num_out_temp, 0);
#ifdef GALOIS_ENABLE_GPU
      }
#endif
    }

    this->layer_type_ = galois::GNNLayerType::kSAGE;
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      // init pointers with size
      p_in_temp_1_ = PointerWithSize<GNNFloat>(gpu_object_.in_temp_1(),
                                               num_in_temp_elements);
      p_in_temp_2_ = PointerWithSize<GNNFloat>(gpu_object_.in_temp_2(),
                                               num_in_temp_elements);
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

    GALOIS_LOG_VERBOSE("SAGE layer initialized");
  }

  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions, const GNNLayerConfig& config)
      : SAGELayer(layer_num, graph, backward_output_matrix, dimensions, config,
                  SAGELayerConfig()) {}

  SAGELayer(size_t layer_num, const galois::graphs::GNNGraph<VTy, ETy>& graph,
            PointerWithSize<GNNFloat>* backward_output_matrix,
            const GNNLayerDimensions& dimensions)
      : SAGELayer(layer_num, graph, backward_output_matrix, dimensions,
                  GNNLayerConfig(), SAGELayerConfig()) {}

  void InitSelfWeightsTo1() {
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      size_t layer_weights_2_size = p_layer_weights_2_.size();
      if (layer_weights_2_size > 0) {
        base_gpu_object_.InitGPUVectorTo1(gpu_object_.layer_weights_2(),
                                          layer_weights_2_size);
      }
    } else {
#endif
      if (layer_weights_2_.size()) {
        layer_weights_2_.assign(layer_weights_2_.size(), 1);
      }
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  //! Returns the 2nd set of weight gradients
  const PointerWithSize<GNNFloat> GetLayerWeightGradients2() {
    return p_layer_weight_gradients_2_;
  }

  // Parent functions
  const PointerWithSize<galois::GNNFloat>
  ForwardPhase(const PointerWithSize<galois::GNNFloat> input_embeddings) final {
    // galois::gDebug(
    //    "Layer ", this->layer_number_, " dims: ",
    //    layer_dimensions_.input_rows, " ", layer_dimensions_.output_rows, " ",
    //    layer_dimensions_.input_columns, "
    //    ", layer_dimensions_.output_columns, " ", input_embeddings.size(), "
    //    ", layer_dimensions_.input_rows * layer_dimensions_.input_columns);
    galois::StatTimer timer("ForwardPhase", kRegionName);
    this->TimerStart(&timer);

    assert(input_embeddings.size() >= (this->layer_dimensions_.input_rows *
                                       this->layer_dimensions_.input_columns));
    assert(this->p_forward_output_matrix_.size() >=
           (this->layer_dimensions_.output_rows *
            this->layer_dimensions_.output_columns));

    // pointer to input to operate on
    const GNNFloat* input_data = input_embeddings.data();
    GNNFloat* agg_data;
    // first, dropout
    if (!this->config_.disable_dropout &&
        (this->layer_phase_ == GNNPhase::kTrain)) {
      this->DoDropout(input_embeddings, &p_in_temp_1_);
      input_data = p_in_temp_1_.data();
      agg_data   = p_in_temp_2_.data();
    } else {
      agg_data = p_in_temp_1_.data();
    }

    // O = FW1 + AFW2 is what is done if concat is on: below is the AFW2 part
    // which is done regardless

    // flip aggregate/update if dimensions favor it (do less work)
    if (this->config_.disable_aggregate_after_update ||
        this->layer_dimensions_.input_columns <=
            this->layer_dimensions_.output_columns) {
      if (!this->config_.disable_dropout &&
          (this->layer_phase_ == GNNPhase::kTrain)) {
        assert(p_in_temp_2_.size() >=
               this->layer_dimensions_.output_rows *
                   this->layer_dimensions_.input_columns);
      } else {
        assert(p_in_temp_1_.size() >=
               this->layer_dimensions_.output_rows *
                   this->layer_dimensions_.input_columns);
      }

      // aggregation and update
      AggregateAll(this->layer_dimensions_.input_columns, input_data, agg_data,
                   &input_column_intermediates_);
      assert(this->p_forward_output_matrix_.size() >=
             this->layer_dimensions_.output_rows *
                 this->layer_dimensions_.output_columns);
      UpdateEmbeddings(agg_data, this->p_forward_output_matrix_.data(), true);
    } else {
      assert(p_out_temp_.size() >= this->layer_dimensions_.input_rows *
                                       this->layer_dimensions_.output_columns);

      // update to aggregate
      // FW
      UpdateEmbeddings(input_data, p_out_temp_.data(), false);

      // A(FW)
      assert(this->p_forward_output_matrix_.size() >=
             this->layer_dimensions_.output_rows *
                 this->layer_dimensions_.output_columns);
      AggregateAll(this->layer_dimensions_.output_columns, p_out_temp_.data(),
                   this->p_forward_output_matrix_.data(),
                   &output_column_intermediates_);
    }

    if (!sage_config_.disable_concat) {
      // FW1 is unaffected by the agg/update flip, so can to it
      // separately
      SelfFeatureUpdateEmbeddings(input_data,
                                  this->p_forward_output_matrix_.data());
    }

    if (!this->config_.disable_activation) {
      GALOIS_LOG_VERBOSE("Doing activation");
      this->Activation();
    }

    assert(this->p_forward_output_matrix_.size() >=
           (this->layer_dimensions_.output_rows *
            this->layer_dimensions_.output_columns));

    this->TimerStop(&timer);

    return this->p_forward_output_matrix_;
  }

  PointerWithSize<galois::GNNFloat>
  BackwardPhase(PointerWithSize<galois::GNNFloat> prev_layer_input,
                PointerWithSize<galois::GNNFloat>* input_gradient) final {
    galois::StatTimer timer("BackwardPhase", kRegionName);
    galois::StatTimer weight_gradient_sync_timer("BackwardPhaseWeightSync",
                                                 kRegionName);
    galois::StatTimer weight_gradient_sync_timer2("BackwardPhaseWeight2Sync",
                                                  kRegionName);
    this->TimerStart(&timer);

    assert(this->layer_phase_ == GNNPhase::kTrain ||
           this->layer_phase_ == GNNPhase::kBatch);

    // derivative of activation
    if (!this->config_.disable_activation) {
      this->ActivationDerivative(input_gradient);
    }

    // if dropout was used, use the dropout matrix for the input
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

    // aggregate this here before gradient starts to get overwritten
    // this is xform ffirst
    if (!this->config_.disable_aggregate_after_update &&
        this->layer_dimensions_.input_columns >
            this->layer_dimensions_.output_columns) {
      // aggregate occurs regardless of layer being equal to 0 because it is
      // required in this case for the weight gradient calculation
      // this is (FW)'
      // TODO: this is absolutely terrible performance wise as well; keep
      // in mind
      AggregateAll(this->layer_dimensions_.output_columns,
                   input_gradient->data(), p_out_temp_.data(),
                   &output_column_intermediates_, true);
    }

    if (!sage_config_.disable_concat) {
      if (this->layer_number_ != 0) {
        if (this->graph_.IsSubgraphOn()) {
          this->MaskInputNonMasters(&input_data,
                                    this->layer_dimensions_.input_rows,
                                    this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskInputNonMasters(&input_data,
                                    this->layer_dimensions_.input_rows);
        }
      } else {
        // if 0 then no input to mask: mask the gradient
        // this is fine because gradient won't be used to get feature gradients
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
        gpu_object_.UpdateWeight2DerivativeGPU(
            this->layer_dimensions_.input_columns,
            this->layer_dimensions_.input_rows,
            this->layer_dimensions_.output_columns, input_data.data(),
            input_gradient->data(), p_layer_weight_gradients_2_.data());
      } else {
#endif
        // input data (prev layer input or temp1) or gradient need mask
        // can mask gradient if layer == 0
        // otherwise must mask other

        galois::StatTimer concat_grad_timer("ConcatGradMultiply", kRegionName);
        this->TimerStart(&concat_grad_timer);
        galois::CBlasSGEMM(
            CblasTrans, CblasNoTrans, this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_rows,
            this->layer_dimensions_.output_columns, input_data.data(),
            input_gradient->data(), p_layer_weight_gradients_2_.data());
        this->TimerStop(&concat_grad_timer);

#ifdef GALOIS_ENABLE_GPU
      }
#endif
    }

    weight_gradient_sync_timer2.start();
    this->WeightGradientSyncSum2();
    weight_gradient_sync_timer2.stop();

    // derivative of aggregation/update
    // TODO clean up logic here to reduce nesting
    if (this->config_.disable_aggregate_after_update ||
        this->layer_dimensions_.input_columns <=
            this->layer_dimensions_.output_columns) {
      // aggdata can == p_intemp1; in other words, need to use before overwrite
      // mask it, then use it
      // XXX masking may not be required in sampling case where rows change
      if (this->layer_number_ != 0 || sage_config_.disable_concat) {
        if (this->graph_.IsSubgraphOn()) {
          this->MaskInputNonMasters(&agg_data,
                                    this->layer_dimensions_.output_rows,
                                    this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskInputNonMasters(&agg_data,
                                    this->layer_dimensions_.output_rows);
        }
      }

#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        // XXX output rows
        gpu_object_.GetWeightGradientsGPU(
            this->layer_dimensions_.input_rows,
            this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_columns, agg_data.data(),
            input_gradient->data(), this->p_layer_weight_gradients_.data());
      } else {
#endif
        // agg data holds aggregated feature vectors from forward phase
        galois::StatTimer normal_grad_timer("NormalGradMultiply", kRegionName);
        this->TimerStart(&normal_grad_timer);
        galois::CBlasSGEMM(
            CblasTrans, CblasNoTrans, this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_rows,
            this->layer_dimensions_.output_columns, agg_data.data(),
            input_gradient->data(), this->p_layer_weight_gradients_.data());
        this->TimerStop(&normal_grad_timer);
#ifdef GALOIS_ENABLE_GPU
      }
#endif

      // 0 means input gradient shouldn't get masked
      if (this->layer_number_ != 0) {
        // NOTE: this is super nice because it avoids aggregation completely
        // in the layer 0 setting
        // ---unmasked---
        // transposed sgemm for derivative; in_temp is output
        assert(input_gradient->size() >=
               this->layer_dimensions_.output_rows *
                   this->layer_dimensions_.output_columns);
        // pintemp1 contains (AF)'
        // overwrites the dropout matrix that was in ptemp1 (needed for second
        // weight matrix)
        UpdateEmbeddingsDerivative(input_gradient->data(), p_in_temp_1_.data(),
                                   true);

        // pback contains F'
        // derivative of aggregate is the same due to symmetric graph
        AggregateAll(this->layer_dimensions_.input_columns, p_in_temp_1_.data(),
                     this->p_backward_output_matrix_.data(),
                     &input_column_intermediates_, true);
      }
    } else {
      // xform first

      // --unmasked--

      // disable concat is part of condition because otherwise this mask
      // should have gotten done elsewhere
      if (this->layer_number_ != 0 && sage_config_.disable_concat) {
        if (this->graph_.IsSubgraphOn()) {
          this->MaskInputNonMasters(&input_data,
                                    this->layer_dimensions_.input_rows,
                                    this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskInputNonMasters(&input_data,
                                    this->layer_dimensions_.input_rows);
        }
      }

      // layer number 0 means output needs to be masked because input cannot
      // be masked
      if (this->layer_number_ == 0) {
        // if 0 then no input to mask: mask the gradient
        // this is fine because gradient won't be used to get feature gradients
        if (this->graph_.IsSubgraphOn()) {
          this->MaskGradientNonMasters(&p_out_temp_,
                                       this->layer_dimensions_.input_rows,
                                       this->graph_.GetNonLayerZeroMasters());
        } else {
          this->MaskGradientNonMasters(&p_out_temp_,
                                       this->layer_dimensions_.input_rows);
        }
      }

      // W' = F^T (FW)'
      // TODO put this in a function
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        gpu_object_.GetWeightGradientsGPU(
            this->layer_dimensions_.input_rows,
            this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_columns, input_data.data(),
            p_out_temp_.data(), this->p_layer_weight_gradients_.data());
      } else {
#endif
        // input col x input row * input row x output col
        galois::StatTimer normal_grad_timer("NormalGradMultiply", kRegionName);
        this->TimerStart(&normal_grad_timer);
        galois::CBlasSGEMM(
            CblasTrans, CblasNoTrans, this->layer_dimensions_.input_columns,
            this->layer_dimensions_.input_rows,
            this->layer_dimensions_.output_columns, input_data.data(),
            p_out_temp_.data(), this->p_layer_weight_gradients_.data());
        this->TimerStop(&normal_grad_timer);
#ifdef GALOIS_ENABLE_GPU
      }
#endif

      // to get a correct result out temp mask cannot be masked;
      // outtemp will only be masked if layer number is 0, so this
      // is safe in all other cases
      if (this->layer_number_ != 0) {
        // derivative for update
        // backout = F'
        UpdateEmbeddingsDerivative(
            p_out_temp_.data(), this->p_backward_output_matrix_.data(), false);
      }
    }

    weight_gradient_sync_timer.start();
    this->WeightGradientSyncSum();
    weight_gradient_sync_timer.stop();

    // full gradient needed here; should occur after all updates
    if (this->layer_number_ != 0) {
      // deal with feature gradients for the self feature here
      // this function will sum directly into the backward matrix
      // input gradient never gets masked if layer number != 0
      SelfFeatureUpdateEmbeddingsDerivative(
          input_gradient->data(), this->p_backward_output_matrix_.data());
    }

    if (!this->config_.disable_dropout && this->layer_number_ != 0) {
      this->DoDropoutDerivative();
    }

    this->TimerStop(&timer);
    return this->p_backward_output_matrix_;
  }

#ifdef GALOIS_ENABLE_GPU
  //! Copies over self weight gradients to CPU from GPU
  const std::vector<GNNFloat>& CopyWeight2GradientsFromGPU() {
    if (!layer_weight_gradients_2_.size()) {
      layer_weight_gradients_2_.resize(p_layer_weight_gradients_2_.size());
    }
    gpu_object_.CopyWeight2GradientsToCPU(&layer_weight_gradients_2_);
    return layer_weight_gradients_2_;
  }
#endif

private:
  static const constexpr char* kRegionName = "SAGELayer";

  //! CPU aggregation
  void
  AggregateAllCPU(size_t column_length, const GNNFloat* node_embeddings,
                  GNNFloat* aggregate_output,
                  galois::substrate::PerThreadStorage<std::vector<GNNFloat>>*,
                  bool is_backward) {
    // aggregation causes a row count change
    size_t num_rows_to_handle;
    if (!is_backward) {
      num_rows_to_handle = this->layer_dimensions_.output_rows;
    } else {
      num_rows_to_handle = this->layer_dimensions_.input_rows;
    }

    galois::do_all(
        galois::iterate(*(this->graph_.begin()), num_rows_to_handle),
        [&](size_t src) {
          size_t index_to_src_feature = src * column_length;
          // zero out src feature first
          for (size_t i = 0; i < column_length; i++) {
            aggregate_output[index_to_src_feature + i] = 0;
          }

          GNNFloat source_norm = 0.0;
          if (!this->config_.disable_normalization) {
            source_norm =
                this->graph_.GetDegreeNorm(src, this->graph_user_layer_number_);
          }

          if (!is_backward) {
            // loop through all destinations to grab the feature to aggregate
            for (auto e = this->graph_.edge_begin(src);
                 e != this->graph_.edge_end(src); e++) {
              if (this->layer_phase_ == GNNPhase::kTrain ||
                  this->layer_phase_ == GNNPhase::kBatch) {
                // XXX
                // galois::gDebug("In here");
                if (this->IsSampledLayer()) {
                  if (!this->graph_.IsEdgeSampled(
                          e, this->graph_user_layer_number_)) {
                    continue;
                  }
                }
              }
              size_t dst = this->graph_.GetEdgeDest(e);
              graphs::bitset_graph_aggregate.set(
                  this->graph_.ConvertToLID(src));
              size_t index_to_dst_feature = dst * column_length;

              if (!this->config_.disable_normalization) {
                GNNFloat norm_scale = source_norm;
                assert(norm_scale != 0);

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
            for (auto e = this->graph_.in_edge_begin(src);
                 e != this->graph_.in_edge_end(src); e++) {
              if (this->layer_phase_ == GNNPhase::kTrain ||
                  this->layer_phase_ == GNNPhase::kBatch) {
                // XXX
                if (this->IsSampledLayer()) {
                  if (!this->graph_.IsInEdgeSampled(
                          e, this->graph_user_layer_number_)) {
                    continue;
                  }
                }
              }
              size_t dst = this->graph_.GetInEdgeDest(e);
              graphs::bitset_graph_aggregate.set(
                  this->graph_.ConvertToLID(src));

              // input row x output row in backward means that i shouldn't be
              // touching nodes past output rows; the above sample check
              // should deal with this where this matters
              assert(dst < this->layer_dimensions_.output_rows);

              size_t index_to_dst_feature = dst * column_length;

              if (!this->config_.disable_normalization) {
                GNNFloat norm_scale = this->graph_.GetDegreeNorm(
                    dst, this->graph_user_layer_number_);

                assert(norm_scale != 0);

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
        galois::loopname("SAGEAggregateAll"));
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
    std::string agg_timer_name      = "AggregateCompute";
    std::string agg_sync_timer_name = "AggregateSync";
    size_t num_rows_to_handle;
    if (!is_backward) {
      agg_timer_name += "Forward";
      agg_sync_timer_name += "Forward";
      num_rows_to_handle = this->layer_dimensions_.output_rows;
    } else {
      agg_timer_name += "Backward";
      agg_sync_timer_name += "Backward";
      num_rows_to_handle = this->layer_dimensions_.input_rows;
    }
    galois::StatTimer timer(agg_timer_name.c_str(), kRegionName);
    galois::StatTimer aggregate_all_sync_timer(agg_sync_timer_name.c_str(),
                                               kRegionName);
    this->TimerStart(&timer);

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      if (!this->IsSampledLayer()) {
        gpu_object_.AggregateAllGPU(
            this->graph_.GetGPUGraph(), this->graph_.size(), column_length,
            node_embeddings, aggregate_output,
            !this->config_.disable_normalization, is_backward);
      } else {
        // TODO(hochan)
        GALOIS_LOG_FATAL("SAMPLING IMPLEMENTATION");
      }
      this->graph_.AggregateSyncGPU(aggregate_output, column_length,
                                    this->layer_number_);
    } else {
#endif
      AggregateAllCPU(column_length, node_embeddings, aggregate_output, pts,
                      is_backward);
      this->TimerStop(&timer);

      // aggregate sync
      aggregate_all_sync_timer.start();
      this->graph_.AggregateSync(aggregate_output, column_length, is_backward,
                                 num_rows_to_handle);
      aggregate_all_sync_timer.stop();
#ifdef GALOIS_ENABLE_GPU
    }
#endif
  }

  //! Do embedding update via mxm with this layer's weights (forward)
  void UpdateEmbeddings(const GNNFloat* node_embeddings, GNNFloat* output,
                        bool after) {
    galois::StatTimer timer("ForwardXForm", kRegionName);
    this->TimerStart(&timer);
#ifdef GALOIS_ENABLE_GPU
    // TODO self change
    // XXX(hochan) output rows
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.UpdateEmbeddingsGPU(this->layer_dimensions_.input_rows,
                                      this->layer_dimensions_.input_columns,
                                      this->layer_dimensions_.output_columns,
                                      node_embeddings,
                                      base_gpu_object_.layer_weights(), output);
    } else {
#endif
      // galois::gDebug("Layer ", this->graph_user_layer_number_, " ",
      //               layer_dimensions_.output_rows, " ",
      //               layer_dimensions_.input_columns, " ",
      //               layer_dimensions_.output_columns);
      // CPU version is just a call into CBlas
      if (after) {
        galois::CBlasSGEMM(
            CblasNoTrans, CblasNoTrans, this->layer_dimensions_.output_rows,
            this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_columns, node_embeddings,
            this->p_layer_weights_.data(), output);
      } else {
        galois::CBlasSGEMM(
            CblasNoTrans, CblasNoTrans, this->layer_dimensions_.input_rows,
            this->layer_dimensions_.input_columns,
            this->layer_dimensions_.output_columns, node_embeddings,
            this->p_layer_weights_.data(), output);
      }
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    this->TimerStop(&timer);
  }

  //! Same as above but uses the second set of weights (self feature weights)
  void SelfFeatureUpdateEmbeddings(const GNNFloat* node_embeddings,
                                   GNNFloat* output) {
    galois::StatTimer timer("SelfForwardXForm", kRegionName);
    this->TimerStart(&timer);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.SelfFeatureUpdateEmbeddingsGPU(
          this->layer_dimensions_.input_rows,
          this->layer_dimensions_.input_columns,
          this->layer_dimensions_.output_columns, node_embeddings, output);
    } else {
#endif
      // note use of layer weights 2 differentiates this from above
      galois::CBlasSGEMM(
          CblasNoTrans, CblasNoTrans, this->layer_dimensions_.output_rows,
          this->layer_dimensions_.input_columns,
          this->layer_dimensions_.output_columns, node_embeddings,
          layer_weights_2_.data(), output, true);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    this->TimerStop(&timer);
  }

  //! Calculate graident via mxm with last layer's gradients (backward)
  void UpdateEmbeddingsDerivative(const GNNFloat* gradients, GNNFloat* output,
                                  bool after) {
    galois::StatTimer timer("BackwardXForm", kRegionName);
    this->TimerStart(&timer);

    assert(this->p_layer_weights_.size() >=
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
      // z normally); result is x by y note input rows is used here due to
      // transpose of aggregation
      if (after) {
        galois::CBlasSGEMM(CblasNoTrans, CblasTrans,
                           this->layer_dimensions_.output_rows,
                           this->layer_dimensions_.output_columns,
                           this->layer_dimensions_.input_columns, gradients,
                           this->p_layer_weights_.data(), output);
      } else {
        galois::CBlasSGEMM(CblasNoTrans, CblasTrans,
                           this->layer_dimensions_.input_rows,
                           this->layer_dimensions_.output_columns,
                           this->layer_dimensions_.input_columns, gradients,
                           this->p_layer_weights_.data(), output);
      }
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    this->TimerStop(&timer);
  }

  //! Same as above but uses the second set of weights (self feature weights)
  void SelfFeatureUpdateEmbeddingsDerivative(const GNNFloat* gradients,
                                             GNNFloat* output) {
    galois::StatTimer timer("SelfBackwardXForm", kRegionName);
    this->TimerStart(&timer);

    assert(this->p_layer_weights_.size() >=
           this->layer_dimensions_.input_columns *
               this->layer_dimensions_.output_columns);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_object_.SelfFeatureUpdateEmbeddingsDerivativeGPU(
          this->layer_dimensions_.input_rows,
          this->layer_dimensions_.output_columns,
          this->layer_dimensions_.input_columns, gradients, output);
    } else {
#endif
      // difference is Trans for B matrix (data) to get z by y (weights is y by
      // z normally); result is x by y true at end -> accumulate
      galois::CBlasSGEMM(CblasNoTrans, CblasTrans,
                         this->layer_dimensions_.output_rows,
                         this->layer_dimensions_.output_columns,
                         this->layer_dimensions_.input_columns, gradients,
                         layer_weights_2_.data(), output, true);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    this->TimerStop(&timer);
  }

  //! override parent function: optimizes the second set of weights as well
  void OptimizeLayer(BaseOptimizer* optimizer, size_t trainable_layer_number) {
    galois::StatTimer total_gradient_timer("GradientDescent", kRegionName);
    total_gradient_timer.start();
    optimizer->GradientDescent(this->p_layer_weight_gradients_,
                               this->p_layer_weights_, trainable_layer_number);
    if (!sage_config_.disable_concat) {
      second_weight_optimizer_->GradientDescent(p_layer_weight_gradients_2_,
                                                p_layer_weights_2_, 0);
    }
    total_gradient_timer.stop();
  }

  //! Sync second set of weight gradients
  void WeightGradientSyncSum2() {
    galois::StatTimer clubbed_timer("Sync_BackwardSync", "Gluon");
    this->TimerStart(&clubbed_timer);
    galois::StatTimer t("Sync_WeightGradientsSum2", kRegionName);
    this->TimerStart(&t);
    int weight_size = static_cast<int>(p_layer_weight_gradients_2_.size());

#ifdef GALOIS_ENABLE_GPU
    bool gpu_direct_enabled = false;
    if (device_personality == DevicePersonality::GPU_CUDA &&
        !gpu_direct_enabled) {
      gpu_object_.CopyWeight2GradientsToCPU(&layer_weight_gradients_2_);
      MPI_Allreduce(MPI_IN_PLACE,
                    static_cast<void*>(layer_weight_gradients_2_.data()),
                    weight_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      gpu_object_.CopyToWeight2Gradients(layer_weight_gradients_2_);
    } else {
#endif
      // TODO(loc) remove this limitation later; can just do a loop over the
      // weight matrix
      if (p_layer_weight_gradients_2_.size() >
          size_t{std::numeric_limits<int>::max()}) {
        GALOIS_LOG_FATAL(
            "Weight sync code does not handle size larger than max "
            "int at the moment");
      }
      MPI_Allreduce(MPI_IN_PLACE,
                    static_cast<void*>(p_layer_weight_gradients_2_.data()),
                    weight_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#ifdef GALOIS_ENABLE_GPU
    }
#endif
    this->TimerStop(&t);
    this->TimerStop(&clubbed_timer);
  }

  void ResizeRows(size_t new_row_count) {
    GNNLayer<VTy, ETy>::ResizeRows(new_row_count);
    ResizeIntermediates(new_row_count, new_row_count);
  }

  void ResizeInputOutputRows(size_t input_row, size_t output_row) {
    GNNLayer<VTy, ETy>::ResizeInputOutputRows(input_row, output_row);
    ResizeIntermediates(input_row, output_row);
  }

  void ResizeIntermediates(size_t new_input_rows, size_t new_output_rows) {
    size_t num_in_temp_elements =
        new_output_rows * this->layer_dimensions_.input_columns;
    // galois::gDebug(this->graph_.host_prefix(), "Layer num ",
    // this->layer_number_, " ",
    //               in_temp_1_.size(), " and ", num_in_temp_elements, " ",
    //               layer_dimensions_.input_columns, " ",
    //               layer_dimensions_.output_columns);

    // if in temp is smaller than out temp, or if dropout exists
    if (!this->config_.disable_dropout ||
        this->config_.disable_aggregate_after_update ||
        this->layer_dimensions_.input_columns <=
            this->layer_dimensions_.output_columns) {
      if (in_temp_1_.size() < num_in_temp_elements) {
        galois::gInfo(this->graph_.host_prefix(), "Resize layer ",
                      this->layer_number_, ", SAGE input temp var 1 ",
                      num_in_temp_elements, " (",
                      this->FloatElementsToGB(num_in_temp_elements), " GB)");
        size_t buffer_size = num_in_temp_elements * 0.02;
#ifdef GALOIS_ENABLE_GPU
        // XXX(hochan)
        if (device_personality == DevicePersonality::GPU_CUDA) {
          gpu_object_.AllocateInTemp1(num_in_temp_elements + buffer_size);
        } else {
#endif
          in_temp_1_.resize(num_in_temp_elements + buffer_size, 0);
#ifdef GALOIS_ENABLE_GPU
        }
#endif
        // XXX(hochan) GPU
        p_in_temp_1_ = PointerWithSize<GNNFloat>(in_temp_1_);
      }
    }

    // only on in dropout case + if in temp is smaller than out temp
    if (!this->config_.disable_dropout &&
        (this->config_.disable_aggregate_after_update ||
         this->layer_dimensions_.input_columns <=
             this->layer_dimensions_.output_columns)) {
      if (in_temp_2_.size() < num_in_temp_elements) {
        galois::gInfo(this->graph_.host_prefix(), "Resize layer ",
                      this->layer_number_, ", SAGE input temp var 2 ",
                      num_in_temp_elements, " (",
                      this->FloatElementsToGB(num_in_temp_elements), " GB)");
        size_t buffer_size = num_in_temp_elements * 0.02;
#ifdef GALOIS_ENABLE_GPU
        if (device_personality == DevicePersonality::GPU_CUDA) {
          gpu_object_.AllocateInTemp2(num_in_temp_elements + buffer_size);
        } else {
#endif
          in_temp_2_.resize(num_in_temp_elements + buffer_size, 0);
#ifdef GALOIS_ENABLE_GPU
        }
#endif
        // XXX(hochan) GPU
        p_in_temp_2_ = PointerWithSize<GNNFloat>(in_temp_2_);
      }
    }

    size_t num_output_temp_elements =
        new_input_rows * this->layer_dimensions_.output_columns;
    // only needed if out temp would be smaller than intemp
    if (!this->config_.disable_aggregate_after_update &&
        this->layer_dimensions_.input_columns >
            this->layer_dimensions_.output_columns) {
      if (out_temp_.size() < num_output_temp_elements) {
        galois::gInfo(
            this->graph_.host_prefix(), "Resize layer ", this->layer_number_,
            ", SAGE output temp var ", num_output_temp_elements, " (",
            this->FloatElementsToGB(num_output_temp_elements), " GB)");
        size_t buffer_size = (num_output_temp_elements * 0.02);
#ifdef GALOIS_ENABLE_GPU
        if (device_personality == DevicePersonality::GPU_CUDA) {
          gpu_object_.AllocateOutTemp(num_output_temp_elements + buffer_size);
        } else {
#endif
          out_temp_.resize(num_output_temp_elements + buffer_size, 0);
#ifdef GALOIS_ENABLE_GPU
        }
#endif
        p_out_temp_ = PointerWithSize<GNNFloat>(out_temp_);
      }
    }
  }

  //! SAGE config params
  SAGELayerConfig sage_config_;
  //! Need own optimizer for the 2nd weight matrix
  std::unique_ptr<AdamOptimizer> second_weight_optimizer_;

  // second set of weights for the concat that may occur
  std::vector<GNNFloat> layer_weights_2_;
  std::vector<GNNFloat> layer_weight_gradients_2_;
  PointerWithSize<GNNFloat> p_layer_weights_2_;
  PointerWithSize<GNNFloat> p_layer_weight_gradients_2_;

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

#ifdef GALOIS_ENABLE_GPU
  SAGEGPUAllocations gpu_object_;
#endif
};

} // namespace galois

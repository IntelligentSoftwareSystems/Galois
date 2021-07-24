#include "galois/GNNMath.h"
#include "galois/GraphNeuralNetwork.h"
#include "galois/layers/DenseLayer.h"
#include "galois/layers/GraphConvolutionalLayer.h"
#include "galois/layers/L2NormLayer.h"
#include "galois/layers/SAGELayer.h"
#include "galois/layers/SigmoidLayer.h"
#include "galois/layers/SoftmaxLayer.h"

galois::GraphNeuralNetwork::GraphNeuralNetwork(
    std::unique_ptr<galois::graphs::GNNGraph> graph,
    std::unique_ptr<BaseOptimizer> optimizer,
    galois::GraphNeuralNetworkConfig&& config)
    : graph_(std::move(graph)), optimizer_(std::move(optimizer)),
      config_(std::move(config)) {
  if (config_.do_sampling_ && config_.use_train_subgraph_) {
    GALOIS_LOG_FATAL("Do not set train subgraph and sampling at same time "
                     "(sampling uses training subgraph already)");
  }
  // max number of rows that can be passed as inputs; allocate space for it as
  // this will be the # of rows for each layer
  size_t max_rows = graph_->size();

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    graph_->ResizeGPULayerVector(config_.num_intermediate_layers());
  }
#endif
  // used for chaining layers together; begins as nullptr
  PointerWithSize<GNNFloat> prev_output_layer(nullptr, 0);
  num_graph_user_layers_ = 0;

  // create the intermediate layers
  for (size_t i = 0; i < config_.num_intermediate_layers(); i++) {
    GNNLayerType layer_type = config_.intermediate_layer_type(i);
    size_t prev_layer_columns;

    if (i != 0) {
      // grab previous layer's size
      prev_layer_columns = config_.intermediate_layer_size(i - 1);
    } else {
      // first layer means the input columns are # features in graph
      prev_layer_columns = graph_->node_feature_length();
    }

    // max dims
    GNNLayerDimensions layer_dims = {.input_rows    = max_rows,
                                     .input_columns = prev_layer_columns,
                                     .output_columns =
                                         config_.intermediate_layer_size(i),
                                     .output_rows = max_rows};

    // test minibatch size: if it's not enabled, then currently the full
    // graph is used (should really only subgraph the test nodes, though;
    // that's a TODO)
    if ((config_.train_minibatch_size() || config_.use_train_subgraph_) &&
        config_.test_minibatch_size()) {
      galois::gInfo("Not allocating rows");
      // set to 0 here to make it allocate nothing
      layer_dims.input_rows  = 0;
      layer_dims.output_rows = 0;
    }

    switch (layer_type) {
    case GNNLayerType::kGraphConvolutional:
      gnn_layers_.push_back(std::move(std::make_unique<GraphConvolutionalLayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
      gnn_layers_.back()->SetGraphUserLayerNumber(num_graph_user_layers_++);
      break;
    case GNNLayerType::kSAGE:
      gnn_layers_.push_back(std::move(std::make_unique<SAGELayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
      gnn_layers_.back()->SetGraphUserLayerNumber(num_graph_user_layers_++);
#ifdef GALOIS_ENABLE_GPU
      // TODO(loc/hochan) sage layer gpu
#endif
      break;
    case GNNLayerType::kL2Norm:
      gnn_layers_.push_back(std::move(std::make_unique<L2NormLayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
      break;
    case GNNLayerType::kDense:
      gnn_layers_.push_back(std::move(std::make_unique<DenseLayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
      break;
    default:
      GALOIS_LOG_FATAL("Invalid layer type during network construction");
    }

    // update output layer for next layer
    prev_output_layer = gnn_layers_.back()->GetForwardOutput();
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      graph_->InitLayerVectorMetaObjects(
          i, galois::runtime::getSystemNetworkInterface().Num,
          layer_dims.input_columns, layer_dims.output_columns);
    }
#endif
  }

  // loop backward and find last GCN/SAGE (main) layer to disable activation
  for (auto back_iter = gnn_layers_.rbegin(); back_iter != gnn_layers_.rend();
       back_iter++) {
    GNNLayerType layer_type = (*back_iter)->layer_type();
    if (layer_type == GNNLayerType::kGraphConvolutional ||
        layer_type == GNNLayerType::kSAGE) {
      galois::gDebug("Disabling activation on layer ",
                     (*back_iter)->layer_number(), "\n");
      (*back_iter)->DisableActivation();
      break;
    }
  }

  if (config_.do_sampling() || config_.use_train_subgraph_ ||
      config.train_minibatch_size() || config.test_minibatch_size()) {
    // output layer not included; it will never involve sampling
    graph_->InitializeSamplingData(num_graph_user_layers_,
                                   config_.use_train_subgraph_);
  }

  num_hosts_ = galois::runtime::getSystemNetworkInterface().Num;
  if (config_.train_minibatch_size()) {
    size_t local_num =
        graph_->SetupTrainBatcher(config_.train_minibatch_size());
    if (num_hosts_ > 1) {
      dist_minibatch_tracker_ = std::make_unique<DistributedMinibatchTracker>(
          galois::runtime::getSystemNetworkInterface().ID, num_hosts_,
          local_num, config_.train_minibatch_size());
    }
  }

  if (config_.test_minibatch_size()) {
    graph_->SetupTestBatcher(config_.test_minibatch_size());
  }

  // create the output layer
  GNNLayerDimensions output_dims = {
      .input_rows = max_rows,
      // get last intermediate layer column size
      .input_columns = config_.intermediate_layer_size(
          config_.num_intermediate_layers() - 1),
      .output_columns = config_.output_layer_size(),
      .output_rows    = max_rows};

  if ((config_.train_minibatch_size() || config_.use_train_subgraph_) &&
      config_.test_minibatch_size()) {
    output_dims.input_rows  = 0;
    output_dims.output_rows = 0;
  }

  switch (config_.output_layer_type()) {
  case (GNNOutputLayerType::kSoftmax):
    gnn_layers_.push_back(std::move(std::make_unique<SoftmaxLayer>(
        config_.num_intermediate_layers(), *graph_, &prev_output_layer,
        output_dims)));
    break;
  case (GNNOutputLayerType::kSigmoid):
    gnn_layers_.push_back(std::move(std::make_unique<SigmoidLayer>(
        config_.num_intermediate_layers(), *graph_, &prev_output_layer,
        output_dims)));
    break;
  default:
    GALOIS_LOG_FATAL("Invalid layer type during network construction");
  }

  // sanity checking multi-class + output layer
  if (!graph_->is_single_class_label() &&
      (config_.output_layer_type() != GNNOutputLayerType::kSigmoid)) {
    GALOIS_LOG_WARN(
        "Using a non-sigmoid output layer with a multi-class label!");
    // if debug mode just kill program
    assert(false);
  }

  // flip sampling on layers
  if (config_.use_train_subgraph_ || config_.do_sampling() ||
      config_.train_minibatch_size()) {
    for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
      ptr->EnableSampling();
    }
  }
}

float galois::GraphNeuralNetwork::MinibatchedTesting() {
  galois::gDebug("Minibatched Testing");
  graph_->DisableSubgraph();
  graph_->ResetTestMinibatcher();
  SetLayerPhases(galois::GNNPhase::kBatch);

  bool choose_all_status = graph_->SubgraphChooseAllStatus();

  uint32_t correct = 0;
  uint32_t total   = 0;
  while (true) {
    work_left_.reset();
    // size_t seed_node_count = graph_->PrepareNextTestMinibatch();
    graph_->PrepareNextTestMinibatch();
    // last layer input size/output rows becomes seed node size
    // gnn_layers_.back()->ResizeInputOutputRows(seed_node_count,
    // seed_node_count);
    size_t num_sampled_layers = 0;

    for (auto back_iter = gnn_layers_.rbegin(); back_iter != gnn_layers_.rend();
         back_iter++) {
      GNNLayerType layer_type = (*back_iter)->layer_type();
      if (layer_type == GNNLayerType::kGraphConvolutional ||
          layer_type == GNNLayerType::kSAGE) {
        // you can minibatch with sampling or minibatch and grab all
        // relevant neighbors
        // size_t current_sample_size;
        graph_->SampleAllEdges((*back_iter)->graph_user_layer_number(), false,
                               num_sampled_layers + 1);
        // resize this layer, change seed node count
        //(*back_iter)
        //    ->ResizeInputOutputRows(current_sample_size, seed_node_count);
        // seed_node_count = current_sample_size;

        num_sampled_layers++;
        // XXX resizes above only work for SAGE layers; will break if other
        // layers are tested
      }
    }

    // resize layer matrices
    CorrectRowCounts(graph_->ConstructSampledSubgraph(num_sampled_layers));
    graph_->EnableSubgraphChooseAll();
    CorrectBackwardLinks();

    const PointerWithSize<galois::GNNFloat> batch_pred = DoInference();
    std::pair<uint32_t, uint32_t> correct_total =
        graph_->GetBatchAccuracy(batch_pred);

    correct += correct_total.first;
    total += correct_total.second;

    work_left_ += graph_->MoreTestMinibatches();
    char global_work_left = work_left_.reduce();
    if (!global_work_left) {
      break;
    }
  }

  galois::gInfo("Minibatching Correct / Total ", correct, " ", total);

  if (choose_all_status) {
    graph_->EnableSubgraphChooseAll();
  } else {
    graph_->DisableSubgraphChooseAll();
  }

  return (1.0 * correct) / (1.0 * total);
}

float galois::GraphNeuralNetwork::Train(size_t num_epochs) {
  EnableTimers();
  const size_t this_host = graph_->host_id();
  float train_accuracy{0.f};
  std::vector<size_t> subgraph_layer_sizes;
  // this subgraph only needs to be created once
  if (config_.use_train_subgraph_ && !config_.train_minibatch_size()) {
    // Setup the subgraph to only be the training graph
    size_t local_seed_node_count = graph_->SetupNeighborhoodSample();
    subgraph_layer_sizes.emplace_back(local_seed_node_count);
    galois::gDebug(graph_->host_prefix(), "Number of local seed nodes is ",
                   local_seed_node_count);
    size_t num_sampled_layers = 0;
    // gnn_layers_.back()->ResizeRows(local_seed_node_count);
    for (auto back_iter = gnn_layers_.rbegin(); back_iter != gnn_layers_.rend();
         back_iter++) {
      GNNLayerType layer_type = (*back_iter)->layer_type();
      if (layer_type == GNNLayerType::kGraphConvolutional ||
          layer_type == GNNLayerType::kSAGE) {
        size_t current_sample_size = graph_->SampleAllEdges(
            (*back_iter)->graph_user_layer_number(),
            config_.inductive_subgraph_, num_sampled_layers + 1);
        galois::gDebug(graph_->host_prefix(),
                       "Number of local nodes for train subgraph for layer ",
                       (*back_iter)->graph_user_layer_number(), " is ",
                       current_sample_size);
        // resizing
        //(*back_iter)
        //    ->ResizeInputOutputRows(current_sample_size,
        //    local_seed_node_count);
        local_seed_node_count = current_sample_size;
        subgraph_layer_sizes.emplace_back(local_seed_node_count);
        num_sampled_layers++;
      }
    }
    CorrectRowCounts(graph_->ConstructSampledSubgraph(num_sampled_layers));
    CorrectBackwardLinks();
  }

  galois::StatTimer epoch_timer("TrainingTime", kRegionName);
  galois::StatTimer validation_timer("ValidationTime", kRegionName);
  galois::StatTimer epoch_test_timer("TestTime", kRegionName);

  for (size_t epoch = 0; epoch < num_epochs; epoch++) {
    epoch_timer.start();
    // swap to train subgraph
    if (config_.use_train_subgraph_ && !config_.train_minibatch_size()) {
      graph_->EnableSubgraph();
      // TODO(loc) this doesn't actually function as expected anymore
      // with the numerous changes to the system; this commenting
      // out is more of a hack for the train subgraph option (which
      // probably shouldn't be used anyways)

      // size_t l_count = 0;
      // gnn_layers_.back()->ResizeRows(subgraph_layer_sizes[0]);
      // for (auto back_iter = gnn_layers_.rbegin();
      //     back_iter != gnn_layers_.rend(); back_iter++) {
      //  GNNLayerType layer_type = (*back_iter)->layer_type();
      //  if (layer_type == GNNLayerType::kGraphConvolutional ||
      //      layer_type == GNNLayerType::kSAGE) {
      //    (*back_iter)
      //        ->ResizeInputOutputRows(subgraph_layer_sizes[l_count + 1],
      //                                subgraph_layer_sizes[l_count]);
      //    l_count++;
      //  }
      //}
      CorrectBackwardLinks();
    }

    // beginning of epoch sampling (no minibatches)
    if (config_.do_sampling() && !config_.train_minibatch_size()) {
      galois::StatTimer mb_timer("EpochSubgraphCreation", kRegionName);
      mb_timer.start();

      size_t local_seed_node_count = graph_->SetupNeighborhoodSample();
      // gnn_layers_.back()->ResizeRows(local_seed_node_count);
      galois::gDebug(graph_->host_prefix(), "Number of local seed nodes is ",
                     local_seed_node_count);
      size_t num_sampled_layers = 0;

      // work backwards on GCN/SAGE layers
      // loop backward and find last GCN/SAGE (main) layer to disable activation
      for (auto back_iter = gnn_layers_.rbegin();
           back_iter != gnn_layers_.rend(); back_iter++) {
        GNNLayerType layer_type = (*back_iter)->layer_type();
        if (layer_type == GNNLayerType::kGraphConvolutional ||
            layer_type == GNNLayerType::kSAGE) {
          size_t current_sample_size = graph_->SampleEdges(
              (*back_iter)->graph_user_layer_number(),
              config_.fan_out_vector_[num_sampled_layers],
              config_.inductive_subgraph_, num_sampled_layers + 1);
          galois::gDebug(graph_->host_prefix(),
                         "Number of local nodes for layer ",
                         (*back_iter)->graph_user_layer_number(), " is ",
                         current_sample_size);

          //(*back_iter)
          //    ->ResizeInputOutputRows(current_sample_size,
          //                            local_seed_node_count);
          local_seed_node_count = current_sample_size;
          num_sampled_layers++;
        }
      }
      // resize layer matrices
      CorrectRowCounts(graph_->ConstructSampledSubgraph(num_sampled_layers));
      CorrectBackwardLinks();
      mb_timer.stop();
    }

    if (!config_.train_minibatch_size()) {
      // no minibatching, full batch
      const PointerWithSize<galois::GNNFloat> predictions = DoInference();
      // have to get accuracy here because gradient prop destroys the
      // predictions matrix
      train_accuracy = GetGlobalAccuracy(predictions);
      GradientPropagation();
    } else {
      graph_->ResetTrainMinibatcher();
      if (num_hosts_ > 1) {
        dist_minibatch_tracker_->ResetEpoch();
      }

      SetLayerPhases(galois::GNNPhase::kBatch);

      size_t batch_num = 0;

      // create mini batch graphs and loop until minibatches on all hosts done
      while (true) {
        galois::StatTimer prep_timer("PrepNextMinibatch", kRegionName);
        galois::StatTimer sample_time("MinibatchSampling", kRegionName);
        galois::StatTimer mb_timer("MinibatchSubgraphCreation", kRegionName);
        mb_timer.start();

        galois::Timer batch_timer;
        batch_timer.start();
        work_left_.reset();
        galois::gInfo("Epoch ", epoch, " batch ", batch_num++);
        // break when all hosts are done with minibatches
        prep_timer.start();
        size_t seed_node_count;
        if (num_hosts_ > 1) {
          size_t num_for_next_batch =
              dist_minibatch_tracker_->GetNumberForNextMinibatch();
          galois::gInfo(graph_->host_prefix(), "Sampling ", num_for_next_batch,
                        " for this minibatch");
          seed_node_count =
              graph_->PrepareNextTrainMinibatch(num_for_next_batch);
        } else {
          seed_node_count = graph_->PrepareNextTrainMinibatch();
        }

        galois::gDebug(graph_->host_prefix(),
                       "Number of local seed nodes is for batch is ",
                       seed_node_count);
        prep_timer.stop();

        // last layer input size/output rows becomes seed node size
        // gnn_layers_.back()->ResizeInputOutputRows(seed_node_count,
        //                                          seed_node_count);

        sample_time.start();
        // +1 later in call because 0 is already taken
        size_t num_sampled_layers = 0;
        for (auto back_iter = gnn_layers_.rbegin();
             back_iter != gnn_layers_.rend(); back_iter++) {
          GNNLayerType layer_type = (*back_iter)->layer_type();
          if (layer_type == GNNLayerType::kGraphConvolutional ||
              layer_type == GNNLayerType::kSAGE) {
            // you can minibatch with sampling or minibatch and grab all
            // relevant neighbors
            size_t current_sample_size;

            if (config_.do_sampling()) {
              current_sample_size = graph_->SampleEdges(
                  (*back_iter)->graph_user_layer_number(),
                  config_.fan_out_vector_[num_sampled_layers],
                  config_.inductive_subgraph_, num_sampled_layers + 1);
            } else {
              current_sample_size = graph_->SampleAllEdges(
                  (*back_iter)->graph_user_layer_number(),
                  config_.inductive_subgraph_, num_sampled_layers + 1);
            }

            galois::gDebug(graph_->host_prefix(),
                           "Number of local nodes for layer ",
                           (*back_iter)->graph_user_layer_number(), " is ",
                           current_sample_size);

            // resize this layer, change seed node count
            //(*back_iter)
            //    ->ResizeInputOutputRows(current_sample_size, seed_node_count);
            seed_node_count = current_sample_size;
            num_sampled_layers++;
          }
        }
        sample_time.stop();

        // resize layer matrices
        CorrectRowCounts(graph_->ConstructSampledSubgraph(num_sampled_layers));
        CorrectBackwardLinks();

        // XXX resizes above only work for SAGE layers; will break if other
        // layers are tested

        mb_timer.stop();

        const PointerWithSize<galois::GNNFloat> batch_pred = DoInference();
        train_accuracy = GetGlobalAccuracy(batch_pred);
        GradientPropagation();

        work_left_ += graph_->MoreTrainMinibatches();
        char global_work_left = work_left_.reduce();
        batch_timer.stop();
        epoch_timer.stop();
        galois::gPrint("Epoch ", epoch, " Batch ", batch_num - 1,
                       ": Train accuracy/F1 micro is ", train_accuracy,
                       " time ", batch_timer.get(), "\n");

        bool test_eval =
            config_.minibatch_test_interval_
                ? (batch_num - 1) % config_.minibatch_test_interval_ == 0
                : false;

        if (test_eval) {
          DisableTimers();
          float test_acc;
          if (!config_.test_minibatch_size()) {
            // TODO something about this path breaks accuracy
            GALOIS_LOG_FATAL("this path breaks accuracy for the rest of the "
                             "run for some reason");
            bool f = graph_->SubgraphChooseAllStatus();
            graph_->DisableSubgraph();
            for (auto layer = gnn_layers_.begin(); layer != gnn_layers_.end();
                 layer++) {
              // TODO nuclear resize
              (*layer)->ResizeRows(graph_->size());
            }
            CorrectBackwardLinks();
            SetLayerPhases(galois::GNNPhase::kTest);
            graph_->EnableSubgraphChooseAll();
            const PointerWithSize<galois::GNNFloat> test_pred = DoInference();
            test_acc = GetGlobalAccuracy(test_pred);
            graph_->SetSubgraphChooseAll(f);
          } else {
            test_acc = MinibatchedTesting();
          }

          if (this_host == 0) {
            galois::gPrint("Epoch ", epoch, " Batch ", batch_num - 1,
                           ": Test accuracy is ", test_acc, "\n");
            const std::string test_name_acc =
                "TestEpoch" + std::to_string(epoch) + "Batch" +
                std::to_string(batch_num - 1) + "Accuracy";
            galois::runtime::reportStat_Single(kRegionName, test_name_acc,
                                               test_acc);
          }

          // report the training time elapsed at this point in time
          galois::runtime::reportStat_Single(
              kRegionName,
              "ElapsedTrainTimeEpoch" + std::to_string(epoch) + "Batch" +
                  std::to_string(batch_num - 1),
              epoch_timer.get());
          // revert to training phase for next epoch
          SetLayerPhases(galois::GNNPhase::kTrain);
          EnableTimers();
        }

        epoch_timer.start();

        if (!global_work_left) {
          if (num_hosts_ > 1) {
            GALOIS_LOG_ASSERT(dist_minibatch_tracker_->OutOfWork());
          }
          break;
        }
      }
    }
    epoch_timer.stop();

    if (this_host == 0) {
      const std::string t_name_acc =
          "TrainEpoch" + std::to_string(epoch) + "Accuracy";
      galois::gPrint("Epoch ", epoch, ": Train accuracy/F1 micro is ",
                     train_accuracy, "\n");
      galois::runtime::reportStat_Single(kRegionName, t_name_acc,
                                         train_accuracy);
    }

    bool do_validate = config_.validation_interval_
                           ? epoch % config_.validation_interval_ == 0
                           : false;
    bool do_test =
        config_.test_interval_ ? epoch % config_.test_interval_ == 0 : false;

    bool subgraph_choose_all_status = graph_->SubgraphChooseAllStatus();

    if (do_validate || do_test) {
      DisableTimers();
      // disable subgraph
      graph_->DisableSubgraph();
      graph_->EnableSubgraphChooseAll();
    }

    if (do_validate) {
      // XXX induced subgraph here
      for (auto layer = gnn_layers_.begin(); layer != gnn_layers_.end();
           layer++) {
        // nuclear resize
        (*layer)->ResizeRows(graph_->size());
      }

      CorrectBackwardLinks();
      validation_timer.start();
      SetLayerPhases(galois::GNNPhase::kValidate);
      const PointerWithSize<galois::GNNFloat> val_pred = DoInference();
      validation_timer.stop();

      float val_acc = GetGlobalAccuracy(val_pred);
      if (this_host == 0) {
        galois::gPrint("Epoch ", epoch, ": Validation accuracy is ", val_acc,
                       "\n");
        const std::string v_name_acc =
            "ValEpoch" + std::to_string(epoch) + "Accuracy";
        galois::runtime::reportStat_Single(kRegionName, v_name_acc, val_acc);
      }
    }

    if (do_test) {
      epoch_test_timer.start();
      float test_acc;

      if (!config_.test_minibatch_size()) {
        for (auto layer = gnn_layers_.begin(); layer != gnn_layers_.end();
             layer++) {
          // nuclear resize
          (*layer)->ResizeRows(graph_->size());
        }
        CorrectBackwardLinks();
        SetLayerPhases(galois::GNNPhase::kTest);
        const PointerWithSize<galois::GNNFloat> test_pred = DoInference();
        epoch_test_timer.stop();
        test_acc = GetGlobalAccuracy(test_pred);
      } else {
        test_acc = MinibatchedTesting();
        epoch_test_timer.stop();
      }

      if (this_host == 0) {
        galois::gPrint("Epoch ", epoch, ": Test accuracy is ", test_acc, "\n");
        const std::string test_name_acc =
            "TestEpoch" + std::to_string(epoch) + "Accuracy";
        galois::runtime::reportStat_Single(kRegionName, test_name_acc,
                                           test_acc);
      }
    }

    if (do_validate || do_test) {
      // report the training time elapsed at this point in time
      galois::runtime::reportStat_Single(
          kRegionName, "ElapsedTrainTimeEpoch" + std::to_string(epoch),
          epoch_timer.get());
      // revert to training phase for next epoch
      SetLayerPhases(galois::GNNPhase::kTrain);
      graph_->SetSubgraphChooseAll(subgraph_choose_all_status);

      // TODO too much code dupe
      // Resconstruct the train subgraph since it was replaced by test subgraph
      if (config_.use_train_subgraph_ && !config_.train_minibatch_size() &&
          config_.test_minibatch_size() && do_test) {
        // Setup the subgraph to only be the training graph
        size_t local_seed_node_count = graph_->SetupNeighborhoodSample();
        galois::gDebug(graph_->host_prefix(), "Number of local seed nodes is ",
                       local_seed_node_count);
        size_t num_sampled_layers = 0;
        // gnn_layers_.back()->ResizeRows(local_seed_node_count);
        for (auto back_iter = gnn_layers_.rbegin();
             back_iter != gnn_layers_.rend(); back_iter++) {
          GNNLayerType layer_type = (*back_iter)->layer_type();
          if (layer_type == GNNLayerType::kGraphConvolutional ||
              layer_type == GNNLayerType::kSAGE) {
            size_t current_sample_size = graph_->SampleAllEdges(
                (*back_iter)->graph_user_layer_number(),
                config_.inductive_subgraph_, num_sampled_layers + 1);
            // resizing
            //(*back_iter)
            //    ->ResizeInputOutputRows(current_sample_size,
            //                            local_seed_node_count);
            local_seed_node_count = current_sample_size;
            num_sampled_layers++;
          }
        }
        CorrectRowCounts(graph_->ConstructSampledSubgraph(num_sampled_layers));
        CorrectBackwardLinks();
      }

      EnableTimers();
    }
  }

  uint64_t average_epoch_time = epoch_timer.get() / num_epochs;
  galois::runtime::reportStat_Tavg(kRegionName, "AverageEpochTime",
                                   average_epoch_time);
  DisableTimers();
  // disable subgraph
  graph_->DisableSubgraph();
  graph_->EnableSubgraphChooseAll();

  // check test accuracy
  galois::StatTimer test_timer("FinalTestRun", kRegionName);
  float global_accuracy;

  test_timer.start();

  if (!config_.test_minibatch_size()) {
    for (auto layer = gnn_layers_.begin(); layer != gnn_layers_.end();
         layer++) {
      // TODO nuclear resize; this is **ridiculously** inefficient
      // because full graph will be used even if not included in test
      // k-hop neighborhood for eval
      (*layer)->ResizeRows(graph_->size());
    }
    CorrectBackwardLinks();
    SetLayerPhases(galois::GNNPhase::kTest);
    const PointerWithSize<galois::GNNFloat> predictions = DoInference();
    global_accuracy = GetGlobalAccuracy(predictions);
  } else {
    global_accuracy = MinibatchedTesting();
  }

  test_timer.stop();

  if (this_host == 0) {
    galois::gPrint("Final test accuracy is ", global_accuracy, "\n");
    galois::runtime::reportStat_Single(kRegionName, "FinalTestAccuracy",
                                       global_accuracy);
  }

  return global_accuracy;
}

const galois::PointerWithSize<galois::GNNFloat>
galois::GraphNeuralNetwork::DoInference() {
  galois::StatTimer timer("DoInference", "GraphNeuralNetwork");
  if (timers_on_) {
    timer.start();
  }

  // start with graph features and pass it through all layers of the network
  galois::PointerWithSize<galois::GNNFloat> layer_input =
      graph_->GetLocalFeatures();

  for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
    layer_input = ptr->ForwardPhase(layer_input);
  }

  if (timers_on_) {
    timer.stop();
  }

  return layer_input;
}

float galois::GraphNeuralNetwork::GetGlobalAccuracy(
    PointerWithSize<GNNFloat> predictions) {
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    if (cpu_pred_.size() != predictions.size()) {
      cpu_pred_.resize(predictions.size());
    }

    // TODO get rid of CPU copy here if possible
    AdamOptimizer* adam = static_cast<AdamOptimizer*>(optimizer_.get());
    adam->CopyToVector(cpu_pred_, predictions);
    return graph_->GetGlobalAccuracy(cpu_pred_, phase_, config_.do_sampling());
  } else {
#endif
    return graph_->GetGlobalAccuracy(predictions, phase_,
                                     config_.do_sampling());
#ifdef GALOIS_ENABLE_GPU
  }
#endif
}

void galois::GraphNeuralNetwork::GradientPropagation() {
  galois::StatTimer timer("GradientPropagation", "GraphNeuralNetwork");
  if (timers_on_) {
    timer.start();
  }

  // from output layer get initial gradients
  std::vector<galois::GNNFloat> dummy;
  std::unique_ptr<galois::GNNLayer>& output_layer = gnn_layers_.back();
  galois::PointerWithSize<galois::GNNFloat> current_gradients =
      output_layer->BackwardPhase(dummy, nullptr);
  // loops through intermediate layers in a backward fashion
  // -1 to ignore output layer which was handled above
  for (size_t i = 0; i < gnn_layers_.size() - 1; i++) {
    // note this assumes you have at least 2 layers (including output)
    size_t layer_index = gnn_layers_.size() - 2 - i;

    // get the input to the layer before this one
    galois::PointerWithSize<galois::GNNFloat> prev_layer_input;
    if (layer_index != 0) {
      prev_layer_input = gnn_layers_[layer_index - 1]->GetForwardOutput();
    } else {
      prev_layer_input = graph_->GetLocalFeatures();
    }

    // backward prop and get a new set of gradients
    current_gradients = gnn_layers_[layer_index]->BackwardPhase(
        prev_layer_input, &current_gradients);
    // if not output do optimization/gradient descent
    // at this point in the layer the gradients exist; use the gradients to
    // update the weights of the layer
    gnn_layers_[layer_index]->OptimizeLayer(optimizer_.get(), layer_index);
  }

  if (timers_on_) {
    timer.stop();
  }
}

void galois::GraphNeuralNetwork::CorrectBackwardLinks() {
  // layer chain pointer
  PointerWithSize<GNNFloat> prev_output_layer(nullptr, 0);
  for (size_t layer_num = 0; layer_num < gnn_layers_.size(); layer_num++) {
    // first layer is nullptr so can be ignored
    if (layer_num != 0) {
      gnn_layers_[layer_num]->UpdateBackwardOutput(&prev_output_layer);
    }
    prev_output_layer = gnn_layers_[layer_num]->GetForwardOutput();
  }
}

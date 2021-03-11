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
  // max number of rows that can be passed as inputs; allocate space for it as
  // this will be the # of rows for each layer
  size_t max_rows = graph_->size();

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    graph_->ResizeLayerVector(config_.num_intermediate_layers());
  }
#endif
  // used for chaining layers together; begins as nullptr
  PointerWithSize<GNNFloat> prev_output_layer(nullptr, 0);

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

    GNNLayerDimensions layer_dims = {.input_rows    = max_rows,
                                     .input_columns = prev_layer_columns,
                                     .output_columns =
                                         config_.intermediate_layer_size(i)};

    switch (layer_type) {
    case GNNLayerType::kGraphConvolutional:
      gnn_layers_.push_back(std::move(std::make_unique<GraphConvolutionalLayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
#ifdef GALOIS_ENABLE_GPU
      if (device_personality == DevicePersonality::GPU_CUDA) {
        graph_->InitLayerVectorMetaObjects(
            i, galois::runtime::getSystemNetworkInterface().Num,
            layer_dims.input_columns, layer_dims.output_columns);
      }
#endif
      break;
    case GNNLayerType::kSAGE:
      gnn_layers_.push_back(std::move(std::make_unique<SAGELayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
#ifdef GALOIS_ENABLE_GPU
      // TODO(loc/hochan) sage layer gpu
#endif
      break;
    case GNNLayerType::kL2Norm:
      gnn_layers_.push_back(std::move(std::make_unique<L2NormLayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
#ifdef GALOIS_ENABLE_GPU
      // TODO(loc/hochan) l2 layer gpu
#endif
      break;
    case GNNLayerType::kDense:
      gnn_layers_.push_back(std::move(std::make_unique<DenseLayer>(
          i, *graph_, &prev_output_layer, layer_dims,
          config_.default_layer_config())));
#ifdef GALOIS_ENABLE_GPU
      // TODO(loc/hochan) dense layer gpu
#endif
      break;
    default:
      GALOIS_LOG_FATAL("Invalid layer type during network construction");
    }
    // update output layer for next layer
    prev_output_layer = gnn_layers_.back()->GetForwardOutput();
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

  // create the output layer
  GNNLayerDimensions output_dims = {
      .input_rows = max_rows,
      // get last intermediate layer column size
      .input_columns = config_.intermediate_layer_size(
          config_.num_intermediate_layers() - 1),
      .output_columns = config_.output_layer_size()};

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

  // flip sampling
  if (config_.do_sampling()) {
    for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
      ptr->EnableSampling();
    }
  }
}

float galois::GraphNeuralNetwork::Train(size_t num_epochs) {
  const size_t this_host = graph_->host_id();
  float train_accuracy{0.f};

  /*
  if (config_.do_sampling()) {
    for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
      assert(ptr->IsSampledLayer());
    }
  }
  */

  bool altered_norm_factor =
      config_.inductive_training_ || config_.do_sampling();

  if (config_.inductive_training_) {
    graph_->CalculateSpecialNormFactor(false, true);
  }
  galois::StatTimer epoch_timer("TrainingTime", "GraphNeuralNetwork");
  galois::StatTimer validation_timer("ValidationTime", "GraphNeuralNetwork");
  galois::StatTimer epoch_test_timer("TestTime", "GraphNeuralNetwork");

  // TODO incorporate validation/test intervals
  for (size_t epoch = 0; epoch < num_epochs; epoch++) {
    epoch_timer.start();
    if (config_.do_sampling()) {
      // subgraph sample every epoch
      graph_->GraphSAINTSample();
      graph_->CalculateSpecialNormFactor(true, config_.inductive_training_);
    }
    const PointerWithSize<galois::GNNFloat> predictions = DoInference();
    GradientPropagation();
    epoch_timer.stop();

    train_accuracy = GetGlobalAccuracy(predictions);

    if (this_host == 0) {
      const std::string t_name_acc =
          "TrainEpoch" + std::to_string(epoch) + "Accuracy";
      galois::gPrint("Epoch ", epoch, ": Train accuracy/F1 micro is ",
                     train_accuracy, "\n");
      galois::runtime::reportStat_Single("GraphNeuralNetwork", t_name_acc,
                                         train_accuracy);
    }

    bool do_validate = config_.validation_interval_
                           ? epoch % config_.validation_interval_ == 0
                           : false;
    bool do_test =
        config_.test_interval_ ? epoch % config_.test_interval_ == 0 : false;

    // get real norm factor back if altered by sampling or inductive training
    if ((do_validate || do_test) && altered_norm_factor) {
      graph_->CalculateFullNormFactor();
    }

    if (do_validate) {
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
        galois::runtime::reportStat_Single("GraphNeuralNetwork", v_name_acc,
                                           val_acc);
      }
    }

    if (do_test) {
      epoch_test_timer.start();
      SetLayerPhases(galois::GNNPhase::kTest);
      const PointerWithSize<galois::GNNFloat> test_pred = DoInference();
      epoch_test_timer.stop();

      float test_acc = GetGlobalAccuracy(test_pred);
      if (this_host == 0) {
        galois::gPrint("Epoch ", epoch, ": Test accuracy is ", test_acc, "\n");
        const std::string test_name_acc =
            "TestEpoch" + std::to_string(epoch) + "Accuracy";
        galois::runtime::reportStat_Single("GraphNeuralNetwork", test_name_acc,
                                           test_acc);
      }
    }

    if (do_validate || do_test) {
      // report the training time elapsed at this point in time
      galois::runtime::reportStat_Single(
          "GraphNeuralNetwork", "ElapsedTrainTimeEpoch" + std::to_string(epoch),
          epoch_timer.get());
      // revert to training phase for next epoch
      SetLayerPhases(galois::GNNPhase::kTrain);
      // get back inductive norm factor as necessary; sampling norm is handled
      // at beginning of every iteration
      if (config_.inductive_training_ && !config_.do_sampling()) {
        graph_->CalculateSpecialNormFactor(false, true);
      }
    }
  }

  uint64_t average_epoch_time = epoch_timer.get() / num_epochs;
  galois::runtime::reportStat_Tavg("GraphNeuralNetwork", "AverageEpochTime",
                                   average_epoch_time);

  if (altered_norm_factor) {
    graph_->CalculateFullNormFactor();
  }

  // check test accuracy
  galois::StatTimer test_timer("FinalTestRun", "GraphNeuralNetwork");
  test_timer.start();
  SetLayerPhases(galois::GNNPhase::kTest);
  const PointerWithSize<galois::GNNFloat> predictions = DoInference();
  float global_accuracy = GetGlobalAccuracy(predictions);
  test_timer.stop();

  if (this_host == 0) {
    galois::gPrint("Final test accuracy is ", global_accuracy, "\n");
    galois::runtime::reportStat_Single("GraphNeuralNetwork",
                                       "FinalTestAccuracy", global_accuracy);
  }

  return global_accuracy;
}

const galois::PointerWithSize<galois::GNNFloat>
galois::GraphNeuralNetwork::DoInference() {
  // start with graph features and pass it through all layers of the network
  galois::PointerWithSize<galois::GNNFloat> layer_input =
      graph_->GetLocalFeatures();
  for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
    layer_input = ptr->ForwardPhase(layer_input);
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
}

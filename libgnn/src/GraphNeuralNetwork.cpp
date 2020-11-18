#include "galois/GNNMath.h"
#include "galois/GraphNeuralNetwork.h"
#include "galois/layers/GraphConvolutionalLayer.h"
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
          i, *graph_, layer_dims, config_.default_layer_config())));
      break;
    default:
      GALOIS_LOG_FATAL("Invalid layer type during network construction");
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
        config_.num_intermediate_layers(), *graph_, output_dims)));
    break;
  default:
    GALOIS_LOG_FATAL("Invalid layer type during network construction");
  }
}

float galois::GraphNeuralNetwork::Train(size_t num_epochs) {
  const size_t this_host = graph_->host_id();
  // TODO incorporate validation/test intervals
  for (size_t epoch = 0; epoch < num_epochs; epoch++) {
    const PointerWithSize<galois::GNNFloat> predictions = DoInference();
    GradientPropagation();
    float train_accuracy = GetGlobalAccuracy(predictions);
    if (this_host == 0) {
      galois::gPrint("Epoch ", epoch, ": Train accuracy is ", train_accuracy,
                     "\n");
    }
    // TODO validation and test as necessary
  }

  // check test accuracy
  galois::StatTimer acc_timer("FinalAccuracyTest");
  acc_timer.start();
  SetLayerPhases(galois::GNNPhase::kTest);
  const PointerWithSize<galois::GNNFloat> predictions = DoInference();
  float global_accuracy = GetGlobalAccuracy(predictions);
  acc_timer.stop();

  if (this_host == 0) {
    galois::gPrint("Final test accuracy is ", global_accuracy, "\n");
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
    const PointerWithSize<GNNFloat> predictions) {
  // TODO mark as a forwarding argument?
#ifndef GALOIS_ENABLE_GPU
  return GetGlobalAccuracyCPU(predictions);
#else
  return gpu_object_.GetGlobalAccuracyGPU(graph_->GetGPUGraph(), phase_,
                                          predictions);
#endif
}

float galois::GraphNeuralNetwork::GetGlobalAccuracyCPU(
    const PointerWithSize<GNNFloat> predictions) {
  // check owned nodes' accuracy
  size_t num_labels = graph_->GetNumLabelClasses();
  assert((graph_->GetNumLabelClasses() * graph_->size()) == predictions.size());
  num_correct_.reset();
  total_checked_.reset();

  galois::do_all(
      galois::iterate(graph_->begin_owned(), graph_->end_owned()),
      [&](const unsigned lid) {
        if (graph_->IsValidForPhase(lid, phase_)) {
          total_checked_ += 1;
          // get prediction by getting max
          size_t predicted_label =
              galois::MaxIndex(num_labels, &(predictions[lid * num_labels]));
          // GALOIS_LOG_VERBOSE("Checking LID {} with label {} against
          // prediction {}",
          //                   lid, graph_->GetSingleClassLabel(lid),
          //                   predicted_label);
          // check against ground truth and track accordingly
          // TODO static cast used here is dangerous
          if (predicted_label ==
              static_cast<size_t>(graph_->GetSingleClassLabel(lid))) {
            num_correct_ += 1;
          }
        }
      },
      // TODO chunk size?
      // steal on as some threads may have nothing to work on
      galois::steal(), galois::loopname("GlobalAccuracy"));
  // TODO revise for later when multi-class labels come in

  size_t global_correct = num_correct_.reduce();
  size_t global_checked = total_checked_.reduce();

  GALOIS_LOG_VERBOSE("Accuracy: {} / {}", global_correct, global_checked);

  return static_cast<float>(global_correct) /
         static_cast<float>(global_checked);
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

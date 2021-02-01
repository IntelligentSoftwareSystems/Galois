#include "galois/GNNMath.h"
#include "galois/GraphNeuralNetwork.h"
#include "galois/layers/GraphConvolutionalLayer.h"
#include "galois/layers/SoftmaxLayer.h"
#include "galois/layers/SigmoidLayer.h"

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
      if (i == config_.num_intermediate_layers() - 1) {
        // last layer before output layer should never have activation
        gnn_layers_.back()->DisableActivation();
      }
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
  case (GNNOutputLayerType::kSigmoid):
    gnn_layers_.push_back(std::move(std::make_unique<SigmoidLayer>(
        config_.num_intermediate_layers(), *graph_, output_dims)));
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
  if (config_.do_sampling()) {
    for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
      assert(ptr->IsSampledLayer());
    }
  }

  // TODO incorporate validation/test intervals
  for (size_t epoch = 0; epoch < num_epochs; epoch++) {
    if (config_.do_sampling()) {
      // subgraph sample every epoch
      graph_->UniformNodeSample();
    }
    const PointerWithSize<galois::GNNFloat> predictions = DoInference();
    GradientPropagation();
    float train_accuracy = GetGlobalAccuracy(predictions);
    if (this_host == 0) {
      galois::gPrint("Epoch ", epoch, ": Train accuracy/F1 micro is ",
                     train_accuracy, "\n");
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
    PointerWithSize<GNNFloat> predictions) {
  return graph_->GetGlobalAccuracy(predictions, phase_);
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

#pragma once
//! @file GraphNeuralNetwork.h
//!
//! Defines the graph neural network class that is used to classify graphs as
//! well as helper enums/classes involved with the GNN.

#include "galois/Logging.h"
#include "galois/GNNOptimizers.h"
#include "galois/graphs/GNNGraph.h"
#include "galois/layers/GNNLayer.h"

namespace galois {

////////////////////////////////////////////////////////////////////////////////

// TODO validation and testing intervals
//! Configuration object passed into constructor of a GraphNeuralNetwork to
//! determine how the network gets constructed.
class GraphNeuralNetworkConfig {
public:
  // default move, no copy
  GraphNeuralNetworkConfig()                                = delete;
  GraphNeuralNetworkConfig(const GraphNeuralNetworkConfig&) = delete;
  GraphNeuralNetworkConfig& operator=(const GraphNeuralNetworkConfig&) = delete;
  GraphNeuralNetworkConfig(GraphNeuralNetworkConfig&&) = default;
  GraphNeuralNetworkConfig& operator=(GraphNeuralNetworkConfig&&) = default;

  //! Construction without a config for layers specified; uses a default
  GraphNeuralNetworkConfig(size_t num_layers,
                           const std::vector<GNNLayerType>& layer_types,
                           const std::vector<size_t>& layer_column_sizes,
                           GNNOutputLayerType output_layer_type)
      : GraphNeuralNetworkConfig(num_layers, layer_types, layer_column_sizes,
                                 output_layer_type, GNNLayerConfig()) {}

  //! Construction with a specified config for layers
  GraphNeuralNetworkConfig(size_t num_layers,
                           const std::vector<GNNLayerType>& layer_types,
                           const std::vector<size_t>& layer_column_sizes,
                           GNNOutputLayerType output_layer_type,
                           const GNNLayerConfig& default_layer_config)
      : num_intermediate_layers_(num_layers), layer_types_(layer_types),
        layer_column_sizes_(layer_column_sizes),
        output_layer_type_(output_layer_type),
        default_layer_config_(default_layer_config) {
    // Do sanity checks on inputs
    // should have a type for each layer
    GALOIS_LOG_ASSERT(num_intermediate_layers_ == layer_types_.size());
    // For now, should be at least 1 intermediate layer
    GALOIS_LOG_ASSERT(num_intermediate_layers_ >= 1);
    // + 1 because it includes output layer
    GALOIS_LOG_ASSERT((num_intermediate_layers_ + 1) ==
                      layer_column_sizes_.size());
  }

  //! # layers NOT including output layer
  size_t num_intermediate_layers() { return num_intermediate_layers_; }
  //! Get intermediate layer i
  GNNLayerType intermediate_layer_type(size_t i) {
    assert(i < num_intermediate_layers_);
    return layer_types_[i];
  }
  //! Get intermediate layer i's size
  size_t intermediate_layer_size(size_t i) {
    assert(i < num_intermediate_layers_);
    return layer_column_sizes_[i];
  }
  //! Type of output layer
  GNNOutputLayerType output_layer_type() { return output_layer_type_; }
  //! Size of output layer is last element of layer column sizes
  size_t output_layer_size() {
    return layer_column_sizes_[num_intermediate_layers_];
  }
  //! Get the default layer config of layers in this GNN
  const GNNLayerConfig& default_layer_config() { return default_layer_config_; }

private:
  //! Number of layers to construct in the GNN not including the output
  //! layer
  size_t num_intermediate_layers_;
  //! Layers to construct for the GNN going from left to right; size should
  //! match num_layers setting
  std::vector<GNNLayerType> layer_types_;
  //! Size (in columns) of each non-output layer; size should match num_layers
  //! + 1 (+1 is for the output layer)
  std::vector<size_t> layer_column_sizes_;
  //! Output layer type
  GNNOutputLayerType output_layer_type_;
  //! Default config to use for layers
  GNNLayerConfig default_layer_config_;
};

////////////////////////////////////////////////////////////////////////////////

//! Class representing the graph neural network: contains the graph to train as
//! well as all the layers that comprise it
class GraphNeuralNetwork {
public:
  //! Construct the graph neural network given the graph to train on as well as
  //! a configuration object
  GraphNeuralNetwork(std::unique_ptr<graphs::GNNGraph> graph,
                     std::unique_ptr<BaseOptimizer> optimizer,
                     GraphNeuralNetworkConfig&& config);

  //! Number of intermediate layers (DOES NOT INCLUDE OUTPUT LAYER)
  size_t num_intermediate_layers() { return gnn_layers_.size() - 1; }

  //! Returns pointer to intermediate layer i
  galois::GNNLayer* GetIntermediateLayer(size_t i) {
    if (i < gnn_layers_.size() - 1) {
      return gnn_layers_[i].get();
    } else {
      GALOIS_LOG_FATAL("Accessing out of bounds intermediate layer {}", i);
    }
  }

  //! Set the phases of all layers at once as well as this network
  void SetLayerPhases(galois::GNNPhase phase) {
    phase_ = phase;
    for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
      ptr->SetLayerPhase(phase);
    }
  }

  //! Set weights on all layers to 1; should be used for debugging only
  void SetAllLayerWeightsTo1() {
    for (std::unique_ptr<galois::GNNLayer>& ptr : gnn_layers_) {
      ptr->InitAllWeightsTo1();
    }
  }

  //! Returns the output layer
  galois::GNNLayer* GetOutputLayer() { return gnn_layers_.back().get(); }

  //! Do training for a specified # of epochs and return test accuracy at the
  //! end of it
  float Train(size_t num_epochs);

  //! Propogates the graph's feature vectors through the network to get a new
  //! vector representation.
  //! Also known as the forward phase in most literature
  //! @returns Output layer's output
  const PointerWithSize<GNNFloat> DoInference();

  float GetGlobalAccuracy(const PointerWithSize<GNNFloat> predictions);

  //! Backpropagate gradients from the output layer backwards through the
  //! network to update the layer weights. Also known as a backward phase in
  //! most literature
  void GradientPropagation();

private:
  //! Underlying graph to train
  std::unique_ptr<graphs::GNNGraph> graph_;
  //! Optimizer object for weight updates
  std::unique_ptr<BaseOptimizer> optimizer_;
  //! Configuration object used to construct this GNN
  GraphNeuralNetworkConfig config_;
  //! GNN layers including the output
  std::vector<std::unique_ptr<galois::GNNLayer>> gnn_layers_;
  //! Current phase of the GNN: train, validation, test
  GNNPhase phase_{GNNPhase::kTrain};
  //! Used to track accurate predictions during accuracy calculation
  DGAccumulator<size_t> num_correct_;
  //! Used to count total number of things checked during accuracy calculation
  DGAccumulator<size_t> total_checked_;
};

} // namespace galois

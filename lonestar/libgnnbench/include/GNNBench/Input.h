#pragma once

#include "galois/GraphNeuralNetwork.h"
#include "galois/graphs/GNNGraph.h"
#include <llvm/Support/CommandLine.h>

#ifdef GALOIS_ENABLE_GPU
extern int gpudevice;
#endif

//! Directory where all files used for GNN training are found
extern llvm::cl::opt<std::string> input_directory;
//! Base graph name (used to find the csgr, features, masks, etc.)
extern llvm::cl::opt<std::string> input_name;
//! Scheme used to partition the graph
extern llvm::cl::opt<galois::graphs::GNNPartitionScheme> partition_scheme;
extern llvm::cl::opt<unsigned> num_layers;
extern llvm::cl::opt<unsigned> layer_size;
extern llvm::cl::opt<float> learning_rate;
extern llvm::cl::opt<galois::GNNOutputLayerType> output_layer_type;
extern llvm::cl::opt<bool> multiclass_labels;
extern llvm::cl::opt<bool> do_graph_sampling;
extern llvm::cl::opt<bool> useWMD;
extern llvm::cl::opt<bool> use_train_subgraph;
extern llvm::cl::opt<unsigned> minibatch_test_interval;
extern llvm::cl::opt<unsigned> test_interval;
extern llvm::cl::opt<unsigned> val_interval;
extern llvm::cl::opt<unsigned> train_minibatch_size;
extern llvm::cl::opt<unsigned> test_minibatch_size;
extern llvm::cl::opt<bool> inductive_subgraph;

const char* GNNPartitionToString(galois::graphs::GNNPartitionScheme s);

std::vector<galois::GNNLayerType> CreateLayerTypesVector();

template <typename VTy, typename ETy>
std::vector<size_t>
CreateLayerSizesVector(const galois::graphs::GNNGraph<VTy, ETy>* gnn_graph) {
  // set layer sizes for intermdiate and output layers
  std::vector<size_t> layer_sizes_vector;

  // if (layer_sizes.size()) {
  //  GALOIS_LOG_ASSERT(layer_sizes.size() == num_layers);
  //  for (size_t i = 0; i < num_layers; i++) {
  //    layer_sizes_vector.emplace_back(layer_sizes[i]);
  //  }
  //  // verify user satisfies last intermediate layer needing to have same size
  //  // as # label classes
  //  if (layer_sizes_vector.back() != gnn_graph->GetNumLabelClasses()) {
  //    galois::gWarn(
  //        "Size of last layer (", layer_sizes_vector.back(),
  //        ") is not equal to # label classes: forcefully changing it to ",
  //        gnn_graph->GetNumLabelClasses());
  //    layer_sizes_vector.back()   = gnn_graph->GetNumLabelClasses();
  //    layer_sizes[num_layers - 1] = gnn_graph->GetNumLabelClasses();
  //  }

  //  GALOIS_LOG_ASSERT(layer_sizes_vector.back() ==
  //                    gnn_graph->GetNumLabelClasses());
  //} else {
  //  // default 16 for everything until last 2
  //  for (size_t i = 0; i < num_layers - 1; i++) {
  //    layer_sizes_vector.emplace_back(16);
  //  }
  //  // last 2 sizes must be equivalent to # label classes; this is the last
  //  // intermediate layer
  //  layer_sizes_vector.emplace_back(gnn_graph->GetNumLabelClasses());
  //}

  for (size_t i = 0; i < num_layers - 1; i++) {
    layer_sizes_vector.emplace_back(layer_size);
  }
  // last 2 sizes must be equivalent to # label classes; this is the last
  // intermediate layer
  layer_sizes_vector.emplace_back(gnn_graph->GetNumLabelClasses());
  // TODO
  // for now only softmax layer which dictates the output size of the last
  // intermediate layer + size of the output layer
  // output layer at the moment required to be same as # label classes
  layer_sizes_vector.emplace_back(gnn_graph->GetNumLabelClasses());

  return layer_sizes_vector;
}

galois::GNNLayerConfig CreateLayerConfig();

template <typename VTy, typename ETy>
std::unique_ptr<galois::BaseOptimizer>
CreateOptimizer(const galois::graphs::GNNGraph<VTy, ETy>* gnn_graph) {
  std::vector<size_t> opt_sizes;

  // optimizer sizes are based on intermediate layer sizes, input feats, and
  // # label classes
  // if (layer_sizes.size()) {
  //  GALOIS_LOG_ASSERT(layer_sizes.size() == num_layers);
  //  opt_sizes.emplace_back(gnn_graph->node_feature_length() * layer_sizes[0]);
  //  // assumption here is that if it reached this point then layer sizes were
  //  // already sanity checked previously (esp. last layer)
  //  for (size_t i = 1; i < num_layers; i++) {
  //    opt_sizes.emplace_back(layer_sizes[i] * layer_sizes[i - 1]);
  //  }
  //} else {
  //  // everything is size 16 until last
  //  if (num_layers == 1) {
  //    // single layer requires a bit of special handling
  //    opt_sizes.emplace_back(gnn_graph->node_feature_length() *
  //                           gnn_graph->GetNumLabelClasses());
  //  } else {
  //    // first
  //    opt_sizes.emplace_back(gnn_graph->node_feature_length() * 16);
  //    for (size_t i = 1; i < num_layers - 1; i++) {
  //      opt_sizes.emplace_back(16 * 16);
  //    }
  //    // last
  //    opt_sizes.emplace_back(16 * gnn_graph->GetNumLabelClasses());
  //  }
  //}

  // everything is size 16 until last
  if (num_layers == 1) {
    // single layer requires a bit of special handling
    opt_sizes.emplace_back(gnn_graph->node_feature_length() *
                           gnn_graph->GetNumLabelClasses());
  } else {
    // first
    opt_sizes.emplace_back(gnn_graph->node_feature_length() * layer_size);
    for (size_t i = 1; i < num_layers - 1; i++) {
      opt_sizes.emplace_back(layer_size * layer_size);
    }
    // last
    opt_sizes.emplace_back(layer_size * gnn_graph->GetNumLabelClasses());
  }
  GALOIS_LOG_ASSERT(opt_sizes.size() == num_layers);

  galois::AdamOptimizer::AdamConfiguration adam_config;
  adam_config.alpha = learning_rate;

  // TODO only adam works right now, add the others later
  return std::make_unique<galois::AdamOptimizer>(adam_config, opt_sizes,
                                                 num_layers);
}

std::vector<unsigned> CreateFanOutVector();

//! Using command line args above, create a GNN using some specified layer type
//! as the intermediate layer.
template <typename VTy, typename ETy>
std::unique_ptr<galois::GraphNeuralNetwork<VTy, ETy>>
InitializeGraphNeuralNetwork() {
  // partition/load graph
  auto gnn_graph = std::make_unique<galois::graphs::GNNGraph<VTy, ETy>>(
      input_directory, input_name, partition_scheme, !multiclass_labels,
      useWMD);

  // create layer types vector
  std::vector<galois::GNNLayerType> layer_types = CreateLayerTypesVector();
  // sizes
  std::vector<size_t> layer_sizes_vector =
      CreateLayerSizesVector(gnn_graph.get());
  // layer config object
  galois::GNNLayerConfig layer_config = CreateLayerConfig();
  // GNN config object
  galois::GraphNeuralNetworkConfig gnn_config(
      num_layers, layer_types, layer_sizes_vector, output_layer_type,
      do_graph_sampling, layer_config);
  gnn_config.use_train_subgraph_      = use_train_subgraph;
  gnn_config.validation_interval_     = val_interval;
  gnn_config.test_interval_           = test_interval;
  gnn_config.train_minibatch_size_    = train_minibatch_size;
  gnn_config.test_minibatch_size_     = test_minibatch_size;
  gnn_config.minibatch_test_interval_ = minibatch_test_interval;
  gnn_config.inductive_subgraph_      = inductive_subgraph;
  gnn_config.fan_out_vector_          = CreateFanOutVector();

  // optimizer
  std::unique_ptr<galois::BaseOptimizer> opt = CreateOptimizer(gnn_graph.get());

  // create the gnn
  return std::make_unique<galois::GraphNeuralNetwork<VTy, ETy>>(
      std::move(gnn_graph), std::move(opt), std::move(gnn_config));
}

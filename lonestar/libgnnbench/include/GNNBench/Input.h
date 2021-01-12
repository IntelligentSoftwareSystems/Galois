#pragma once

#include "galois/GraphNeuralNetwork.h"
#include "galois/graphs/GNNGraph.h"
#include <llvm/Support/CommandLine.h>

//! Directory where all files used for GNN training are found
extern llvm::cl::opt<std::string> input_directory;
//! Base graph name (used to find the csgr, features, masks, etc.)
extern llvm::cl::opt<std::string> input_name;
//! Scheme used to partition the graph
extern llvm::cl::opt<galois::graphs::GNNPartitionScheme> partition_scheme;
// Control layer count and size
extern llvm::cl::opt<size_t> num_layers;
extern llvm::cl::list<size_t> layer_sizes;
// Control dropout
extern llvm::cl::opt<bool> do_dropout;
extern llvm::cl::opt<float> dropout_rate;
// Control activation
extern llvm::cl::opt<bool> do_activation;
// TODO activation layer type once more are supported
//! Controls weight normalization based on degree
extern llvm::cl::opt<bool> do_normalization;
//! Output layer type
extern llvm::cl::opt<galois::GNNOutputLayerType> output_layer_type;
//! If true, use multiclass ground truth
extern llvm::cl::opt<bool> multiclass_labels;
// TODO optimizer type
//! Toggles an optimization that flips aggregate/update step if it would be
//! beneficial
extern llvm::cl::opt<bool> agg_after_update;

const char* GNNPartitionToString(galois::graphs::GNNPartitionScheme s);

//! Using command line args above, create a GNN using some specified layer type
//! as the intermediate layer.
std::unique_ptr<galois::GraphNeuralNetwork>
InitializeGraphNeuralNetwork(galois::GNNLayerType layer_type);

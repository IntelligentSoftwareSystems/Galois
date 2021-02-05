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

const char* GNNPartitionToString(galois::graphs::GNNPartitionScheme s);

//! Using command line args above, create a GNN using some specified layer type
//! as the intermediate layer.
std::unique_ptr<galois::GraphNeuralNetwork>
InitializeGraphNeuralNetwork(galois::GNNLayerType layer_type);

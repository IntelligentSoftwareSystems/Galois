#include "GNNBench/Input.h"

namespace cll = llvm::cl;

// Self documented via the desc argument

llvm::cl::opt<std::string> input_directory(
    "inputDirectory",
    cll::desc("Base directory to find all files required for doing GNN "
              "training (features, graph topology, masks, etc.)"),
    cll::init(galois::default_gnn_dataset_path));

llvm::cl::opt<std::string> input_name(
    cll::Positional,
    cll::desc("Base name of graph: used to find csgr, features, etc."),
    cll::Required);

llvm::cl::opt<galois::graphs::GNNPartitionScheme> partition_scheme(
    "partition", cll::desc("Type of partitioning."),
    cll::values(clEnumValN(galois::graphs::GNNPartitionScheme::kOEC, "oec",
                           "Outgoing Edge-Cut (default)"),
                clEnumValN(galois::graphs::GNNPartitionScheme::kCVC, "cvc",
                           "Cartesian Vertex-Cut")),
    cll::init(galois::graphs::GNNPartitionScheme::kOEC));

llvm::cl::opt<size_t> num_layers(
    "numLayers",
    cll::desc(
        "Number of intermediate layers in the neural network (default 2))"),
    cll::init(2));

llvm::cl::list<size_t> layer_sizes(
    "layerSizes",
    cll::desc(
        "Comma separated list of numbers specifying intermediate layer sizes"),
    cll::CommaSeparated);

llvm::cl::opt<bool> do_dropout(
    "doDropout",
    cll::desc("If true (on by default), does dropout of input during training"),
    cll::init(true));

llvm::cl::opt<float> dropout_rate(
    "dropoutRate",
    cll::desc("Specifies probability that any one weight is DROPPED (e.g., if "
              "0.1, then 10 percent chance of dropping) (default 0.5)"),
    cll::init(0.5));

llvm::cl::opt<bool>
    do_activation("doActivation",
                  cll::desc("If true (off by default), does activation at the "
                            "end of an intermediate layer"),
                  cll::init(false));

llvm::cl::opt<bool>
    do_normalization("doNormalization",
                     cll::desc("If true (on by default), normalizes vertex "
                               "features based on their degree"),
                     cll::init(true));

const char* GNNPartitionToString(galois::graphs::GNNPartitionScheme s) {
  switch (s) {
  case galois::graphs::GNNPartitionScheme::kOEC:
    return "oec";
  case galois::graphs::GNNPartitionScheme::kCVC:
    return "cvc";
  default:
    GALOIS_LOG_FATAL("Invalid partitioning scheme");
    return "";
  }
}

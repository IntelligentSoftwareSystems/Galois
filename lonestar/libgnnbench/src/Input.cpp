#include "galois/Logging.h"
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

llvm::cl::list<size_t>
    layer_sizes("layerSizes",
                cll::desc("Comma separated list of numbers specifying "
                          "intermediate layer sizes (does not include output)"),
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

llvm::cl::opt<galois::GNNOutputLayerType> output_layer_type(
    "outputLayer", cll::desc("Type of output layer"),
    cll::values(clEnumValN(galois::GNNOutputLayerType::kSoftmax, "softmax",
                           "Softmax (default)"),
                clEnumValN(galois::GNNOutputLayerType::kSigmoid, "sigmoid",
                           "Sigmoid")),
    cll::init(galois::GNNOutputLayerType::kSoftmax));

llvm::cl::opt<bool>
    multiclass_labels("multiclassLabels",
                      cll::desc("If true (off by default), use multi-class "
                                "ground truth; required for some inputs"),
                      cll::init(false));

llvm::cl::opt<bool> disable_agg_after_update(
    "disableAggregationAfterUpdate",
    cll::desc("If true (off by default), disables aggregate "
              "after update optimization"),
    cll::init(false));

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

//! Initializes the vector of layer sizes from command line args + graph
std::vector<size_t>
CreateLayerSizesVector(const galois::graphs::GNNGraph* gnn_graph) {
  // set layer sizes for intermdiate and output layers
  std::vector<size_t> layer_sizes_vector;
  if (layer_sizes.size()) {
    GALOIS_LOG_ASSERT(layer_sizes.size() == num_layers);
    for (size_t i = 0; i < num_layers; i++) {
      layer_sizes_vector.emplace_back(layer_sizes[i]);
    }
    // verify user satisfies last intermediate layer needing to have same size
    // as # label classes
    GALOIS_LOG_ASSERT(layer_sizes_vector.back() ==
                      gnn_graph->GetNumLabelClasses());
  } else {
    // default 16 for everything until last 2
    for (size_t i = 0; i < num_layers - 1; i++) {
      layer_sizes_vector.emplace_back(16);
    }
    // last 2 sizes must be equivalent to # label classes; this is the last
    // intermediate layer
    layer_sizes_vector.emplace_back(gnn_graph->GetNumLabelClasses());
  }

  // TODO
  // for now only softmax layer which dictates the output size of the last
  // intermediate layer + size of the output layer
  // output layer at the moment required to be same as # label classes
  layer_sizes_vector.emplace_back(gnn_graph->GetNumLabelClasses());

  return layer_sizes_vector;
}

//! Setup layer config struct based on cli args
galois::GNNLayerConfig CreateLayerConfig() {
  galois::GNNLayerConfig layer_config;
  layer_config.do_dropout                     = do_dropout;
  layer_config.dropout_rate                   = dropout_rate;
  layer_config.do_activation                  = do_activation;
  layer_config.do_normalization               = do_normalization;
  layer_config.disable_aggregate_after_update = disable_agg_after_update;
  return layer_config;
}

std::unique_ptr<galois::BaseOptimizer>
CreateOptimizer(const galois::graphs::GNNGraph* gnn_graph) {
  std::vector<size_t> opt_sizes;

  // optimizer sizes are based on intermediate layer sizes, input feats, and
  // # label classes
  if (layer_sizes.size()) {
    GALOIS_LOG_ASSERT(layer_sizes.size() == num_layers);
    opt_sizes.emplace_back(gnn_graph->node_feature_length() * layer_sizes[0]);
    // assumption here is that if it reached this point then layer sizes were
    // already sanity checked previously (esp. last layer)
    for (size_t i = 1; i < num_layers; i++) {
      opt_sizes.emplace_back(layer_sizes[i] * layer_sizes[i - 1]);
    }
  } else {
    // everything is size 16 until last
    if (num_layers == 1) {
      // single layer requires a bit of special handling
      opt_sizes.emplace_back(gnn_graph->node_feature_length() *
                             gnn_graph->GetNumLabelClasses());
    } else {
      // first
      opt_sizes.emplace_back(gnn_graph->node_feature_length() * 16);
      for (size_t i = 1; i < num_layers - 1; i++) {
        opt_sizes.emplace_back(16 * 16);
      }
      // last
      opt_sizes.emplace_back(16 * gnn_graph->GetNumLabelClasses());
    }
  }
  GALOIS_LOG_ASSERT(opt_sizes.size() == num_layers);

  // TODO only adam works right now, add the others later
  return std::make_unique<galois::AdamOptimizer>(opt_sizes, num_layers);
}

std::unique_ptr<galois::GraphNeuralNetwork>
InitializeGraphNeuralNetwork(galois::GNNLayerType layer_type) {
  // partition/load graph
  auto gnn_graph = std::make_unique<galois::graphs::GNNGraph>(
      input_directory, input_name, partition_scheme, !multiclass_labels);

  // create layer types vector
  std::vector<galois::GNNLayerType> layer_types;
  for (size_t i = 0; i < num_layers; i++) {
    layer_types.push_back(layer_type);
  }
  // sizes
  std::vector<size_t> layer_sizes_vector =
      CreateLayerSizesVector(gnn_graph.get());
  // layer config object
  galois::GNNLayerConfig layer_config = CreateLayerConfig();
  // GNN config object
  galois::GraphNeuralNetworkConfig gnn_config(num_layers, layer_types,
                                              layer_sizes_vector,
                                              output_layer_type, layer_config);
  // optimizer
  std::unique_ptr<galois::BaseOptimizer> opt = CreateOptimizer(gnn_graph.get());

  // create the gnn
  return std::make_unique<galois::GraphNeuralNetwork>(
      std::move(gnn_graph), std::move(opt), std::move(gnn_config));
}

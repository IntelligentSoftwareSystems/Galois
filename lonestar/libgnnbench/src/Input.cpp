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

llvm::cl::list<size_t> layer_sizes(
    "layerSizes",
    cll::desc(
        "Comma separated list of numbers specifying "
        "intermediate layer sizes (does not include output); default sizes are "
        "16 until last layer which is the size of the # of labels"),
    cll::CommaSeparated);

llvm::cl::list<galois::GNNLayerType> cl_layer_types(
    "layerTypes",
    cll::desc("Comma separated list of layer types specifying "
              "intermediate layers (does not include output)"),
    cll::values(clEnumValN(galois::GNNLayerType::kGraphConvolutional, "gcn",
                           "Graph Convolutional Layer (default)"),
                clEnumValN(galois::GNNLayerType::kSAGE, "sage",
                           "SAGE layer (GCN with concat + mean)"),
                clEnumValN(galois::GNNLayerType::kDense, "dense",
                           "Dense Layer")),
    cll::CommaSeparated);

llvm::cl::opt<bool>
    disable_dropout("disableDropout",
                    cll::desc("If true (off by default), disables dropout of "
                              "layer weights during training"),
                    cll::init(false));

llvm::cl::opt<float> dropout_rate(
    "dropoutRate",
    cll::desc("Specifies probability that any one weight is DROPPED (e.g., if "
              "0.1, then 10 percent chance of dropping) (default 0.5)"),
    cll::init(0.5));

llvm::cl::opt<bool> disable_activation(
    "disableActivation",
    cll::desc("If true (off by default), disable activation at the "
              "end of an intermediate layers"),
    cll::init(false));

llvm::cl::opt<bool> disable_normalization(
    "disableNormalization",
    cll::desc("If true (off by default), disable normalizing vertex "
              "features based on their degree"),
    cll::init(false));

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

llvm::cl::opt<bool> disable_self_aggregate(
    "disableSelfAggregation",
    cll::desc("If true (off by default), disables aggregate of self feature"),
    cll::init(false));

llvm::cl::opt<bool>
    do_graph_sampling("doGraphSampling",
                      cll::desc("If true (off by default), sample nodes for "
                                "use every epoch at a 50\% drop rate"),
                      cll::init(false));

llvm::cl::opt<bool>
    do_inductive_training("doInductiveTraining",
                          cll::desc("If true (off by default), during training "
                                    "all non-train nodes are ignored"),
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
std::vector<galois::GNNLayerType> CreateLayerTypesVector() {
  std::vector<galois::GNNLayerType> layer_types;
  if (!cl_layer_types.size()) {
    // default is all GCN layers
    for (size_t i = 0; i < num_layers; i++) {
      layer_types.emplace_back(galois::GNNLayerType::kGraphConvolutional);
    }
  } else {
    GALOIS_LOG_VASSERT(cl_layer_types.size() == num_layers,
                       "Number layer types should be {} not {}", num_layers,
                       cl_layer_types.size());
    for (size_t i = 0; i < num_layers; i++) {
      layer_types.emplace_back(cl_layer_types[i]);
    }
  }
  return layer_types;
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
  layer_config.disable_dropout                = disable_dropout;
  layer_config.dropout_rate                   = dropout_rate;
  layer_config.disable_activation             = disable_activation;
  layer_config.disable_normalization          = disable_normalization;
  layer_config.disable_aggregate_after_update = disable_agg_after_update;
  layer_config.disable_self_aggregate         = disable_self_aggregate;
  layer_config.inductive_training_            = do_inductive_training;
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

std::unique_ptr<galois::GraphNeuralNetwork> InitializeGraphNeuralNetwork() {
  // partition/load graph
  auto gnn_graph = std::make_unique<galois::graphs::GNNGraph>(
      input_directory, input_name, partition_scheme, !multiclass_labels);

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
  gnn_config.inductive_training_ = do_inductive_training;
  // optimizer
  std::unique_ptr<galois::BaseOptimizer> opt = CreateOptimizer(gnn_graph.get());

  // create the gnn
  return std::make_unique<galois::GraphNeuralNetwork>(
      std::move(gnn_graph), std::move(opt), std::move(gnn_config));
}

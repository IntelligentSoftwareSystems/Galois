#pragma once

#include "galois/GNNTypes.h"
#include "galois/graphs/CuSPPartitioner.h"
#include "galois/graphs/GluonSubstrate.h"
#include "galois/graphs/GraphAggregationSyncStructures.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/graphs/GNNGraph.cuh"
#endif

namespace galois {

// TODO remove the need to hardcode this path
//! Path to location of all gnn files
static const std::string default_gnn_dataset_path =
    "/net/ohm/export/iss/inputs/Learning/";

//! Helper struct to maintain start/end/size of any particular range. Mostly
//! used for mask ranges.
struct GNNRange {
  size_t begin{0};
  size_t end{0};
  size_t size{0};
};

namespace graphs {

//! Possible partitioning schemes for the GNN graph
enum class GNNPartitionScheme { kOEC, kCVC };

//! XXX
class GNNGraph {
public:
  using GNNDistGraph = galois::graphs::DistGraph<char, void>;
  using WholeGraph   = galois::graphs::LC_CSR_Graph<char, void>;
  using GraphNode    = GNNDistGraph::GraphNode;
  // defined as such because dist graph range objects used long unsigned
  using NodeIterator = boost::counting_iterator<size_t>;
  using EdgeIterator = GNNDistGraph::edge_iterator;

  GNNGraph(const std::string& dataset_name, GNNPartitionScheme partition_scheme,
           bool has_single_class_label);
  //! Loads a graph and all relevant metadata (labels, features, masks, etc.)
  GNNGraph(const std::string& input_directory, const std::string& dataset_name,
           GNNPartitionScheme partition_scheme, bool has_single_class_label);

  //! Returns host id
  size_t host_id() const { return host_id_; }

  //! Returns host id in brackets to use for printing things
  const std::string& host_prefix() const { return host_prefix_; }

  //! Length of a node feature
  size_t node_feature_length() const { return node_feature_length_; }

  //! Return the number of label classes (i.e. number of possible outputs)
  size_t GetNumLabelClasses() const { return num_label_classes_; };

  //////////////////////////////////////////////////////////////////////////////
  // Graph accessors
  //////////////////////////////////////////////////////////////////////////////

  //! Return # of nodes in the partitioned graph
  size_t size() const { return partitioned_graph_->size(); }

  //! Node begin for all local nodes
  NodeIterator begin() const {
    return partitioned_graph_->allNodesRange().begin();
  }
  //! Node end for all local nodes
  NodeIterator end() const { return partitioned_graph_->allNodesRange().end(); }
  //! Return GID of some local node
  size_t GetGID(unsigned lid) const { return partitioned_graph_->getGID(lid); }

  NodeIterator begin_owned() const {
    return partitioned_graph_->masterNodesRange().begin();
  }

  NodeIterator end_owned() const {
    return partitioned_graph_->masterNodesRange().end();
  }

  // All following functions take a local node id
  EdgeIterator EdgeBegin(GraphNode n) const {
    return partitioned_graph_->edge_begin(n);
  };
  EdgeIterator EdgeEnd(GraphNode n) const {
    return partitioned_graph_->edge_end(n);
  };
  GraphNode EdgeDestination(EdgeIterator ei) const {
    return partitioned_graph_->getEdgeDst(ei);
  };
  GNNFloat NormFactor(GraphNode n) const { return norm_factors_[n]; }

  //! Returns the ground truth label of some local id assuming labels are single
  //! class labels.
  GNNFloat GetSingleClassLabel(const unsigned lid) const {
    assert(using_single_class_labels_);
    return local_ground_truth_labels_[lid];
  }

  //! Return matrix of the local node features
  const std::vector<GNNFloat>& GetLocalFeatures() const {
    return local_node_features_;
  }

  //! Given an LID and the current phase of GNN computation, determine if the
  //! lid in question is valid for the current phase (i.e., it is part of
  //! a training, validation, or test phase mask)
  bool IsValidForPhase(const unsigned lid,
                       const galois::GNNPhase current_phase) const;

  //////////////////////////////////////////////////////////////////////////////

  //! Given a matrix and the column size, do an aggregate sync where each row
  //! is considered a node's data and sync using the graph's Gluon
  //! substrate
  //! Note that it's const because the only thing being used is the graph
  //! topology of this object; the thing modified is the passed in matrix
  void AggregateSync(GNNFloat* matrix_to_sync,
                     const size_t matrix_column_size) const;

private:
  //! Directory for input data
  const std::string input_directory_;
  //! In a multi-host setting, this variable stores the host id that the graph
  //! is currently running on
  unsigned host_id_;
  //! String header that can be used for debug print statements to get the host
  //! this graph is on
  std::string host_prefix_;
  //! Number of classes for a single vertex label
  size_t num_label_classes_{1};
  //! Length of a feature node
  size_t node_feature_length_{0};
  //! Partitioned graph
  std::unique_ptr<GNNDistGraph> partitioned_graph_;
  //! The entire topology of the dataset: used for things like norm factor
  //! calculation or sampling
  WholeGraph whole_graph_;
  //! Sync substrate for the partitioned graph
  std::unique_ptr<galois::graphs::GluonSubstrate<GNNDistGraph>> sync_substrate_;
  //! True if labels are single class
  bool using_single_class_labels_;
  //! Ground truth label for nodes in the partitioned graph; Nx1 if single
  //! class, N x num classes if multi-class label
  std::vector<GNNLabel> local_ground_truth_labels_;
  //! Feature vectors for nodes in partitioned graph
  std::vector<GNNFeature> local_node_features_;

  // TODO maybe revisit this and use an actual bitset
  //! Bitset indicating which nodes are training nodes
  std::vector<GNNLabel> local_training_mask_;
  //! Bitset indicating which nodes are validation nodes
  std::vector<GNNLabel> local_validation_mask_;
  //! Bitset indicating which nodes are testing nodes
  std::vector<GNNLabel> local_testing_mask_;

  //! Global mask range for training nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_training_mask_range_;
  //! Global mask range for validation nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_validation_mask_range_;
  //! Global mask range for testing nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_testing_mask_range_;

  //! Normalization constant based on structure of the graph (degrees)
  std::vector<GNNFloat> norm_factors_;

  // TODO vars for subgraphs as necessary

  //////////////////////////////////////////////////////////////////////////////
  // Initialization
  //////////////////////////////////////////////////////////////////////////////

  //! Read labels of local nodes only
  void ReadLocalLabels(const std::string& dataset_name,
                       bool has_single_class_label);
  //! Read features of local nodes only
  void ReadLocalFeatures(const std::string& dataset_str);
  //! Helper function to read masks from file into the appropriate structures
  //! given a name, mask type, and arrays to save into
  size_t ReadLocalMasksFromFile(const std::string& dataset_name,
                                const std::string& mask_type,
                                GNNRange* mask_range, GNNLabel* masks);
  //! Read masks of local nodes only for training, validation, and testing
  void ReadLocalMasks(const std::string& dataset_name);
  //! Reads the entire graph topology in (but nothing else)
  void ReadWholeGraph(const std::string& dataset_name);
  //! Initializes the norm factors using the entire graph's topology for global
  //! degree access
  void InitNormFactor();

  //////////////////////////////////////////////////////////////////////////////
  // GPU things
  //////////////////////////////////////////////////////////////////////////////

#ifdef GALOIS_ENABLE_GPU
  //! This satisfies the cuda context forward declaration in host decls:
  //! context fields
  GNNGraphGPUAllocations gpu_memory_;
  //! Call this to setup GPU memory for this graph: allocates necessary GPU
  //! memory and copies things over
  void InitGPUMemory();
#endif
};

} // namespace graphs
} // namespace galois

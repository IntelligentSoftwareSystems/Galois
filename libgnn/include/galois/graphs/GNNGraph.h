#pragma once

#include "galois/GNNTypes.h"
#include "galois/graphs/CuSPPartitioner.h"
#include "galois/graphs/GluonSubstrate.h"

namespace galois {

// TODO remove the need to hardcode this path
//! Path to location of all gnn files
static const std::string gnn_dataset_path =
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
  // using LocalGraphType    = LearningGraph;
  using GNNDistGraph = galois::graphs::DistGraph<char, void>;

  //! Loads a graph and all relevant metadata (labels, features, masks, etc.)
  GNNGraph(const std::string& dataset_name, GNNPartitionScheme partition_scheme,
           bool has_single_class_label);

private:
  //! In a multi-host setting, this variable stores the host id that the graph
  //! is currently running on
  unsigned host_id_;
  //! Number of classes for a single vertex label
  size_t num_label_classes_{1};
  //! Length of a feature node
  size_t node_feature_length_{0};
  //! Partitioned graph
  std::unique_ptr<GNNDistGraph> partitioned_graph_;
  // XXX is this necessary
  //! Copy of underlying topology of the distributed graph
  // std::unique_ptr<LocalGraphType> local_graph_;
  //! Sync substrate for the partitioned graph
  std::unique_ptr<galois::graphs::GluonSubstrate<GNNDistGraph>> sync_substrate_;
  //! Ground truth label for nodes in the partitioned graph; Nx1 if single
  //! class, N x num classes if multi-class label
  std::unique_ptr<GNNLabel[]> local_ground_truth_labels_;
  //! Feature vectors for nodes in partitioned graph
  std::unique_ptr<GNNFeature[]> local_node_features_;

  // TODO maybe revisit this and use an actual bitset
  //! Bitset indicating which nodes are training nodes
  std::unique_ptr<GNNLabel[]> local_training_mask_;
  //! Bitset indicating which nodes are validation nodes
  std::unique_ptr<GNNLabel[]> local_validation_mask_;
  //! Bitset indicating which nodes are testing nodes
  std::unique_ptr<GNNLabel[]> local_testing_mask_;

  //! Global mask range for training nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_training_mask_range_;
  //! Global mask range for validation nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_validation_mask_range_;
  //! Global mask range for testing nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_testing_mask_range_;

  // XXX figure out what this is really used for
  //! Normalization constant based on structure of the graph
  std::vector<GNNFloat> norm_factors_;

  // TODO vars for subgraphs as necessary

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

  // public:
  //
  //  DGraph* getGraphPointer() { return partitionedGraph; }
  //  Graph* getLGraphPointer() { return lGraph; }
  //  Graph* getSubgraphPointer(int id) { return partitionedSubgraphs[id]; };
  //
  //  void initializeSyncSubstrate();
  //
  //  void saveDistGraph(DGraph* a);
  //  galois::graphs::GluonSubstrate<DGraph>* getSyncSubstrate();
  //  float_t* get_feats_ptr() { return h_feats; }
  //  float_t* get_feats_subg_ptr() { return h_feats_subg.data(); }
  //  label_t* get_labels_ptr() { return h_labels; }
  //  label_t* get_labels_subg_ptr() { return h_labels_subg.data(); }
  //  float_t* get_norm_factors_ptr() { return normFactors.data(); }
  //  float_t* get_norm_factors_subg_ptr() { return &normFactorsSub[0]; }
  //
  //  //! allocate the norm factor vector
  //  void allocNormFactor();
  //  void allocNormFactorSub(int subID);
  //  //! construct norm factor vector by using data from global graph
  //  void constructNormFactor(deepgalois::Context* globalContext);
  //  void constructNormFactorSub(int subgraphID);
  //
  //  void constructSubgraphLabels(size_t m, const mask_t* masks);
  //  void constructSubgraphFeatures(size_t m, const mask_t* masks);
  //
  //  //! return label for some node
  //  //! NOTE: this is LID, not GID
  //  label_t get_label(size_t lid) { return h_labels[lid]; }
  //
  //  //! returns pointer to the features of each local node
  //  float_t* get_in_ptr();
  //
  //  //! allocate memory for subgraphs (don't actually build them)
  //  void allocateSubgraphs(int num_subgraphs, unsigned max_size);
  //
  //  //! return if a vertex is owned by the partitioned graph this context
  //  contains bool isOwned(unsigned gid);
  //  //! return if part graph has provided vertex for given gid locally
  //  bool isLocal(unsigned gid);
  //  //! get GID of an lid for a vertex
  //  unsigned getGID(unsigned lid);
  //  //! get local id of a vertex given a global id for that vertex
  //  unsigned getLID(unsigned gid);
};

} // namespace graphs
} // namespace galois

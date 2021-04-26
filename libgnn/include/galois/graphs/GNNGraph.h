#pragma once

#include "galois/GNNTypes.h"
#include "galois/PerThreadRNG.h"
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
enum class GNNPartitionScheme { kOEC, kCVC, kOCVC };

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
  size_t GetNumLabelClasses() const { return num_label_classes_; }

  bool is_single_class_label() const { return using_single_class_labels_; }

  //////////////////////////////////////////////////////////////////////////////
  // Graph accessors
  //////////////////////////////////////////////////////////////////////////////

  //! Return # of nodes in the partitioned graph
  size_t size() const { return partitioned_graph_->size(); }
  //! Returns # of nodes in the *graph that is currently active*.
  size_t active_size() const {
    if (!use_subgraph_) {
      return partitioned_graph_->size();
    } else {
      return subgraph_->size();
    }
  }

  bool is_local(size_t gid) const { return partitioned_graph_->isLocal(gid); }
  size_t GetLID(size_t gid) const { return partitioned_graph_->getLID(gid); }
  size_t GetGID(size_t lid) const { return partitioned_graph_->getGID(lid); }

  //! Node begin for all local nodes
  NodeIterator begin() const {
    if (!use_subgraph_) {
      return partitioned_graph_->allNodesRange().begin();
    } else {
      return subgraph_->begin();
    }
  }
  //! Node end for all local nodes
  NodeIterator end() const {
    if (!use_subgraph_) {
      return partitioned_graph_->allNodesRange().end();
    } else {
      return subgraph_->end();
    }
  }

  NodeIterator begin_owned() const {
    if (!use_subgraph_) {
      return partitioned_graph_->masterNodesRange().begin();
    } else {
      return subgraph_->begin_owned();
    }
  }

  NodeIterator end_owned() const {
    if (!use_subgraph_) {
      return partitioned_graph_->masterNodesRange().end();
    } else {
      return subgraph_->end_owned();
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Edges
  //////////////////////////////////////////////////////////////////////////////

  void InitializeSamplingData() { InitializeSamplingData(1); }
  //! Initialize data required to do graph sampling
  void InitializeSamplingData(size_t num_layers);

  //////////////////////////////////////////////////////////////////////////////
  // Out Edges
  //////////////////////////////////////////////////////////////////////////////

  // All following functions take a local node id
  EdgeIterator edge_begin(GraphNode n) const {
    if (!use_subgraph_) {
      return partitioned_graph_->edge_begin(n);
    } else {
      return subgraph_->edge_begin(n);
    }
  };

  EdgeIterator edge_end(GraphNode n) const {
    if (!use_subgraph_) {
      return partitioned_graph_->edge_end(n);
    } else {
      return subgraph_->edge_end(n);
    }
  };
  GraphNode GetEdgeDest(EdgeIterator ei) const {
    if (!use_subgraph_) {
      return partitioned_graph_->getEdgeDst(ei);
    } else {
      return subgraph_->GetEdgeDest(ei);
    }
  };
  galois::runtime::iterable<
      galois::NoDerefIterator<GNNDistGraph::edge_iterator>>
  edges(GraphNode N) const {
    if (!use_subgraph_) {
      return partitioned_graph_->edges(N);
    } else {
      return subgraph_->edges(N);
    }
  }

  bool IsEdgeSampledAny(EdgeIterator ei) const {
    for (bool b : edge_sample_status_[*ei]) {
      if (b)
        return true;
    }
    return false;
  }
  bool IsEdgeSampled(uint32_t ei, size_t layer_num) const {
    if (!use_subgraph_) {
      return edge_sample_status_[ei][layer_num];
    } else {
      GALOIS_LOG_FATAL("This shouldn't be called with subgraph");
      return false;
    }
  };
  bool IsEdgeSampled(EdgeIterator ei, size_t layer_num) const {
    if (!use_subgraph_) {
      return edge_sample_status_[*ei][layer_num];
    } else {
      return subgraph_->OutEdgeSampled(ei, layer_num, *this);
    }
  };
  //! Always use original graph's edge iterator here
  bool IsEdgeSampledOriginalGraph(EdgeIterator ei, size_t layer_num) const {
    return edge_sample_status_[*ei][layer_num];
  };

  //! Set the flag on the edge to 1; makes it sampled
  void MakeEdgeSampled(EdgeIterator ei, size_t layer_num) {
    edge_sample_status_[*ei][layer_num] = 1;
  };
  //! Set the flag on the edge to 0; makes it not sampled
  void MakeEdgeUnsampled(EdgeIterator ei, size_t layer_num) {
    edge_sample_status_[*ei][layer_num] = 0;
  };

  //////////////////////////////////////////////////////////////////////////////
  // in edges
  //////////////////////////////////////////////////////////////////////////////
  EdgeIterator in_edge_begin(GraphNode n) const {
    if (!use_subgraph_) {
      return partitioned_graph_->in_edge_begin(n);
    } else {
      return subgraph_->in_edge_begin(n);
    }
  }
  EdgeIterator in_edge_end(GraphNode n) const {
    if (!use_subgraph_) {
      return partitioned_graph_->in_edge_end(n);
    } else {
      return subgraph_->in_edge_end(n);
    }
  }
  galois::runtime::iterable<
      galois::NoDerefIterator<GNNDistGraph::edge_iterator>>
  in_edges(GraphNode N) const {
    if (!use_subgraph_) {
      return partitioned_graph_->in_edges(N);
    } else {
      return subgraph_->in_edges(N);
    }
  }
  GraphNode GetInEdgeDest(EdgeIterator ei) const {
    if (!use_subgraph_) {
      return partitioned_graph_->GetInEdgeDest(ei);
    } else {
      return subgraph_->GetInEdgeDest(ei);
    }
  };

  EdgeIterator InEdgeToOutEdge(EdgeIterator in_edge_iter) const {
    return partitioned_graph_->InEdgeToOutEdge(in_edge_iter);
  }

  bool IsInEdgeSampledAny(EdgeIterator ei) const {
    for (bool b :
         edge_sample_status_[partitioned_graph_->InEdgeToOutEdge(ei)]) {
      if (b)
        return true;
    }
    return false;
  };
  bool IsInEdgeSampled(EdgeIterator ei, size_t layer_num) const {
    if (!use_subgraph_) {
      return edge_sample_status_[partitioned_graph_->InEdgeToOutEdge(ei)]
                                [layer_num];
    } else {
      return subgraph_->InEdgeSampled(ei, layer_num, *this);
    }
  };

  //! Set the flag on the edge to 1; makes it sampled
  void MakeInEdgeSampled(EdgeIterator ei, size_t layer_num) {
    edge_sample_status_[partitioned_graph_->InEdgeToOutEdge(ei)][layer_num] = 1;
  };
  //! Set the flag on the edge to 0; makes it not sampled
  void MakeInEdgeUnsampled(EdgeIterator ei, size_t layer_num) {
    edge_sample_status_[partitioned_graph_->InEdgeToOutEdge(ei)][layer_num] = 0;
  };

  //////////////////////////////////////////////////////////////////////////////
  // neighborhood sampling
  //////////////////////////////////////////////////////////////////////////////

  //! Set seed nodes, i.e., nodes that are being predicted on
  void SetupNeighborhoodSample();

  //! Choose all edges from sampled nodes
  void SampleAllEdges(size_t agg_layer_num);
  //! Sample neighbors of nodes that are marked as ready for sampling
  void SampleEdges(size_t sample_layer_num, size_t num_to_sample);

  //! Construct the subgraph from sampled edges and corresponding nodes
  size_t ConstructSampledSubgraph() {
    // false first so that the build process can use functions to access the
    // real graph
    use_subgraph_             = false;
    size_t num_subgraph_nodes = subgraph_->BuildSubgraph(*this);
    // after this, this graph is a subgraph
    use_subgraph_ = true;
    return num_subgraph_nodes;
  }

  void EnableSubgraph() { use_subgraph_ = true; }

  void DisableSubgraph() { use_subgraph_ = false; }

  //////////////////////////////////////////////////////////////////////////////

  GNNFloat GetNormFactor(GraphNode n) const { return norm_factors_[n]; }
  //! Degree norm (1 / degree) of current functional graph (e.g., sampled,
  //! inductive graph, etc); calculated whenever norm factor is calculated
  GNNFloat GetDegreeNorm(GraphNode n) const {
    if (!use_subgraph_) {
      return degree_norm_[n];
    } else {
      // XXX does not work in distributed case, fix there
      // XXX also need to account for current layer number in sampling
      // case because degrees in each layer differ
      return 1.0 / subgraph_->GetLocalDegree(n);
    }
  }

  // Get accuracy: sampling is by default false
  float GetGlobalAccuracy(PointerWithSize<GNNFloat> predictions,
                          GNNPhase phase);
  float GetGlobalAccuracy(PointerWithSize<GNNFloat> predictions, GNNPhase phase,
                          bool sampling);

  //! Returns the ground truth label of some local id assuming labels are single
  //! class labels.
  GNNFloat GetSingleClassLabel(const unsigned lid) const {
    assert(using_single_class_labels_);
    unsigned to_use = lid;
    if (use_subgraph_) {
      to_use = subgraph_->SIDToLID(lid);
    }

    if (local_ground_truth_labels_[to_use] != num_label_classes_) {
      // galois::gPrint(lid, " ", to_use, " ",
      // (int)local_ground_truth_labels_[to_use], "\n");
      return local_ground_truth_labels_[to_use];
    } else {
      GALOIS_LOG_FATAL(
          "should not get the label of a node that has no ground truth {}",
          to_use);
    }
  }

  //! Returns pointer to start of ground truth vector for some local id assuming
  //! labels are multi-class.
  const GNNLabel* GetMultiClassLabel(const unsigned lid) const {
    assert(!using_single_class_labels_);
    return static_cast<const GNNLabel*>(local_ground_truth_labels_.data() +
                                        (lid * num_label_classes_));
  }

  //! Return matrix of the local node features
  const PointerWithSize<GNNFloat> GetLocalFeatures() {
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      // TODO remove reliance on local_node_features
      return PointerWithSize(gpu_memory_.feature_vector(),
                             local_node_features_.size());
    }
#endif
    if (!use_subgraph_) {
      return PointerWithSize(local_node_features_);
    } else {
      return PointerWithSize(subgraph_->GetLocalFeatures().data(),
                             subgraph_->GetLocalFeatures().size());
    }
  }

  //! Given an LID and the current phase of GNN computation, determine if the
  //! lid in question is valid for the current phase (i.e., it is part of
  //! a training, validation, or test phase mask)
  bool IsValidForPhase(const unsigned lid,
                       const galois::GNNPhase current_phase) const {
    // XXX maybe just map this all over to subgraph, though in that case
    // issue is that subgraph doesn't necessarily know about test/val
    unsigned to_use = lid;
    if (use_subgraph_) {
      to_use = subgraph_->SIDToLID(lid);
    }
    if (!incomplete_masks_ && current_phase != GNNPhase::kOther) {
      return IsValidForPhaseCompleteRange(to_use, current_phase);
    } else {
      return IsValidForPhaseMasked(to_use, current_phase);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  //! Given a matrix and the column size, do an aggregate sync where each row
  //! is considered a node's data and sync using the graph's Gluon
  //! substrate
  //! Note that it's const because the only thing being used is the graph
  //! topology of this object; the thing modified is the passed in matrix
  void AggregateSync(GNNFloat* matrix_to_sync,
                     const size_t matrix_column_size) const;

  //////////////////////////////////////////////////////////////////////////////
  // Sampling related
  //////////////////////////////////////////////////////////////////////////////

  //! Loops through all master nodes and determines if it is "on" or "off"
  //! (the meaning of on and off depends on how it is used; for now, it is used
  //! to indicate subgraph presence); droprate controls chance of being dropped
  //! (e.g. if 0.8, a node is 80% likely to not be included in subgraph)
  void UniformNodeSample() { UniformNodeSample(0.5); }
  void UniformNodeSample(float droprate);

  //! Use the sampling method present in GraphSAINT
  void GraphSAINTSample() { GraphSAINTSample(3000, 2); };
  void GraphSAINTSample(size_t num_roots, size_t walk_depth);

  //! Makes a node "sampled"; used for debugging/testing
  void SetSampledNode(size_t node) { partitioned_graph_->getData(node) = 1; }
  //! Makes a node "not sampled"; used for debugging/testing
  void UnsetSampledNode(size_t node) { partitioned_graph_->getData(node) = 0; }

  //! Returns true if a particular node is currently considered "in" a sampled
  //! graph
  bool IsInSampledGraph(const NodeIterator& ni) const {
    // TODO(loc) GPU
    assert(*ni < size());
    return partitioned_graph_->getData(*ni);
  }
  bool IsInSampledGraph(size_t node_id) const {
    // TODO(loc) GPU
    assert(node_id < size());
    return partitioned_graph_->getData(node_id);
  }

  //! Calculate norm factor considering the entire graph
  void CalculateFullNormFactor();
  //! Calculate norm factor considering sampled nodes and/or training nodes
  //! only (inductive)
  void CalculateSpecialNormFactor(bool is_sampled, bool is_inductive);

#ifdef GALOIS_ENABLE_GPU
  void AggregateSync(GNNFloat* matrix_to_sync, const size_t matrix_column_size,
                     const unsigned layer_number) const;

  void InitLayerVectorMetaObjects(size_t layer_number, unsigned num_hosts,
                                  size_t infl_in_size, size_t infl_out_size);

  void ResizeLayerVector(size_t num_layers);

  const GNNGraphGPUAllocations& GetGPUGraph() const { return gpu_memory_; }

  void GetMarshalGraph(MarshalGraph& m) const {
    sync_substrate_->getMarshalGraph(m, false);
  }

  void GetPartitionedGraphInfo(PartitionedGraphInfo& g_info) const {
    sync_substrate_->getPartitionedGraphInfo(g_info);
  }
#endif

private:
// included like this to avoid cyclic dependency issues + not used anywhere but
// in this class anyways
#include "galois/graphs/GNNSubgraph.h"

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
                                GNNRange* mask_range, char* masks);
  //! Finds nodes that aren't part of the 3 main GNN phase classifications
  size_t FindOtherMask();
  //! Read masks of local nodes only for training, validation, and testing
  void ReadLocalMasks(const std::string& dataset_name);
  //! Reads the entire graph topology in (but nothing else)
  void ReadWholeGraph(const std::string& dataset_name);
  //! Initializes the norm factors using the entire graph's topology for global
  //! degree access
  void InitNormFactor();

  //! Used if ranges for a mask are complete (if in range, it's part of mask).
  bool IsValidForPhaseCompleteRange(const unsigned lid,
                                    const galois::GNNPhase current_phase) const;

  //! Used if ranges for a mask are incomplete, meaning I actually have to
  //! check the mask.
  bool IsValidForPhaseMasked(const unsigned lid,
                             const galois::GNNPhase current_phase) const;

  //////////////////////////////////////////////////////////////////////////////
  // Accuracy
  //////////////////////////////////////////////////////////////////////////////

  float GetGlobalAccuracyCPU(PointerWithSize<GNNFloat> predictions,
                             GNNPhase phase, bool sampling);
  float GetGlobalAccuracyCPUSingle(PointerWithSize<GNNFloat> predictions,
                                   GNNPhase phase, bool sampling);
  float GetGlobalAccuracyCPUMulti(PointerWithSize<GNNFloat> predictions,
                                  GNNPhase phase, bool sampling);

  //////////////////////////////////////////////////////////////////////////////
  // Vars
  //////////////////////////////////////////////////////////////////////////////

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

  //////////////////////////////////////////////////////////////////////////////

  std::unique_ptr<GNNSubgraph> subgraph_;
  // Degrees for sampled subgraph
  galois::LargeArray<uint32_t> sampled_out_degrees_;
  galois::LargeArray<uint32_t> sampled_in_degrees_;
  //! Sample data on edges: each edge gets a small bitset to mark
  //! if it's been sampled for a particular layer
  galois::LargeArray<std::vector<bool>> edge_sample_status_;
  //! Indicates newly sampled nodes (for distributed synchronization of sampling
  //! status
  galois::DynamicBitSet new_sampled_nodes_;

  //////////////////////////////////////////////////////////////////////////////

  // TODO maybe revisit this and use an actual bitset
  //! Bitset indicating which nodes are training nodes
  std::vector<char> local_training_mask_;
  //! Bitset indicating which nodes are validation nodes
  std::vector<char> local_validation_mask_;
  //! Bitset indicating which nodes are testing nodes
  std::vector<char> local_testing_mask_;
  size_t valid_other_{0};
  //! Bitset indicating which nodes don't fall anywhere
  std::vector<char> other_mask_;

  //! Global mask range for training nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_training_mask_range_;
  //! Global mask range for validation nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_validation_mask_range_;
  //! Global mask range for testing nodes; must convert to LIDs when using
  //! in this class
  GNNRange global_testing_mask_range_;

  //! If true, then node splits of train/val/test aren't complete (i.e.
  //! falling in range != part of that set)
  bool incomplete_masks_{false};

  //! Normalization constant based on structure of the graph (degrees)
  std::vector<GNNFloat> norm_factors_;
  //! Normalization constant based on degrees (unlike nomral norm factors
  //! it's only division without a square root)
  std::vector<GNNFloat> degree_norm_;

  //! RNG for subgraph sampling
  galois::PerThreadRNG sample_rng_;

  // TODO vars for subgraphs as necessary
  bool use_subgraph_{false};

  //////////////////////////////////////////////////////////////////////////////
  // GPU things
  //////////////////////////////////////////////////////////////////////////////

#ifdef GALOIS_ENABLE_GPU
  struct CUDA_Context* cuda_ctx_;
  //! Object that holds all GPU allocated pointers to memory related to graphs.
  GNNGraphGPUAllocations gpu_memory_;
  //! Call this to setup GPU memory for this graph: allocates necessary GPU
  //! memory and copies things over
  void InitGPUMemory();
#endif
  //! Used to track accurate predictions during accuracy calculation
  DGAccumulator<size_t> num_correct_;
  //! Used to count total number of things checked during accuracy calculation
  DGAccumulator<size_t> total_checked_;
  // Below are used for multi-class accuracy
  DGAccumulator<size_t> local_true_positive_;
  DGAccumulator<size_t> local_true_negative_;
  DGAccumulator<size_t> local_false_positive_;
  DGAccumulator<size_t> local_false_negative_;
};

} // namespace graphs
} // namespace galois

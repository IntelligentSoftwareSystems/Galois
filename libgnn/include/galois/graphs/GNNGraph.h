#pragma once

#include "galois/GNNTypes.h"
#include "galois/PerThreadRNG.h"
#include "galois/graphs/CuSPPartitioner.h"
#include "galois/graphs/GluonSubstrate.h"
#include "galois/graphs/GraphAggregationSyncStructures.h"
#include "galois/MinibatchGenerator.h"

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
  using GraphNode    = GNNDistGraph::GraphNode;
  // defined as such because dist graph range objects used long unsigned
  using NodeIterator = boost::counting_iterator<size_t>;
  using EdgeIterator = GNNDistGraph::edge_iterator;

  // using GNNEdgeSortIterator = internal::EdgeSortIterator<GraphNode,
  //  uint64_t, galois::LargeArray<uint32_t>,
  //  galois::LargeArray<std::vector<bool>>>;

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
    if (!use_subgraph_ && !use_subgraph_view_) {
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
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->allNodesRange().begin();
    } else {
      return subgraph_->begin();
    }
  }
  //! Node end for all local nodes
  NodeIterator end() const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->allNodesRange().end();
    } else {
      return subgraph_->end();
    }
  }

  NodeIterator begin_owned() const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->masterNodesRange().begin();
    } else {
      return subgraph_->begin_owned();
    }
  }

  NodeIterator end_owned() const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->masterNodesRange().end();
    } else {
      return subgraph_->end_owned();
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Edges
  //////////////////////////////////////////////////////////////////////////////

  void InitializeSamplingData() { InitializeSamplingData(1, false); }
  //! Initialize data required to do graph sampling
  void InitializeSamplingData(size_t num_layers, bool is_inductive);

  //////////////////////////////////////////////////////////////////////////////
  // Out Edges
  //////////////////////////////////////////////////////////////////////////////

  // All following functions take a local node id
  EdgeIterator edge_begin(GraphNode n) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->edge_begin(n);
    } else if (use_subgraph_view_) {
      return partitioned_graph_->edge_begin(ConvertToLID(n));
    } else {
      return subgraph_->edge_begin(n);
    }
  };

  EdgeIterator edge_end(GraphNode n) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->edge_end(n);
    } else if (use_subgraph_view_) {
      return partitioned_graph_->edge_end(ConvertToLID(n));
    } else {
      return subgraph_->edge_end(n);
    }
  };
  GraphNode GetEdgeDest(EdgeIterator ei) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->getEdgeDst(ei);
    } else if (use_subgraph_view_) {
      // WARNING: this may return max of uint32 if the edge destination doesn't
      // exist in the subgraph view
      // get edge dest should NOT be called in that case
      GraphNode rv = ConvertToSID(partitioned_graph_->getEdgeDst(ei));
      assert(rv != std::numeric_limits<uint32_t>::max());
      return rv;
    } else {
      return subgraph_->GetEdgeDest(ei);
    }
  };

  galois::runtime::iterable<
      galois::NoDerefIterator<GNNDistGraph::edge_iterator>>
  edges(GraphNode N) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->edges(N);
    } else if (use_subgraph_view_) {
      return partitioned_graph_->edges(ConvertToLID(N));
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
      // view uses original graph edge iterators
      return edge_sample_status_[ei][layer_num];
    } else {
      return subgraph_->OutEdgeSampled(ei, layer_num, *this);
      return false;
    }
  };
  bool IsEdgeSampled(EdgeIterator ei, size_t layer_num) const {
    if (!use_subgraph_) {
      // view uses original graph edge iterators
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

  // GNNEdgeSortIterator EdgeSortBegin(GraphNode n) {
  //  return GNNEdgeSortIterator(*edge_begin(n),
  //  partitioned_graph_->edge_dst_ptr_LA(), &edge_sample_status_);
  //}
  // GNNEdgeSortIterator EdgeSortEnd(GraphNode n) {
  //  return GNNEdgeSortIterator(*edge_begin(n),
  //  partitioned_graph_->edge_dst_ptr_LA(), &edge_sample_status_);
  //}

  //////////////////////////////////////////////////////////////////////////////
  // in edges
  //////////////////////////////////////////////////////////////////////////////
  EdgeIterator in_edge_begin(GraphNode n) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->in_edge_begin(n);
    } else if (use_subgraph_view_) {
      return partitioned_graph_->in_edge_begin(ConvertToLID(n));
    } else {
      return subgraph_->in_edge_begin(n);
    }
  }
  EdgeIterator in_edge_end(GraphNode n) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->in_edge_end(n);
    } else if (use_subgraph_view_) {
      return partitioned_graph_->in_edge_end(ConvertToLID(n));
    } else {
      return subgraph_->in_edge_end(n);
    }
  }
  galois::runtime::iterable<
      galois::NoDerefIterator<GNNDistGraph::edge_iterator>>
  in_edges(GraphNode N) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->in_edges(N);
    } else if (use_subgraph_view_) {
      return partitioned_graph_->in_edges(ConvertToLID(N));
    } else {
      return subgraph_->in_edges(N);
    }
  }
  GraphNode GetInEdgeDest(EdgeIterator ei) const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->GetInEdgeDest(ei);
    } else if (use_subgraph_view_) {
      return partitioned_graph_->GetInEdgeDest(ei);
      GraphNode rv = ConvertToSID(partitioned_graph_->GetInEdgeDest(ei));
      assert(rv != std::numeric_limits<uint32_t>::max());
      return rv;
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
      // view can use this fine + requires it
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
  size_t SetupNeighborhoodSample() {
    return SetupNeighborhoodSample(GNNPhase::kTrain);
  }
  size_t SetupNeighborhoodSample(GNNPhase seed_phase);

  //! Choose all edges from sampled nodes
  size_t SampleAllEdges(size_t agg_layer_num, bool inductive_subgraph,
                        size_t timestamp);
  //! Sample neighbors of nodes that are marked as ready for sampling
  size_t SampleEdges(size_t sample_layer_num, size_t num_to_sample,
                     bool inductive_subgraph, size_t timestamp);

  size_t ConstructSampledSubgraph(size_t num_sampled_layers) {
    return ConstructSampledSubgraph(num_sampled_layers, false);
  };
  //! Construct the subgraph from sampled edges and corresponding nodes
  size_t ConstructSampledSubgraph(size_t num_sampled_layers, bool use_view);

  unsigned SampleNodeTimestamp(unsigned lid) const {
    return sample_node_timestamps_[lid];
  }

  void EnableSubgraph() { use_subgraph_ = true; }
  void EnableSubgraphView() { use_subgraph_view_ = true; }
  void DisableSubgraph() {
    use_subgraph_      = false;
    use_subgraph_view_ = false;
  }
  bool IsSubgraphOn() const { return use_subgraph_; }
  bool IsSubgraphViewOn() const { return use_subgraph_view_; }

  //! Converts an id to an lid for the graph if subgraphs are in use
  uint32_t ConvertToLID(GraphNode sid) const {
    if (use_subgraph_ || use_subgraph_view_) {
      return subgraph_->SIDToLID(sid);
    } else {
      return sid;
    }
  }
  //! Converts an LID to an SID if subgraphs are in use
  uint32_t ConvertToSID(GraphNode lid) const {
    if (use_subgraph_ || use_subgraph_view_) {
      return subgraph_->LIDToSID(lid);
    } else {
      return lid;
    }
  }
  //! Converts SID to GID if subgraphs in use (else just return GID)
  uint32_t SIDToGID(GraphNode sid) const {
    if (use_subgraph_ || use_subgraph_view_) {
      return GetGID(subgraph_->SIDToLID(sid));
    } else {
      return GetGID(sid);
    }
  }
  //! Returns a pointer to the LID to SID map from the subgraph if subgraphs
  //! are in use
  galois::LargeArray<uint32_t>* GetLIDToSIDPointer() {
    if (use_subgraph_ || use_subgraph_view_) {
      return subgraph_->GetLIDToSIDPointer();
    } else {
      return nullptr;
    }
  }

  // void SortAllInEdgesBySID() {
  //  // check it out for node 0
  //  //for (auto iter : in_edges(0)) {
  //  //  galois::gInfo("0 to ", GetInEdgeDest(*iter), " with in out edge map ",
  //  *InEdgeToOutEdge(iter), " SID ",
  //  subgraph_->LIDToSID(GetInEdgeDest(*iter)));
  //  //}
  //  //galois::gInfo("Starting sort");
  //  galois::StatTimer t("SortBySID");
  //  t.start();
  //  partitioned_graph_->SortAllInEdgesBySID(*(subgraph_->GetLIDToSIDPointer()));
  //  t.stop();
  //  galois::gInfo("sort took ", t.get());
  //  //galois::gInfo("End Sort");
  //  //for (auto iter : in_edges(0)) {
  //  //  galois::gInfo("0 to ", GetInEdgeDest(*iter), " with in out edge map ",
  //  *InEdgeToOutEdge(iter), " SID ",
  //  subgraph_->LIDToSID(GetInEdgeDest(*iter)));
  //  //}
  //}

  //////////////////////////////////////////////////////////////////////////////
  void SetupTrainBatcher(size_t train_batch_size) {
    if (train_batcher_) {
      // clear before remake
      train_batcher_.reset();
    }
    train_batcher_ = std::make_unique<MinibatchGenerator>(
        local_training_mask_, train_batch_size, *end_owned());
    local_minibatch_mask_.resize(partitioned_graph_->size());
  }

  void ResetTrainMinibatcher() { train_batcher_->ResetMinibatchState(); }

  //! Setup the state for the next minibatch sampling call by using the
  //! minibatcher to pick up the next set batch of nodes
  size_t PrepareNextTrainMinibatch();
  //! Returns true if there are still more minibatches in this graph
  bool MoreTrainMinibatches() { return !train_batcher_->NoMoreMinibatches(); };

  //////////////////////////////////////////////////////////////////////////////

  void SetupTestBatcher(size_t test_batch_size) {
    if (test_batcher_) {
      // clear before remake
      test_batcher_.reset();
    }
    test_batcher_ = std::make_unique<MinibatchGenerator>(
        local_testing_mask_, test_batch_size, *end_owned());
    local_minibatch_mask_.resize(partitioned_graph_->size());
  }
  void ResetTestMinibatcher() { test_batcher_->ResetMinibatchState(); }
  //! Setup the state for the next minibatch sampling call by using the
  //! minibatcher to pick up the next set batch of nodes
  size_t PrepareNextTestMinibatch();
  //! Returns true if there are still more minibatches in this graph
  bool MoreTestMinibatches() { return !test_batcher_->NoMoreMinibatches(); };

  //////////////////////////////////////////////////////////////////////////////
  GNNFloat GetGCNNormFactor(GraphNode lid) const {
    if (global_degrees_[lid]) {
      return 1.0 / std::sqrt(static_cast<float>(global_degrees_[lid]) + 1);
    } else {
      return 0.0;
    }
  }

  GNNFloat GetGlobalDegreeNorm(GraphNode n) const {
    if (global_degrees_[n]) {
      return 1.0 / global_degrees_[n];
    } else {
      return 0.0;
    }
  }

  GNNFloat GetGlobalTrainDegreeNorm(GraphNode n) const {
    if (global_train_degrees_[n]) {
      return 1.0 / global_train_degrees_[n];
    } else {
      return 0.0;
    }
  }

  //! Get degree norm of subgraph for particular layer (i.e. includes training)
  GNNFloat GetDegreeNorm(GraphNode n, size_t graph_user_layer_num) const {
    if (use_subgraph_ || use_subgraph_view_) {
      size_t degree;
      if (!subgraph_choose_all_) {
        // case because degrees in each layer differ
        degree =
            sampled_out_degrees_[graph_user_layer_num][subgraph_->SIDToLID(n)];
      } else {
        // XXX if inductive
        // degree = global_train_degrees_[subgraph_->SIDToLID(n)];
        degree = global_degrees_[subgraph_->SIDToLID(n)];
      }
      if (degree) {
        return 1.0 / degree;
      } else {
        return 0;
      }
    } else {
      return GetGlobalDegreeNorm(n);
    }
  }

  // Get accuracy: sampling is by default false
  float GetGlobalAccuracy(PointerWithSize<GNNFloat> predictions,
                          GNNPhase phase);
  float GetGlobalAccuracy(PointerWithSize<GNNFloat> predictions, GNNPhase phase,
                          bool sampling);

  std::pair<uint32_t, uint32_t>
  GetBatchAccuracy(PointerWithSize<GNNFloat> predictions);

  //! Returns the ground truth label of some local id assuming labels are single
  //! class labels.
  GNNFloat GetSingleClassLabel(const unsigned lid) const {
    assert(using_single_class_labels_);
    unsigned to_use = lid;
    if (use_subgraph_ || use_subgraph_view_) {
      to_use = subgraph_->SIDToLID(lid);
    }

    if (local_ground_truth_labels_[to_use] != num_label_classes_) {
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
    if (!use_subgraph_ && !use_subgraph_view_) {
      return PointerWithSize(local_node_features_);
    } else {
      return PointerWithSize(subgraph_->GetLocalFeatures().data(),
                             subgraph_->size() * node_feature_length_);
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
    if (use_subgraph_ || use_subgraph_view_) {
      to_use = subgraph_->SIDToLID(lid);
    }
    // re: phase checks in this if: ranges are not used for these
    // phases even if they might exist; it's something to look into
    // possibly, though at the same time it may not be worth it
    if (!incomplete_masks_ && current_phase != GNNPhase::kOther &&
        current_phase != GNNPhase::kBatch) {
      return IsValidForPhaseCompleteRange(to_use, current_phase);
    } else {
      return IsValidForPhaseMasked(to_use, current_phase);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  // TODO(loc) Should not be a default version of this to avoid potential
  // issues later
  void AggregateSync(GNNFloat* matrix_to_sync,
                     const size_t matrix_column_size) const {
    AggregateSync(matrix_to_sync, matrix_column_size, false);
  };

  //! Given a matrix and the column size, do an aggregate sync where each row
  //! is considered a node's data and sync using the graph's Gluon
  //! substrate
  //! Note that it's const because the only thing being used is the graph
  //! topology of this object; the thing modified is the passed in matrix
  void AggregateSync(GNNFloat* matrix_to_sync, const size_t matrix_column_size,
                     bool is_backward) const;

  //////////////////////////////////////////////////////////////////////////////
  // Sampling related
  //////////////////////////////////////////////////////////////////////////////

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
  bool IsInSampledGraphSubgraph(size_t node_id) const {
    // TODO(loc) GPU
    assert(node_id < size());
    if (use_subgraph_) {
      return partitioned_graph_->getData(ConvertToLID(node_id));
    } else {
      return partitioned_graph_->getData(node_id);
    }
  }

  //! Calculate norm factor considering the entire graph
  void CalculateFullNormFactor();

#ifdef GALOIS_ENABLE_GPU
  void AggregateSyncGPU(GNNFloat* matrix_to_sync,
                        const size_t matrix_column_size,
                        const unsigned layer_number) const;

  void InitLayerVectorMetaObjects(size_t layer_number, unsigned num_hosts,
                                  size_t infl_in_size, size_t infl_out_size);

  void ResizeGPULayerVector(size_t num_layers);

  const GNNGraphGPUAllocations& GetGPUGraph() const { return gpu_memory_; }

  void GetMarshalGraph(MarshalGraph& m) const {
    sync_substrate_->getMarshalGraph(m, false);
  }

  void GetPartitionedGraphInfo(PartitionedGraphInfo& g_info) const {
    sync_substrate_->getPartitionedGraphInfo(g_info);
  }
#endif

  void ContiguousRemap(const std::string& new_name);

  void EnableTimers() {
    use_timer_ = true;
    if (subgraph_) {
      subgraph_->EnableTimers();
    }
  }
  void DisableTimers() {
    use_timer_ = false;
    if (subgraph_) {
      subgraph_->DisableTimers();
    }
  }

  bool SubgraphChooseAllStatus() { return subgraph_choose_all_; }
  void EnableSubgraphChooseAll() { subgraph_choose_all_ = true; }
  void DisableSubgraphChooseAll() { subgraph_choose_all_ = false; }
  void SetSubgraphChooseAll(bool a) { subgraph_choose_all_ = a; }

private:
// included like this to avoid cyclic dependency issues + not used anywhere but
// in this class anyways
#include "galois/graphs/GNNSubgraph.h"

  //////////////////////////////////////////////////////////////////////////////
  // Initialization
  //////////////////////////////////////////////////////////////////////////////

  void ReadLocalLabelsBin(const std::string& dataset_name);
  //! Read labels of local nodes only
  void ReadLocalLabels(const std::string& dataset_name,
                       bool has_single_class_label);
  //! Read features of local nodes only
  void ReadLocalFeatures(const std::string& dataset_str);
  //! Helper function to read masks from file into the appropriate structures
  //! given a name, mask type, and arrays to save into
  size_t ReadLocalMasksFromFile(const std::string& dataset_name,
                                const std::string& mask_type,
                                GNNRange* mask_range, std::vector<char>* masks);
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
  std::vector<galois::LargeArray<uint32_t>> sampled_out_degrees_;
  //! Sample data on edges: each edge gets a small bitset to mark
  //! if it's been sampled for a particular layer
  galois::LargeArray<std::vector<bool>> edge_sample_status_;
  // TODO use a char maybe? unlikely anyone will go over 2^8 layers...
  //! What timestep a node was added to sampled set; used to determine
  //! size of subgraph at each layer
  galois::LargeArray<unsigned> sample_node_timestamps_;
  //! Indicates newly sampled nodes (for distributed synchronization of sampling
  //! status
  galois::DynamicBitSet new_sampled_nodes_;

  //////////////////////////////////////////////////////////////////////////////

  // TODO maybe revisit this and use an actual bitset
  //! Bitset indicating which nodes are training nodes
  GNNMask local_training_mask_;
  //! Bitset indicating which nodes are validation nodes
  GNNMask local_validation_mask_;
  //! Bitset indicating which nodes are testing nodes
  GNNMask local_testing_mask_;
  //! Bitset indicating which nodes don't fall anywhere
  GNNMask other_mask_;
  //! Bitset indicating which nodes are part of the minibatch
  GNNMask local_minibatch_mask_;

  size_t valid_other_{0};

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

  //! RNG for subgraph sampling
  galois::PerThreadRNG sample_rng_;

  // TODO LargeArray instead of vector?
  //! Degrees: needed since graph is distributed
  std::vector<uint32_t> global_degrees_;
  std::vector<uint32_t> global_train_degrees_;

  // TODO vars for subgraphs as necessary
  bool use_subgraph_{false};
  bool use_subgraph_view_{false};
  bool subgraph_choose_all_{false};

  std::unique_ptr<MinibatchGenerator> train_batcher_;
  std::unique_ptr<MinibatchGenerator> test_batcher_;

  std::vector<uint32_t> node_remapping_;

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

  bool use_timer_{true};
};

} // namespace graphs
} // namespace galois

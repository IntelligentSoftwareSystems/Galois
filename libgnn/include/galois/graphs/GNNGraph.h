#pragma once

#include "galois/GNNTypes.h"
#include "galois/PerThreadRNG.h"
#include "galois/graphs/CuSPPartitioner.h"
#include "galois/graphs/GluonSubstrate.h"
#include "galois/graphs/GraphAggregationSyncStructures.h"
#include "galois/MinibatchGenerator.h"
#include "galois/Logging.h"
#include "galois/graphs/ReadGraph.h"
#include "galois/GNNMath.h"
#include "galois/graphs/DegreeSyncStructures.h"

#include <fstream>
#include <limits>
#include <unordered_set>

#ifdef GALOIS_ENABLE_GPU
#include "galois/graphs/GNNGraph.cuh"
#endif

namespace galois {

// TODO remove the need to hardcode this path
//! Path to location of all gnn files
static const std::string default_gnn_dataset_path =
    "/home/hochan/inputs/Learning/";

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

template <typename VTy, typename ETy>
class GNNGraph {
public:
  using GNNDistGraph = galois::graphs::DistGraph<VTy, ETy>;
  using GraphNode    = typename GNNDistGraph::GraphNode;
  // defined as such because dist graph range objects used long unsigned
  using NodeIterator = boost::counting_iterator<size_t>;
  using EdgeIterator = typename GNNDistGraph::edge_iterator;

  // using GNNEdgeSortIterator = internal::EdgeSortIterator<GraphNode,
  //  uint64_t, galois::LargeArray<uint32_t>,
  //  galois::LargeArray<std::vector<bool>>>;

  GNNGraph(const std::string& dataset_name, GNNPartitionScheme partition_scheme,
           bool has_single_class_label, bool use_wmd = false)
      : GNNGraph(galois::default_gnn_dataset_path, dataset_name,
                 partition_scheme, has_single_class_label, use_wmd) {}

  //! Loads a graph and all relevant metadata (labels, features, masks, etc.)
  GNNGraph(const std::string& input_directory, const std::string& dataset_name,
           GNNPartitionScheme partition_scheme, bool has_single_class_label,
           bool use_wmd = false)
      : input_directory_(input_directory), use_wmd_(use_wmd) {
    GALOIS_LOG_VERBOSE("[{}] Constructing partitioning for {}", host_id_,
                       dataset_name);
    // save host id
    host_id_ = galois::runtime::getSystemNetworkInterface().ID;
    host_prefix_ =
        std::string("[") +
        std::to_string(galois::runtime::getSystemNetworkInterface().ID) +
        std::string("] ");
    // load partition
    partitioned_graph_ =
        LoadPartition(input_directory_, dataset_name, partition_scheme);
    galois::gInfo(host_prefix_, "Loading partition is completed");
    // reverse edges
    partitioned_graph_->ConstructIncomingEdges();
    // mark a node if it is sampled
    mark_sampled_nodes_.resize(partitioned_graph_->size());

    galois::gInfo(host_prefix_, "Number of local proxies is ",
                  partitioned_graph_->size());
    galois::gInfo(host_prefix_, "Number of local edges is ",
                  partitioned_graph_->sizeEdges());

    // init gluon from the partitioned graph
    sync_substrate_ =
        std::make_unique<galois::graphs::GluonSubstrate<GNNDistGraph>>(
            *partitioned_graph_, host_id_,
            galois::runtime::getSystemNetworkInterface().Num, false,
            partitioned_graph_->cartesianGrid());
    bitset_graph_aggregate.resize(partitioned_graph_->size());

    // Construct/read additional graph data
    if (use_wmd) {
      galois::gInfo("Feature is constructed by aggregating 2-hop features, "
                    "instead from feature files");
      this->ConstructFeatureBy2HopAggregation();
      this->ConstructLocalLabels();
      this->SetLocalMasksRandomly();
    } else {
      if (dataset_name != "ogbn-papers100M-remap") {
        ReadLocalLabels(dataset_name, has_single_class_label);
      } else {
        galois::gInfo("Remapped ogbn 100M");
        ReadLocalLabelsBin(dataset_name);
      }
      ReadLocalFeatures(dataset_name);
      ReadLocalMasks(dataset_name);
    }

    // init norm factors (involves a sync call)
    InitNormFactor();

#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      // allocate/copy data structures over to GPU
      GALOIS_LOG_VERBOSE("[{}] Initializing GPU memory", host_id_);
      InitGPUMemory();

      // initialize CUDA context
      cuda_ctx_ = get_CUDA_context(host_id_);
      if (!init_CUDA_context(cuda_ctx_, ::gpudevice)) {
        GALOIS_DIE("Failed to initialize CUDA context");
      }
      PartitionedGraphInfo g_info;
      GetPartitionedGraphInfo(g_info);
      load_graph_CUDA_GNN(cuda_ctx_, g_info,
                          galois::runtime::getSystemNetworkInterface().Num);
    }
#endif
  }

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
  size_t global_size() const { return partitioned_graph_->globalSize(); }
  //! Returns # of nodes in the *graph that is currently active*.
  size_t active_size() const {
    if (!use_subgraph_ && !use_subgraph_view_) {
      return partitioned_graph_->size();
    } else {
      return subgraph_->size();
    }
  }

  bool is_owned(size_t gid) const { return partitioned_graph_->isOwned(gid); }
  bool is_local(size_t gid) const { return partitioned_graph_->isLocal(gid); }
  size_t GetLID(size_t gid) const { return partitioned_graph_->getLID(gid); }
  size_t GetGID(size_t lid) const { return partitioned_graph_->getGID(lid); }
  size_t GetHostID(size_t gid) const {
    return partitioned_graph_->getHostID(gid);
  }

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
  void InitializeSamplingData(size_t num_layers, bool choose_all) {
    subgraph_ = std::make_unique<GNNSubgraph>(partitioned_graph_->size());
    sample_node_timestamps_.create(partitioned_graph_->size(),
                                   std::numeric_limits<uint32_t>::max());
    edge_sample_status_.resize(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
      edge_sample_status_[i].resize(partitioned_graph_->sizeEdges());
    }
    sampled_edges_.resize(partitioned_graph_->sizeEdges());
    // this is to hold the degree of a sampled graph considering all hosts; yes,
    // memory wise this is slightly problematic possibly, but each layer is its
    // own subgraph
    if (!choose_all) {
      sampled_out_degrees_.resize(num_layers);
      for (galois::LargeArray<uint32_t>& array : sampled_out_degrees_) {
        array.create(partitioned_graph_->size());
      }
    } else {
      subgraph_choose_all_ = true;
    }
    definitely_sampled_nodes_.resize(partitioned_graph_->size());
    master_offset_accum_.resize(num_layers + 1);
    mirror_offset_accum_.resize(num_layers + 1);
    sample_master_offsets_.resize(num_layers + 1, 0);
    sample_mirror_offsets_.resize(num_layers + 1, 0);
  }

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
      galois::NoDerefIterator<typename GNNDistGraph::edge_iterator>>
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
    return sampled_edges_.test(*ei);
  }
  bool IsEdgeSampled(uint32_t ei, size_t layer_num) const {
    if (!use_subgraph_) {
      // view uses original graph edge iterators
      return edge_sample_status_[layer_num].test(ei);
    } else {
      return subgraph_->OutEdgeSampled(ei, layer_num, *this);
    }
  };
  bool IsEdgeSampled(EdgeIterator ei, size_t layer_num) const {
    if (!use_subgraph_) {
      // view uses original graph edge iterators
      return edge_sample_status_[layer_num].test(*ei);
    } else {
      return subgraph_->OutEdgeSampled(ei, layer_num, *this);
    }
  };
  //! Always use original graph's edge iterator here
  bool IsEdgeSampledOriginalGraph(EdgeIterator ei, size_t layer_num) const {
    return edge_sample_status_[layer_num].test(*ei);
  };

  //! Set the flag on the edge to 1; makes it sampled
  void MakeEdgeSampled(EdgeIterator ei, size_t layer_num) {
    sampled_edges_.set(*ei);
    edge_sample_status_[layer_num].set(*ei);
  };
  //! Set the flag on the edge to 0; makes it not sampled
  void MakeEdgeUnsampled(EdgeIterator ei, size_t layer_num) {
    edge_sample_status_[layer_num].reset(*ei, *ei);
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
      galois::NoDerefIterator<typename GNNDistGraph::edge_iterator>>
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
    return sampled_edges_.test(partitioned_graph_->InEdgeToOutEdge(ei));
  };
  bool IsInEdgeSampled(EdgeIterator ei, size_t layer_num) const {
    if (!use_subgraph_) {
      // view can use this fine + requires it
      return edge_sample_status_[layer_num].test(
          partitioned_graph_->InEdgeToOutEdge(ei));
    } else {
      return subgraph_->InEdgeSampled(ei, layer_num, *this);
    }
  };

  //! Set the flag on the edge to 1; makes it sampled
  void MakeInEdgeSampled(EdgeIterator ei, size_t layer_num) {
    edge_sample_status_[layer_num].set(partitioned_graph_->InEdgeToOutEdge(ei));
  };
  //! Set the flag on the edge to 0; makes it not sampled
  void MakeInEdgeUnsampled(EdgeIterator ei, size_t layer_num) {
    edge_sample_status_[layer_num].reset(
        partitioned_graph_->InEdgeToOutEdge(ei),
        partitioned_graph_->InEdgeToOutEdge(ei));
  };

  //////////////////////////////////////////////////////////////////////////////
  // neighborhood sampling
  //////////////////////////////////////////////////////////////////////////////

  //! Set seed nodes, i.e., nodes that are being predicted on
  size_t SetupNeighborhoodSample() {
    return SetupNeighborhoodSample(GNNPhase::kTrain);
  }
  size_t SetupNeighborhoodSample(GNNPhase seed_phase) {
    DisableSubgraph();

    if (!bitset_sample_flag_.size()) {
      bitset_sample_flag_.resize(size());
    }
    bitset_sample_flag_.ParallelReset();
    definitely_sampled_nodes_.ParallelReset();

    galois::do_all(
        galois::iterate(begin_owned(), end_owned()),
        [&](const NodeIterator& x) {
          if (IsValidForPhase(*x, seed_phase)) {
            SetSampledNode(*x);
            bitset_sample_flag_.set(*x);
            definitely_sampled_nodes_.set(*x);
          } else {
            UnsetSampledNode(*x);
          }
        },
        galois::loopname("InitialSeedSetting"));
    // unsets nodes set in previous iterations; for some reason they get
    // synchronized along  with everything else even though bitset sample flag
    // should prevent it (that, or it's because they don't get sync'd that they
    // remain the same)
    galois::do_all(galois::iterate(end_owned(), end()),
                   [&](const NodeIterator& x) { UnsetSampledNode(*x); });

    // clear node timestamps
    galois::StatTimer fill_time("ClearFillTime");
    fill_time.start();
    galois::ParallelSTL::fill(sample_node_timestamps_.begin(),
                              sample_node_timestamps_.end(),
                              std::numeric_limits<uint32_t>::max());
    galois::ParallelSTL::fill(sample_master_offsets_.begin(),
                              sample_master_offsets_.end(), 0);
    galois::ParallelSTL::fill(sample_mirror_offsets_.begin(),
                              sample_mirror_offsets_.end(), 0);
    fill_time.stop();

    for (unsigned i = 0; i < master_offset_accum_.size(); i++) {
      master_offset_accum_[i].reset();
      mirror_offset_accum_[i].reset();
    }

    // clear all sampled edges
    galois::StatTimer ctime("ClearSampleEdges");
    ctime.start();
    for (galois::DynamicBitSet& edge_layer : edge_sample_status_) {
      edge_layer.ParallelReset();
    }
    ctime.stop();
    //  galois::do_all(
    //      galois::iterate(edge_sample_status_.begin(),
    //      edge_sample_status_.end()),
    //      [&](galois::DynamicBitSet& edge_layer) { edge_layer.reset(); },
    //      galois::loopname("ClearSampleEdges"));

    sampled_edges_.ParallelReset();

    // reset all degrees
    if (!subgraph_choose_all_) {
      galois::StatTimer cad_timer("ClearAllDegrees");
      cad_timer.start();
      for (galois::LargeArray<uint32_t>& array : sampled_out_degrees_) {
        galois::ParallelSTL::fill(array.begin(), array.end(), 0);
      }
      cad_timer.stop();
    }

    if (!bitset_sampled_degrees_.size()) {
      bitset_sampled_degrees_.resize(partitioned_graph_->size());
    }
    bitset_sampled_degrees_.reset();

    // Seed nodes sync
    SampleNodeSync("SeedNodeSample");

    galois::GAccumulator<unsigned> local_seed_count;
    local_seed_count.reset();
    galois::GAccumulator<unsigned> master_offset;
    master_offset.reset();
    galois::GAccumulator<unsigned> mirror_offset;
    mirror_offset.reset();
    // count # of seed nodes
    galois::do_all(
        galois::iterate(begin(), end()),
        [&](const NodeIterator& x) {
          if (IsInSampledGraph(x)) {
            if (*x < *end_owned()) {
              master_offset += 1;
            } else {
              // mirror
              mirror_offset += 1;
            }

            // galois::gInfo(host_prefix_, "Seed node is ", GetGID(*x));
            local_seed_count += 1;
            // 0 = seed node
            sample_node_timestamps_[*x] = 0;
          }
        },
        galois::loopname("SeedNodeOffsetCounting"));

    sample_master_offsets_[0] = master_offset.reduce();
    sample_mirror_offsets_[0] = mirror_offset.reduce();

    return local_seed_count.reduce();
  }

  //! Choose all edges from sampled nodes
  size_t SampleAllEdges(size_t agg_layer_num, bool inductive_subgraph,
                        size_t timestamp) {
    DisableSubgraph();

    galois::do_all(
        galois::iterate(begin(), end()),
        [&](const NodeIterator& src_iter) {
          // only operate on if sampled
          if (IsInSampledGraph(src_iter)) {
            // marks ALL edges of nodes that connect to train/other nodes
            for (auto edge_iter : partitioned_graph_->edges(*src_iter)) {
              // total += 1;
              if (inductive_subgraph) {
                if (!IsValidForPhase(partitioned_graph_->getEdgeDst(edge_iter),
                                     GNNPhase::kTrain) &&
                    !IsValidForPhase(partitioned_graph_->getEdgeDst(edge_iter),
                                     GNNPhase::kOther)) {
                  continue;
                }
              }
              MakeEdgeSampled(edge_iter, agg_layer_num);
              uint32_t dest = partitioned_graph_->getEdgeDst(edge_iter);
              if (!IsInSampledGraph(dest)) {
                bitset_sample_flag_.set(dest);
              }
              definitely_sampled_nodes_.set(*src_iter);
              definitely_sampled_nodes_.set(dest);
            }
          }
        },
        galois::steal(), galois::loopname("ChooseAllEdges"));

    // update nodes, then communicate update to all hosts so that they can
    // continue the exploration
    galois::do_all(
        galois::iterate(size_t{0}, bitset_sample_flag_.size()),
        [&](uint32_t new_node_id) {
          if (bitset_sample_flag_.test(new_node_id)) {
            SetSampledNode(new_node_id);
          }
        },
        galois::loopname("NeighborhoodSampleSet"));

    SampleNodeSync("SampleFlag");

    galois::GAccumulator<unsigned> local_sample_count;
    local_sample_count.reset();
    // count # of seed nodes
    galois::do_all(galois::iterate(begin(), end()), [&](const NodeIterator& x) {
      if (IsInSampledGraph(x)) {
        local_sample_count += 1;
        if (sample_node_timestamps_[*x] ==
            std::numeric_limits<uint32_t>::max()) {
          if (x < end_owned()) {
            // owned nodes that are activated on other hosts shoudl always
            // be activated because it's responsible for keeping others in
            // sync during comms; ignoring it = bad
            // TODO(gluon) make it so you don't have to deal with this
            // and just use host as a reducer point
            definitely_sampled_nodes_.set(*x);
          }
          sample_node_timestamps_[*x] = timestamp;
        }
      }
    });

    EnableSubgraphChooseAll();
    return local_sample_count.reduce();
  }

  //! Sample neighbors of nodes that are marked as ready for sampling
  size_t SampleEdges(size_t sample_layer_num, size_t num_to_sample,
                     bool inductive_subgraph, size_t timestamp) {
    use_subgraph_      = false;
    use_subgraph_view_ = false;

    galois::do_all(
        galois::iterate(begin(), end()),
        [&](const NodeIterator& src_iter) {
          // only operate on if sampled
          if (IsInSampledGraph(src_iter)) {
            // chance of not uniformly choosing an edge of this node
            // num_to_sample times (degree norm is 1 / degree)
            double probability_of_reject;
            if (!inductive_subgraph) {
              probability_of_reject =
                  std::pow(1 - GetGlobalDegreeNorm(*src_iter), num_to_sample);
            } else {
              probability_of_reject = std::pow(
                  1 - GetGlobalTrainDegreeNorm(*src_iter), num_to_sample);
            }

            // loop through edges, turn "on" edge with some probability
            for (auto edge_iter : partitioned_graph_->edges(*src_iter)) {
              if (sample_rng_.DoBernoulli(probability_of_reject)) {
                if (inductive_subgraph) {
                  // only take if node is training node or a node not classified
                  // into train/test/val
                  if (!IsValidForPhase(
                          partitioned_graph_->getEdgeDst(edge_iter),
                          GNNPhase::kTrain) &&
                      !IsValidForPhase(
                          partitioned_graph_->getEdgeDst(edge_iter),
                          GNNPhase::kOther)) {
                    continue;
                  }
                }

                uint32_t edge_dst = partitioned_graph_->getEdgeDst(edge_iter);
                // if here, it means edge accepted; set sampled on, mark
                // as part of next set
                MakeEdgeSampled(edge_iter, sample_layer_num);
                if (!IsInSampledGraph(edge_dst)) {
                  bitset_sample_flag_.set(edge_dst);
                }
                bitset_sampled_degrees_.set(*src_iter);
                definitely_sampled_nodes_.set(*src_iter);
                definitely_sampled_nodes_.set(edge_dst);
                // degree increment
                sampled_out_degrees_[sample_layer_num][*src_iter]++;
              }
            }
          }
        },
        galois::steal(), galois::loopname("NeighborhoodSample"));

    // update nodes, then communicate update to all hosts so that they can
    // continue the exploration
    galois::do_all(
        galois::iterate(size_t{0}, bitset_sample_flag_.size()),
        [&](uint32_t new_node_id) {
          if (bitset_sample_flag_.test(new_node_id)) {
            SetSampledNode(new_node_id);
          }
        },
        galois::loopname("NeighborhoodSampleSet"));

    // why not read source? even if it doesn't need to sample anything, it needs
    // to know that it's active so that subgraph construction can proceed
    // correctly
    SampleNodeSync("SampleFlag");

    // count sampled node size
    galois::GAccumulator<unsigned> local_sample_count;
    local_sample_count.reset();
    // count # of seed nodes
    galois::do_all(galois::iterate(begin(), end()), [&](const NodeIterator& x) {
      if (IsInSampledGraph(x)) {
        local_sample_count += 1;
        if (sample_node_timestamps_[*x] ==
            std::numeric_limits<uint32_t>::max()) {
          if (x < end_owned()) {
            // owned nodes that are activated on other hosts shoudl always
            // be activated because it's responsible for keeping others in
            // sync during comms; ignoring it = bad
            // TODO(gluon) make it so you don't have to deal with this
            // and just use host as a reducer point
            definitely_sampled_nodes_.set(*x);
          }
          sample_node_timestamps_[*x] = timestamp;
        }
      }
    });

    DisableSubgraphChooseAll();
    return local_sample_count.reduce();
  }

  std::vector<unsigned> ConstructSampledSubgraph(size_t num_sampled_layers) {
    return ConstructSampledSubgraph(num_sampled_layers, false);
  };
  //! Construct the subgraph from sampled edges and corresponding nodes
  std::vector<unsigned> ConstructSampledSubgraph(size_t num_sampled_layers,
                                                 bool use_view) {
    // false first so that the build process can use functions to access the
    // real graph
    DisableSubgraph();

    gnn_sampled_out_degrees_ = &sampled_out_degrees_;

    // first, sync the degres of the sampled edges across all hosts
    // read any because destinations need it to for reverse phase
    if (use_timer_) {
      sync_substrate_->template sync<
          writeSource, readAny, SubgraphDegreeSync<VTy>, SubgraphDegreeBitset>(
          "SubgraphDegree");
    } else {
      sync_substrate_->template sync<
          writeSource, readAny, SubgraphDegreeSync<VTy>, SubgraphDegreeBitset>(
          "Ignore");
    }

    galois::StatTimer offsets_n_rows_time("OffsetRowSubgraphTime");
    offsets_n_rows_time.start();
    galois::do_all(
        galois::iterate(begin(), end()),
        [&](const NodeIterator& x) {
          if (IsActiveInSubgraph(*x)) {
            if (sample_node_timestamps_[*x] !=
                std::numeric_limits<uint32_t>::max()) {
              if (*x < *end_owned()) {
                // master
                master_offset_accum_[sample_node_timestamps_[*x]] += 1;
              } else {
                // mirror
                mirror_offset_accum_[sample_node_timestamps_[*x]] += 1;
              }
            } else {
              GALOIS_LOG_FATAL(
                  "should have been timestamped at some point if active");
            }
          }
        },
        galois::loopname("MasterMirrorOffset"));

    std::vector<unsigned> new_rows(master_offset_accum_.size());
    for (unsigned i = 0; i < master_offset_accum_.size(); i++) {
      sample_master_offsets_[i] = master_offset_accum_[i].reduce();
      sample_mirror_offsets_[i] = mirror_offset_accum_[i].reduce();
      new_rows[i] = sample_master_offsets_[i] + sample_mirror_offsets_[i];
      if (i > 0) {
        new_rows[i] += new_rows[i - 1];
      }
    }

    offsets_n_rows_time.stop();

    if (!use_view) {
      subgraph_->BuildSubgraph(*this, num_sampled_layers);
    } else {
      // a view only has lid<->sid mappings
      subgraph_->BuildSubgraphView(*this, num_sampled_layers);
    }

    sync_substrate_->SetupSubgraphMirrors(subgraph_->GetSubgraphMirrors(),
                                          use_timer_);

    // after this, this graph is a subgraph
    if (!use_view) {
      use_subgraph_ = true;
    } else {
      use_subgraph_view_ = true;
    }

    return new_rows;
  }

  unsigned SampleNodeTimestamp(unsigned lid) const {
    return sample_node_timestamps_[lid];
  }

  void EnableSubgraph() { use_subgraph_ = true; }
  void EnableSubgraphView() { use_subgraph_view_ = true; }
  void DisableSubgraph() {
    use_subgraph_      = false;
    use_subgraph_view_ = false;
    sync_substrate_->RevertHandshakeToRealGraph();
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
  size_t SetupTrainBatcher(size_t train_batch_size) {
    if (train_batcher_) {
      // clear before remake
      train_batcher_.reset();
    }
    train_batcher_ = std::make_unique<MinibatchGenerator>(
        local_training_mask_, train_batch_size, *end_owned());
    train_batcher_->ShuffleMode();
    // train_batcher_->DistributedShuffleMode(*partitioned_graph_,
    // global_training_mask_, global_training_count_);
    local_minibatch_mask_.resize(partitioned_graph_->size());
    return train_batcher_->ShuffleMinibatchTotal();
  }

  void ResetTrainMinibatcher() { train_batcher_->ResetMinibatchState(); }

  //! Setup the state for the next minibatch sampling call by using the
  //! minibatcher to pick up the next set batch of nodes
  size_t PrepareNextTrainMinibatch() {
    train_batcher_->GetNextMinibatch(&local_minibatch_mask_);
#ifndef NDEBUG
    size_t count = 0;
    // galois::gPrint("Minibatch : ");
    for (unsigned i = 0; i < local_minibatch_mask_.size(); i++) {
      if (local_minibatch_mask_[i]) {
        // galois::gPrint(partitioned_graph_->getGID(i), ",");
        count++;
      }
    }
    // galois::gPrint("\n");
    galois::gInfo(host_prefix(), "Batched nodes ", count);
#endif
    return SetupNeighborhoodSample(GNNPhase::kBatch);
  }

  // Used with distributed minibatch tracker
  // size_t PrepareNextTrainMinibatch(size_t num_to_get) {
  //  train_batcher_->GetNextMinibatch(&local_minibatch_mask_, num_to_get);
  //  return SetupNeighborhoodSample(GNNPhase::kBatch);
  //}
  //! Returns true if there are still more minibatches in this graph
  bool MoreTrainMinibatches() { return !train_batcher_->NoMoreMinibatches(); };

  template <
      typename T                                                      = VTy,
      typename std::enable_if_t<std::is_same_v<T, shad::ShadNodeTy>>* = nullptr>
  void ConstructFeatureBy2HopAggregation() {
    galois::StatTimer timer("ConstructFeatureBy2HopAggregation");
    if (this->use_timer_) {
      timer.start();
    }

    // TODO(hc): This constant is from SHAD implementation.
    //           This will be an user parameter for general/flexible support.

    // The first 15 floats are for the current node feature,
    // and the another 15 floats are for the aggregated neighbor's node feature.
    // These two 15-dimension features are concateneated to a single feature
    // for each node.
    this->node_feature_length_ = 30;
    this->local_node_features_.resize(
        this->partitioned_graph_->size() * this->node_feature_length_, 0.f);
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      this->ConstructFeatureBy2HopAggregationGPU();
    } else {
#endif
      this->ConstructFeatureBy2HopAggregationCPU();
#ifdef GALOIS_ENABLE_GPU
    }
#endif

    if (this->use_timer_) {
      timer.stop();
    }
  }

  template <typename T = VTy,
            typename std::enable_if_t<!std::is_same_v<T, shad::ShadNodeTy>>* =
                nullptr>
  void ConstructFeatureBy2HopAggregation() {}

  void ConstructFeatureBy2HopAggregationGPU() {
    // TODO(hc): This might not be used in the future.
    //           This might be renamed to use "PANDO" instead of "GPU".
    //           For now, just following the existing code format.
    GALOIS_LOG_FATAL(
        "ConstructFeatureBy2HopAggregationGPU() is not supported.");
  }

  void ConstructFeatureBy2HopAggregationCPU() {
    galois::gInfo("Construct an initial feature on CPU by "
                  "aggregating and concatenating neighbors' features.");
    // this->PrintFeatures("0hop");
    //  this->FillTestNodeType();
    // this->PrintGraphTopo("before");
    this->Construct1HopFeatureCPU();
    // this->PrintFeatures("1hop");
    this->Construct2HopFeatureCPU();
    // this->PrintFeatures("2hop");
  }

  void PrintFeatures(std::string postfix) {
    // XXX(hc): Printing code for correctness check.
    auto& net        = galois::runtime::getSystemNetworkInterface();
    unsigned host_id = net.ID;
    std::ofstream fp(postfix + "." + std::to_string(host_id) + ".feat");
    for (size_t lid = 0; lid < this->partitioned_graph_->size(); ++lid) {
      /*
      size_t gid = this->partitioned_graph_->getGID(lid);
      fp << "src:" << gid << ", " <<
          this->partitioned_graph_->getData(lid).type << ", " <<
          this->partitioned_graph_->getData(lid).key << "\n";
      for (size_t i = 0; i < this->node_feature_length_; ++i) {
        fp << "\t [" << i << "] = " <<
            this->local_node_features_[lid * this->node_feature_length_ + i]
            << "\n";
      }
      */
      fp << this->partitioned_graph_->getData(lid).key;
      for (size_t i = 0; i < this->node_feature_length_; ++i) {
        fp << ","
           << this->local_node_features_[lid * this->node_feature_length_ + i];
      }
      fp << "\n";
    }
    fp.close();
  }

  /// Construct feature from 1-hop neighbors.
  /// This method traverses 1-hop outgoing neighbors from each vertex
  /// and constructs a histogram of the outgoing edge type and
  /// the outgoing neighbor type.
  void Construct1HopFeatureCPU() {
    auto& graph = *(this->partitioned_graph_);
    // Aggregate adjacent node and edge types and construct
    // an intermediate feature.
    galois::do_all(
        galois::iterate(size_t{0}, graph.size()),
        [&](size_t src_lid) {
          bitset_graph_aggregate.set(src_lid);
          for (auto edge_iter = graph.edge_begin(src_lid);
               edge_iter < graph.edge_end(src_lid); ++edge_iter) {
            size_t dst_lid     = graph.getEdgeDst(edge_iter);
            uint32_t dst_type  = graph.getData(dst_lid).type;
            uint64_t edge_type = graph.getEdgeData(edge_iter);
            // Aggregate out neighbors' types.
            ++this->local_node_features_[this->node_feature_length_ * src_lid +
                                         dst_type];
            // TODO(hc): Assume that edge type is always 0.
            //           So, the 0th feature value of a node should be
            //           (degree of the node + sum of type-0 neighbors).
            ++this->local_node_features_[this->node_feature_length_ * src_lid +
                                         edge_type];
          }
        },
        galois::steal(), galois::loopname("Construct1HopFeatureCPU"));

    gnn_matrix_to_sync_               = this->local_node_features_.data();
    gnn_matrix_to_sync_column_length_ = this->node_feature_length_;
    // All the source vertices reduce and update proxies' data
    // and both the source and destination vertices set those
    // updated data to their data.
    sync_substrate_->template sync<writeSource, readAny, GNNSumAggregate<VTy>,
                                   Bitset_graph_aggregate>(
        "GraphAggregateSync");
  }

  /// Construct feature from 2-hop neighbors.
  /// After `Construct1HopFeatureCPU()`, each vertex aggregates types of
  /// the outgoing edges and neighbors, and constructs a histogram for
  /// its feature. Now, in this method, each vertex aggregates those
  /// histograms from outgoing neighbors and constructs a new histogram.
  /// Then, each vertex appends this new histogram to the old histogram
  /// as its feature.
  void Construct2HopFeatureCPU() {
    auto& graph = *(this->partitioned_graph_);
    // Aggregate neighbor nodes' features and append (concatenate) it to the
    // current node feature. So the first half is the current node and
    // the next half is the aggregated node feature.
    galois::do_all(
        galois::iterate(size_t{0}, graph.size()),
        [&](size_t src_lid) {
          // Offset for the second part of the source node feature.
          size_t src_foffset = this->node_feature_length_ * src_lid +
                               this->node_feature_length_ / 2;
          bitset_graph_aggregate.set(src_lid);
          for (auto edge_iter = graph.edge_begin(src_lid);
               edge_iter < graph.edge_end(src_lid); ++edge_iter) {
            size_t dst_lid = graph.getEdgeDst(edge_iter);
            // Offset for the first part of the destination node feature.
            size_t dst_foffset = this->node_feature_length_ * dst_lid;
            for (size_t fid = 0; fid < this->node_feature_length_ / 2; ++fid) {
              // Aggregate outgoing neighbors' features and,
              // construct and append a new histogram to the old one.
              this->local_node_features_[src_foffset + fid] +=
                  this->local_node_features_[dst_foffset + fid];
            }
          }
        },
        galois::steal(), galois::loopname("Construct2HopFeatureCPU"));
    this->SHADFeatureAggregateSync(this->local_node_features_.data(),
                                   this->node_feature_length_);
  }

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
  size_t PrepareNextTestMinibatch() {
    test_batcher_->GetNextMinibatch(&local_minibatch_mask_);
    return SetupNeighborhoodSample(GNNPhase::kBatch);
  }

  //! Returns true if there are still more minibatches in this graph
  bool MoreTestMinibatches() { return !test_batcher_->NoMoreMinibatches(); };

  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Normalization factor calculation for GCN without graph sampling
   *
   * @detail This function calculates normalization factor for nodes
   * on a GCN layer, but not with graph sampling (ego graph construction).
   * This normalization is proposed in GCN paper, and its equation is
   * D^(-1/2)*A*D^(-1/2).
   * XXX(hc): This degraded accuracy when graph sampling was enabled.
   * That could be many reasons for that, for example, a graph was already
   * small, and so, sampled graphs across layers are too small to normalize,
   * or, it might be theoretical design reason as the original GCN
   * did not consider ego graph construction.
   * For example, the one possible reason is that backward phase and
   * forward phase edge iterators are also different and maybe need to
   * use different iterators.
   * For now, I stopped this analysis and
   * just enabled this method for only GCN without graph
   * sampling. With graph sampling, I used SAGE's graph normalization.
   */
  GNNFloat GetGCNNormFactor(GraphNode lid
                            /*, size_t graph_user_layer_num*/) const {
#if 0
    if (use_subgraph_ || use_subgraph_view_) {
      size_t degree;
      if (!subgraph_choose_all_) {
        // case because degrees in each layer differ
        degree =
            sampled_out_degrees_[graph_user_layer_num][
                subgraph_->SIDToLID(lid)];
      } else {
        // XXX if inductive
        // degree = global_train_degrees_[subgraph_->SIDToLID(n)];
        degree = global_degrees_[subgraph_->SIDToLID(lid)];
      }
      if (degree) {
        return 1.0 / std::sqrt(static_cast<float>(degree) + 1);
      } else {
        return 0;
      }
    } else {
      if (global_degrees_[lid]) {
        if (this->size() != this->active_size()) {
          std::cout << lid << " does not match\n";
        }
        return 1.0 / std::sqrt(static_cast<float>(global_degrees_[lid]) + 1);
      } else {
        return 0.0;
      }
    }
#endif
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
                          GNNPhase phase) {
    // No GPU version yet, but this is where it would be
    return GetGlobalAccuracy(predictions, phase, false);
  }

  float GetGlobalAccuracy(PointerWithSize<GNNFloat> predictions, GNNPhase phase,
                          bool sampling) {
    // No GPU version yet, but this is where it would be
    return GetGlobalAccuracyCPU(predictions, phase, sampling);
  }

  /**
   * @brief Compare predictions from a model and ground truths, and return the
   * results.
   */
  std::pair<float, float>
  GetGlobalAccuracyCheckResult(PointerWithSize<GNNFloat> predictions,
                               GNNPhase phase, bool sampling) {
    return GetGlobalAccuracyCPUSingle(predictions, phase, sampling);
  }

  std::pair<uint32_t, uint32_t>
  GetBatchAccuracy(PointerWithSize<GNNFloat> predictions) {
    // check owned nodes' accuracy
    num_correct_.reset();
    total_checked_.reset();

    galois::do_all(
        // will only loop over sampled nodes if sampling is on
        galois::iterate(begin_owned(), end_owned()),
        // this is possibly the subgraph id
        [&](const unsigned node_id) {
          if (IsValidForPhase(node_id, GNNPhase::kBatch)) {
            total_checked_ += 1;
            size_t predicted_label =
                galois::MaxIndex(num_label_classes_,
                                 &(predictions[node_id * num_label_classes_]));
            if (predicted_label ==
                static_cast<size_t>(GetSingleClassLabel(node_id))) {
              num_correct_ += 1;
            }
          }
        },
        // steal on as some threads may have nothing to work on
        galois::steal(), galois::loopname("GlobalAccuracy"));

    size_t global_correct = num_correct_.reduce();
    size_t global_checked = total_checked_.reduce();

    return std::make_pair(global_correct, global_checked);
  }

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

  //! @brief Variant of the plain feature aggregation.
  //! @detail This is a variant version of the dense feature aggregation
  //! that follows SHAD GNN feature construction. This aggregates features of
  //! the neighbor vertices that are from (vertex's feature offset +
  //! 1/2 * feature length) to (vertex's feature offset + feature length),
  //! to (vertex's feature offset) of the current vertex, from its proxies.
  //!
  //! @param matrix_to_sync Float pointer pointing to features of the target
  //! vertex
  //! @param matrix_column_size Feature length to calculate a base offset of
  //! each vertex
  void SHADFeatureAggregateSync(GNNFloat* matrix_to_sync,
                                const size_t matrix_column_size) const {
    gnn_matrix_to_sync_               = matrix_to_sync;
    gnn_matrix_to_sync_column_length_ = matrix_column_size;

    // set globals for the sync substrate
    if (use_timer_) {
      sync_substrate_
          ->template sync<writeSource, readAny, SHADGNNSumAggregate<VTy>,
                          Bitset_graph_aggregate>("SHADGraphAggregateSync");
    } else {
      sync_substrate_
          ->template sync<writeSource, readAny, SHADGNNSumAggregate<VTy>,
                          Bitset_graph_aggregate>("Ignore");
    }
  }

  void SampleNodeSync(std::string stat_str) {
    sampled_nodes_ = &(this->mark_sampled_nodes_);

    // set globals for the sync substrate
    if (use_timer_) {
      sync_substrate_->template sync<writeSource, readDestination,
                                     SampleFlagSync<VTy>, SampleFlagBitset>(
          stat_str);
    } else {
      sync_substrate_->template sync<writeSource, readDestination,
                                     SampleFlagSync<VTy>, SampleFlagBitset>(
          "Ignore");
    }
  }

  // TODO(loc) Should not be a default version of this to avoid potential
  // issues later
  void AggregateSync(GNNFloat* matrix_to_sync,
                     const size_t matrix_column_size) const {
    AggregateSync(matrix_to_sync, matrix_column_size, false,
                  std::numeric_limits<uint32_t>::max());
  };

  //! Given a matrix and the column size, do an aggregate sync where each row
  //! is considered a node's data and sync using the graph's Gluon
  //! substrate
  //! Note that it's const because the only thing being used is the graph
  //! topology of this object; the thing modified is the passed in matrix
  void AggregateSync(GNNFloat* matrix_to_sync, const size_t matrix_column_size,
                     bool is_backward, uint32_t active_row_boundary) const {
    gnn_matrix_to_sync_               = matrix_to_sync;
    gnn_matrix_to_sync_column_length_ = matrix_column_size;
    subgraph_size_                    = active_size();
    num_active_layer_rows_            = active_row_boundary;

    if (!use_subgraph_ && !use_subgraph_view_) {
      // set globals for the sync substrate
      if (!is_backward) {
        if (use_timer_) {
          sync_substrate_
              ->template sync<writeSource, readAny, GNNSumAggregate<VTy>,
                              Bitset_graph_aggregate>("GraphAggregateSync");
        } else {
          sync_substrate_
              ->template sync<writeSource, readAny, GNNSumAggregate<VTy>,
                              Bitset_graph_aggregate>("Ignore");
        }
      } else {
        galois::StatTimer clubbed_timer("Sync_BackwardSync", "Gluon");
        clubbed_timer.start();
        sync_substrate_
            ->template sync<writeDestination, readAny, GNNSumAggregate<VTy>,
                            Bitset_graph_aggregate>(
                "BackwardGraphAggregateSync");
        clubbed_timer.stop();
      }
    } else {
      // setup the SID to LID map for the sync substrate to use (SID != LID)
      gnn_lid_to_sid_pointer_ = subgraph_->GetLIDToSIDPointer();

      if (!is_backward) {
        if (use_timer_) {
          sync_substrate_
              ->template sync<writeSource, readAny, GNNSampleSumAggregate<VTy>,
                              Bitset_graph_aggregate>("GraphAggregateSync");
        } else {
          sync_substrate_
              ->template sync<writeSource, readAny, GNNSampleSumAggregate<VTy>,
                              Bitset_graph_aggregate>("Ignore");
        }
      } else {
        galois::StatTimer clubbed_timer("Sync_BackwardSync", "Gluon");
        clubbed_timer.start();
        sync_substrate_
            ->template sync<writeDestination, readAny,
                            GNNSampleSumAggregate<VTy>, Bitset_graph_aggregate>(
                "BackwardGraphAggregateSync");
        clubbed_timer.stop();
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Sampling related
  //////////////////////////////////////////////////////////////////////////////

  //! Makes a node "sampled"; used for debugging/testing
  void SetSampledNode(size_t node) { mark_sampled_nodes_[node] = 1; }
  //! Makes a node "not sampled"; used for debugging/testing
  void UnsetSampledNode(size_t node) { mark_sampled_nodes_[node] = 0; }

  //! Returns true if a particular node is currently considered "in" a sampled
  //! graph
  bool IsInSampledGraph(const NodeIterator& ni) const {
    // TODO(loc) GPU
    assert(*ni < size());
    return mark_sampled_nodes_[*ni];
  }
  bool IsInSampledGraph(size_t node_id) const {
    // TODO(loc) GPU
    assert(node_id < size());
    return mark_sampled_nodes_[node_id];
  }
  bool IsInSampledGraphSubgraph(size_t node_id) const {
    // TODO(loc) GPU
    assert(node_id < size());
    if (use_subgraph_) {
      return mark_sampled_nodes_[ConvertToLID(node_id)];
    } else {
      return mark_sampled_nodes_[node_id];
    }
  }

  bool IsActiveInSubgraph(size_t node_id) const {
    return definitely_sampled_nodes_.test(node_id);
  }

  //! Calculate norm factor considering the entire graph
  void CalculateFullNormFactor() {
    // TODO(loc) reset all degrees if this is called multiple times?
    // get the norm factor contribution for each node based on the GLOBAL graph
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), partitioned_graph_->size()),
        [&](size_t src) {
          for (auto edge_iter = partitioned_graph_->edge_begin(src);
               edge_iter != partitioned_graph_->edge_end(src); edge_iter++) {
            // count degrees for all + train/other
            size_t dest = GetEdgeDest(edge_iter);
            if (IsValidForPhase(dest, GNNPhase::kTrain) ||
                IsValidForPhase(dest, GNNPhase::kOther)) {
              global_train_degrees_[src] += 1;
            }
            global_degrees_[src] += 1;
          }
        },
        galois::loopname("CalculateLocalDegrees"));
    // degree sync
    gnn_degree_vec_1_ = global_train_degrees_.data();
    gnn_degree_vec_2_ = global_degrees_.data();
    sync_substrate_
        ->template sync<writeSource, readAny, InitialDegreeSync<VTy>>(
            "InitialDegreeSync");
  }

#ifdef GALOIS_ENABLE_GPU
  void AggregateSyncGPU(GNNFloat* matrix_to_sync,
                        const size_t matrix_column_size,
                        const unsigned layer_number) const {
    size_t layer_input_mtx_column_size =
        getLayerInputMatrixColumnSize(cuda_ctx_, layer_number);
    size_t layer_output_mtx_column_size =
        getLayerOutputMatrixColumnSize(cuda_ctx_, layer_number);
    // set globals for the sync substrate
    gnn_matrix_to_sync_               = matrix_to_sync;
    gnn_matrix_to_sync_column_length_ = matrix_column_size;
    cuda_ctx_for_sync                 = cuda_ctx_;
    layer_number_to_sync              = layer_number;
    // TODO bitset setting
    // call sync
    cudaSetLayerInputOutput(cuda_ctx_, matrix_to_sync, matrix_column_size,
                            size(), layer_number);

    // XXX no timer if use_timer is off
    if (gnn_matrix_to_sync_column_length_ == layer_input_mtx_column_size) {
      if (use_timer_) {
        sync_substrate_->template sync<writeSource, readAny,
                                       GNNSumAggregate_layer_input<VTy>>(
            "GraphAggregateSync", gnn_matrix_to_sync_column_length_);
      } else {
        sync_substrate_->template sync<writeSource, readAny,
                                       GNNSumAggregate_layer_input<VTy>>(
            "Ignore", gnn_matrix_to_sync_column_length_);
      }
    } else if (gnn_matrix_to_sync_column_length_ ==
               layer_output_mtx_column_size) {
      if (use_timer_) {
        sync_substrate_->template sync<writeSource, readAny,
                                       GNNSumAggregate_layer_output<VTy>>(
            "GraphAggregateSync", gnn_matrix_to_sync_column_length_);
      } else {
        sync_substrate_->template sync<writeSource, readAny,
                                       GNNSumAggregate_layer_output<VTy>>(
            "Ignore", gnn_matrix_to_sync_column_length_);
      }
    } else {
      GALOIS_LOG_FATAL("Column size of the synchronized matrix does not"
                       " match to the column size of the CUDA context");
    }
  }

  void InitLayerVectorMetaObjects(size_t layer_number, unsigned num_hosts,
                                  size_t infl_in_size, size_t infl_out_size) {
    init_CUDA_layer_vector_meta_obj(cuda_ctx_, layer_number, num_hosts, size(),
                                    infl_in_size, infl_out_size);
  }

  void ResizeGPULayerVector(size_t num_layers) {
    resize_CUDA_layer_vector(cuda_ctx_, num_layers);
  }

  const GNNGraphGPUAllocations& GetGPUGraph() const { return gpu_memory_; }

  void GetMarshalGraph(MarshalGraph& m) const {
    sync_substrate_->getMarshalGraph(m, false);
  }

  void GetPartitionedGraphInfo(PartitionedGraphInfo& g_info) const {
    sync_substrate_->getPartitionedGraphInfo(g_info);
  }
#endif

  void ContiguousRemap(const std::string& new_name) {
    node_remapping_.resize(partitioned_graph_->size());

    uint32_t new_node_id = 0;

    // serial loops because new ID needs to be kept consistent
    // first, train nodes
    for (size_t cur_node = 0; cur_node < partitioned_graph_->size();
         cur_node++) {
      if (IsValidForPhase(cur_node, GNNPhase::kTrain)) {
        node_remapping_[new_node_id++] = cur_node;
      }
    }
    galois::gInfo("Train nodes are from 0 to ", new_node_id);

    // second, val nodes
    uint32_t val_start = new_node_id;
    for (size_t cur_node = 0; cur_node < partitioned_graph_->size();
         cur_node++) {
      if (IsValidForPhase(cur_node, GNNPhase::kValidate)) {
        node_remapping_[new_node_id++] = cur_node;
      }
    }
    galois::gInfo("Val nodes are from ", val_start, " to ", new_node_id, "(",
                  new_node_id - val_start, ")");

    // third, test nodes
    uint32_t test_start = new_node_id;
    for (size_t cur_node = 0; cur_node < partitioned_graph_->size();
         cur_node++) {
      if (IsValidForPhase(cur_node, GNNPhase::kTest)) {
        node_remapping_[new_node_id++] = cur_node;
      }
    }
    galois::gInfo("Test nodes are from ", test_start, " to ", new_node_id, "(",
                  new_node_id - test_start, ")");

    // last, everything else
    uint32_t other_start = new_node_id;
    for (size_t cur_node = 0; cur_node < partitioned_graph_->size();
         cur_node++) {
      if (IsValidForPhase(cur_node, GNNPhase::kOther)) {
        node_remapping_[new_node_id++] = cur_node;
      }
    }
    galois::gInfo("Other nodes are from ", other_start, " to ", new_node_id,
                  "(", new_node_id - other_start, ")");
    GALOIS_LOG_ASSERT(new_node_id == partitioned_graph_->size());

    // save the mapping to a binary file for use by graph convert to deal with
    // the gr
    std::string label_filename = input_directory_ + new_name + "-mapping.bin";
    std::ofstream label_write_stream;
    label_write_stream.open(label_filename, std::ios::binary | std::ios::out);
    label_write_stream.write((char*)node_remapping_.data(),
                             sizeof(uint32_t) * node_remapping_.size());
    label_write_stream.close();
  }

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

  std::vector<unsigned>& GetMasterOffsets() { return sample_master_offsets_; }
  std::vector<unsigned>& GetMirrorOffsets() { return sample_mirror_offsets_; }

  galois::DynamicBitSet& GetNonLayerZeroMasters() {
    return non_layer_zero_masters_;
  }
  const galois::DynamicBitSet& GetNonLayerZeroMasters() const {
    return non_layer_zero_masters_;
  }

  // TODO(hc): `ResizeSamplingBitsets()` and
  // `GetDefinitelySampledNodesBset()` expose private member variables
  // for unit tests. Other than them, these should not be used.

  void ResizeSamplingBitsets() {
    if (!bitset_sampled_degrees_.size()) {
      bitset_sampled_degrees_.resize(partitioned_graph_->size());
    }
    if (!bitset_sample_flag_.size()) {
      bitset_sample_flag_.resize(size());
    }
    if (!definitely_sampled_nodes_.size()) {
      definitely_sampled_nodes_.resize(partitioned_graph_->size());
    }
  }

  galois::DynamicBitSet& GetDefinitelySampledNodesBset() {
    return definitely_sampled_nodes_;
  }

  /* @brief Return true if this is constructed from a WMD graph otherwise false.
   */
  bool is_using_wmd() { return this->use_wmd_; }

private:
// included like this to avoid cyclic dependency issues + not used anywhere but
// in this class anyways
#include "galois/graphs/GNNSubgraph.h"

  //////////////////////////////////////////////////////////////////////////////
  // Initialization
  //////////////////////////////////////////////////////////////////////////////

  //! Partitions a particular dataset given some partitioning scheme
  std::unique_ptr<GNNDistGraph>
  LoadPartition(const std::string& input_directory,
                const std::string& dataset_name,
                galois::graphs::GNNPartitionScheme partition_scheme) {
    // XXX input path
    std::string input_file = input_directory + dataset_name + ".csgr";
    if (this->use_wmd_) {
      input_file = dataset_name;
    }
    GALOIS_LOG_VERBOSE("Partition loading: File to read is {}", input_file);

    // load partition
    switch (partition_scheme) {
    case galois::graphs::GNNPartitionScheme::kOEC:
      return galois::cuspPartitionGraph<GnnOEC, VTy, ETy>(
          input_file, galois::CUSP_CSR, galois::CUSP_CSR, this->use_wmd_, true,
          "", "", false, 1);
    case galois::graphs::GNNPartitionScheme::kCVC:
      return galois::cuspPartitionGraph<GnnCVC, VTy, ETy>(
          input_file, galois::CUSP_CSR, galois::CUSP_CSR, this->use_wmd_, true,
          "", "", false, 1);
    case galois::graphs::GNNPartitionScheme::kOCVC:
      return galois::cuspPartitionGraph<GenericCVC, VTy, ETy>(
          input_file, galois::CUSP_CSR, galois::CUSP_CSR, this->use_wmd_, true,
          "", "", false, 1);
    default:
      GALOIS_LOG_FATAL("Error: partition scheme specified is invalid");
      return nullptr;
    }
  }

  template <
      typename T                                                      = VTy,
      typename std::enable_if_t<std::is_same_v<T, shad::ShadNodeTy>>* = nullptr>
  void ConstructLocalLabels() {
    GALOIS_LOG_VERBOSE("[{}] Constructing labels from disk...", host_id_);
    auto& graph = *(this->partitioned_graph_);
    // For WMD graph, we always assume a single class label.
    // allocate memory for labels
    // single-class (one-hot) label for each vertex: N x 1
    using_single_class_labels_ = true;
    local_ground_truth_labels_.resize(graph.size());
    // In WMD graphs, a vertex class is a vertex type.
    // As the vertex type is already materialized in a vertex data,
    // iterate a graph and extract that.
    // TODO(hc): Using concurrent set using a finer-grained lock
    // is better
    std::mutex label_class_set_mtx;
    std::unordered_set<int> label_class_set;
    num_label_classes_ = 0;
    galois::do_all(galois::iterate(size_t{0}, graph.size()), [&](size_t lid) {
      local_ground_truth_labels_[lid] = graph.getData(lid).type;
      label_class_set_mtx.lock();
      auto found = label_class_set.find(local_ground_truth_labels_[lid]);
      if (found == label_class_set.end()) {
        label_class_set.emplace(local_ground_truth_labels_[lid]);
        ++num_label_classes_;
      }
      label_class_set_mtx.unlock();
    });

    // Exchange found local vertex classes with other hosts to
    // calculate the total number of the classes.
    //
    // Serialize the label class set to a vector to serialize this data
    // to galois::runtime::SendBuffer. The current libdist does not
    // support std::set and std::unordered_set de/serialization.
    // TODO(hc): support this type of serialization.
    std::vector<int> label_vec(label_class_set.begin(), label_class_set.end());
    auto& net = galois::runtime::getSystemNetworkInterface();
    for (uint32_t h = 0; h < net.Num; ++h) {
      if (h == net.ID) {
        continue;
      }
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, label_vec);
      net.sendTagged(h, galois::runtime::evilPhase, std::move(b));
    }
    net.flush();
    for (uint32_t h = 0; h < net.Num - 1; ++h) {
      decltype(net.recieveTagged(galois::runtime::evilPhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase);
      } while (!p);

      std::vector<int> h_label_vec;
      galois::runtime::gDeserialize(p->second, h_label_vec);
      galois::do_all(galois::iterate(h_label_vec), [&](int i) {
        label_class_set_mtx.lock();
        auto found = label_class_set.find(i);
        if (found == label_class_set.end()) {
          label_class_set.emplace(i);
          // Increaes the number of classes only if
          // it was not found in the local host.
          ++num_label_classes_;
        }
        label_class_set_mtx.unlock();
      });
    }
    increment_evilPhase();
  }

  template <typename T = VTy,
            typename std::enable_if_t<!std::is_same_v<T, shad::ShadNodeTy>>* =
                nullptr>
  void ConstructLocalLabels() {}

  void ReadLocalLabelsBin(const std::string& dataset_name) {
    GALOIS_LOG_VERBOSE("[{}] Reading labels from disk...", host_id_);

    std::ifstream file_stream;
    file_stream.open(input_directory_ + dataset_name + "-labels-dims.txt",
                     std::ios::in);
    size_t num_nodes;
    file_stream >> num_nodes >> num_label_classes_ >> std::ws;
    assert(num_nodes == partitioned_graph_->globalSize());
    if (host_id_ == 0) {
      galois::gInfo("Number of label classes is ", num_label_classes_);
    }
    file_stream.close();

    std::string filename = input_directory_ + dataset_name + "-labels.bin";
    std::ifstream file_stream_bin;
    file_stream_bin.open(filename, std::ios::binary | std::ios::in);

    std::vector<GNNLabel> all_labels(num_nodes);
    // read all labels into a vector
    file_stream_bin.read((char*)all_labels.data(),
                         sizeof(GNNLabel) * num_nodes);

    using_single_class_labels_ = true;
    local_ground_truth_labels_.resize(partitioned_graph_->size());

    galois::GAccumulator<size_t> found_local_vertices;
    found_local_vertices.reset();

    // save only local ones; can do in parallel as well
    // assumes -1 already dealt with
    galois::do_all(galois::iterate(size_t{0}, partitioned_graph_->size()),
                   [&](size_t lid) {
                     local_ground_truth_labels_[lid] = all_labels[GetGID(lid)];
                     found_local_vertices += 1;
                   });

    size_t fli = found_local_vertices.reduce();
    galois::gInfo(host_prefix_, "Read ", fli, " labels (",
                  local_ground_truth_labels_.size() * double{4} / (1 << 30),
                  " GB)");
    GALOIS_LOG_ASSERT(fli == partitioned_graph_->size());
  }

  //! Read labels of local nodes only
  void ReadLocalLabels(const std::string& dataset_name,
                       bool has_single_class_label) {
    GALOIS_LOG_VERBOSE("[{}] Reading labels from disk...", host_id_);
    std::string filename;
    if (has_single_class_label) {
      filename = input_directory_ + dataset_name + "-labels.txt";
    } else {
      filename = input_directory_ + dataset_name + "-mlabels.txt";
    }

    // read file header, save num label classes while at it
    std::ifstream file_stream;
    file_stream.open(filename, std::ios::in);
    size_t num_nodes;
    file_stream >> num_nodes >> num_label_classes_ >> std::ws;
    assert(num_nodes == partitioned_graph_->globalSize());
    if (host_id_ == 0) {
      galois::gInfo("Number of label classes is ", num_label_classes_);
    }

    // allocate memory for labels
    if (has_single_class_label) {
      // single-class (one-hot) label for each vertex: N x 1
      using_single_class_labels_ = true;
      local_ground_truth_labels_.resize(partitioned_graph_->size());
    } else {
      // multi-class label for each vertex: N x num classes
      using_single_class_labels_ = false;
      local_ground_truth_labels_.resize(partitioned_graph_->size() *
                                        num_label_classes_);
    }

    size_t cur_gid              = 0;
    size_t found_local_vertices = 0;
    // each line contains a set of 0s and 1s
    std::string read_line;

    // loop through all labels of the graph
    while (std::getline(file_stream, read_line)) {
      // only process label if this node is local
      if (partitioned_graph_->isLocal(cur_gid)) {
        uint32_t cur_lid = partitioned_graph_->getLID(cur_gid);
        // read line as bitset of 0s and 1s
        std::istringstream label_stream(read_line);
        int cur_bit;
        // bitset size is # of label classes
        for (size_t cur_class = 0; cur_class < num_label_classes_;
             ++cur_class) {
          // read a bit
          label_stream >> cur_bit;

          if (has_single_class_label) {
            // no label
            if (cur_bit == -1) {
              local_ground_truth_labels_[cur_lid] = num_label_classes_;
              break;
            }

            // in single class, only 1 bit is set in bitset; that represents the
            // class to take
            if (cur_bit != 0) {
              // set class and break (assumption is that's the only bit that is
              // set)
              local_ground_truth_labels_[cur_lid] = cur_class;
              break;
            }
          } else {
            // else the entire bitset needs to be copied over to the label array
            // TODO this can possibly be saved all at once rather than bit by
            // bit?
            local_ground_truth_labels_[cur_lid * num_label_classes_ +
                                       cur_class] = cur_bit;
          }
        }
        found_local_vertices++;
      }
      // always increment cur_gid
      cur_gid++;
    }

    file_stream.close();

    galois::gInfo(host_prefix_, "Read ", found_local_vertices, " labels (",
                  local_ground_truth_labels_.size() * double{4} / (1 << 30),
                  " GB)");
    GALOIS_LOG_ASSERT(found_local_vertices == partitioned_graph_->size());
  }

  //! Read features of local nodes only
  void ReadLocalFeatures(const std::string& dataset_name) {
    GALOIS_LOG_VERBOSE("[{}] Reading features from disk...", host_id_);

    // read in dimensions of features, specifically node feature length
    size_t num_global_vertices;

    std::string file_dims = input_directory_ + dataset_name + "-dims.txt";
    std::ifstream ifs;
    ifs.open(file_dims, std::ios::in);
    ifs >> num_global_vertices >> node_feature_length_;
    ifs.close();

    GALOIS_LOG_ASSERT(num_global_vertices == partitioned_graph_->globalSize());
    GALOIS_LOG_VERBOSE("[{}] N x D: {} x {}", host_id_, num_global_vertices,
                       node_feature_length_);

    // memory for all features of all nodes in graph
    // TODO read features without loading entire feature file into memory; this
    // is quite inefficient
    std::unique_ptr<GNNFloat[]> full_feature_set = std::make_unique<GNNFloat[]>(
        num_global_vertices * node_feature_length_);

    // read in all features
    std::ifstream file_stream;
    std::string feature_file = input_directory_ + dataset_name + "-feats.bin";
    file_stream.open(feature_file, std::ios::binary | std::ios::in);
    file_stream.read((char*)full_feature_set.get(), sizeof(GNNFloat) *
                                                        num_global_vertices *
                                                        node_feature_length_);
    file_stream.close();

    // allocate memory for local features
    local_node_features_.resize(partitioned_graph_->size() *
                                node_feature_length_);

    // copy over features for local nodes only
    galois::GAccumulator<size_t> num_kept_vertices;
    num_kept_vertices.reset();
    galois::do_all(
        galois::iterate(size_t{0}, num_global_vertices), [&](size_t gid) {
          if (partitioned_graph_->isLocal(gid)) {
            // copy over feature vector
            std::copy(full_feature_set.get() + gid * node_feature_length_,
                      full_feature_set.get() + (gid + 1) * node_feature_length_,
                      &local_node_features_[partitioned_graph_->getLID(gid) *
                                            node_feature_length_]);
            num_kept_vertices += 1;
          }
        });
    full_feature_set.reset();

    galois::gInfo(host_prefix_, "Read ", local_node_features_.size(),
                  " features (",
                  local_node_features_.size() * double{4} / (1 << 30), " GB)");
    GALOIS_LOG_ASSERT(num_kept_vertices.reduce() == partitioned_graph_->size());
  }

  //! Helper function to read masks from file into the appropriate structures
  //! given a name, mask type, and arrays to save into
  size_t ReadLocalMasksFromFile(const std::string& dataset_name,
                                const std::string& mask_type,
                                GNNRange* mask_range,
                                std::vector<char>* masks) {
    size_t range_begin;
    size_t range_end;

    // read mask range
    std::string mask_filename =
        input_directory_ + dataset_name + "-" + mask_type + "_mask.txt";
    bool train_is_on = false;
    if (mask_type == "train") {
      train_is_on = true;
    }

    std::ifstream mask_stream;
    mask_stream.open(mask_filename, std::ios::in);
    mask_stream >> range_begin >> range_end >> std::ws;
    GALOIS_LOG_ASSERT(range_begin <= range_end);

    // set the range object
    mask_range->begin = range_begin;
    mask_range->end   = range_end;
    mask_range->size  = range_end - range_begin;

    size_t cur_line_num = 0;
    // valid nodes on this host
    size_t local_sample_count = 0;
    // this tracks TOTAL # of valid nodes in this group (not necessarily valid
    // ones on this host)
    size_t valid_count = 0;
    std::string line;
    // each line is a number signifying if mask is set for the vertex
    while (std::getline(mask_stream, line)) {
      std::istringstream mask_stream(line);
      // only examine vertices/lines in range
      if (cur_line_num >= range_begin && cur_line_num < range_end) {
        unsigned mask = 0;
        mask_stream >> mask;
        if (mask == 1) {
          valid_count++;
          if (partitioned_graph_->isLocal(cur_line_num)) {
            (*masks)[partitioned_graph_->getLID(cur_line_num)] = 1;
            local_sample_count++;
          }
          if (train_is_on) {
            global_training_mask_[cur_line_num] = 1;
          }
        }
      }
      cur_line_num++;
    }
    mask_stream.close();

    if (train_is_on) {
      global_training_count_ = valid_count;
    }

    if (valid_count != mask_range->size) {
      // overlapping masks: need to actually check the masks rather than use
      // ranges
      if (!incomplete_masks_) {
        galois::gInfo(
            "Masks are not contained in range: must actually check mask");
      }
      incomplete_masks_ = true;
    }

    return valid_count;
  }

  //! Finds nodes that aren't part of the 3 main GNN phase classifications
  size_t FindOtherMask() {
    galois::GAccumulator<size_t> other_accum;
    other_accum.reset();
    other_mask_.resize(partitioned_graph_->size());

    galois::do_all(
        galois::iterate(size_t{0}, partitioned_graph_->size()),
        [&](size_t local_id) {
          if (!IsValidForPhase(local_id, GNNPhase::kTrain) &&
              !IsValidForPhase(local_id, GNNPhase::kValidate) &&
              !IsValidForPhase(local_id, GNNPhase::kTest)) {
            other_mask_[local_id] = 1;
            other_accum += 1;
          }
        },
        galois::loopname("FindOtherMask"));
    return other_accum.reduce();
  }

  //! @brief Choose and set local training/validation/testing vertices
  //! consecutively.
  void SetLocalMasksConsecutively() {
    // allocate the memory for the local masks
    global_training_mask_.resize(partitioned_graph_->globalSize());
    local_training_mask_.resize(partitioned_graph_->size());
    local_validation_mask_.resize(partitioned_graph_->size());
    local_testing_mask_.resize(partitioned_graph_->size());

    global_training_count_        = partitioned_graph_->globalSize() / 4;
    size_t global_testing_count   = global_training_count_ / 2;
    global_training_mask_range_   = {.begin = 0,
                                     .end   = global_training_count_,
                                     .size  = global_training_count_};
    global_testing_mask_range_    = {.begin = global_training_count_,
                                     .end   = global_training_count_ +
                                            global_testing_count,
                                     .size = global_testing_count};
    global_validation_mask_range_ = {
        .begin = global_training_count_ + global_testing_count,
        .end   = global_training_count_ + 2 * global_testing_count,
        .size  = global_testing_count};
    // training
    for (size_t i = global_training_mask_range_.begin;
         i < global_training_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_training_mask_[partitioned_graph_->getLID(i)] = 1;
      }
      global_training_mask_[i] = 1;
    }

    // validation
    for (size_t i = global_validation_mask_range_.begin;
         i < global_validation_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_validation_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }

    // testing
    for (size_t i = global_testing_mask_range_.begin;
         i < global_testing_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_testing_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }
  }

  //! @brief Randomly choose and set local training/validation/testing
  //! vertices. This mimics what AGILE GNN does through Pytorch
  //! `DistributedRandomSampler`.
  void DistributedRandomSampling(size_t local_sample_size,
                                 std::vector<char>* masks) {
    // Pytorch's DistributedRandomSampler,
    // first materializes an array populated with
    // 0 to (num_local_vertices - 1), shuffles this array, and
    // extracts 0 to (num_local_shuffle - 1) vertices.
    // This method mimics this operation.
    // Like Pytorch, all the hosts use the same seed, and so,
    // deterministically choose each type of vertices for not only
    // the current host, but also others, and mark vertices to
    // the corresponding mask array if they are locals.
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::vector<std::pair<uint64_t, uint64_t>> num_masters_per_hosts(net.Num);
    std::pair<uint64_t, uint64_t> master_ranges = {
        partitioned_graph_->getGID(0),
        partitioned_graph_->getGID(partitioned_graph_->numMasters() - 1)};
    // 1) Exchange node master ranges, and so, each host knows
    // the range of vertex sampling.
    for (uint32_t h = 0; h < net.Num; ++h) {
      if (h == net.ID) {
        continue;
      }
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, master_ranges);
      net.sendTagged(h, galois::runtime::evilPhase, std::move(b));
    }
    net.flush();
    for (uint32_t h = 0; h < net.Num - 1; ++h) {
      decltype(net.recieveTagged(galois::runtime::evilPhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase);
      } while (!p);

      galois::runtime::gDeserialize(p->second, num_masters_per_hosts[p->first]);
    }
    increment_evilPhase();

    // 2) Sample vertices and mark them to the `masks` array
    // if a vertex is local.
    for (uint32_t h = 0; h < net.Num; ++h) {
      size_t h_begin =
          (h == net.ID) ? master_ranges.first : num_masters_per_hosts[h].first;
      size_t h_end = (h == net.ID) ? master_ranges.second
                                   : num_masters_per_hosts[h].second;
      std::vector<uint64_t> h_all_indices(h_end - h_begin);
      // Fill global vertex ids to h_global_ids.
      galois::do_all(galois::iterate(h_begin, h_end),
                     [&](size_t i) { h_all_indices[i - h_begin] = i; });
      std::mt19937 rand(0);
      std::shuffle(h_all_indices.begin(), h_all_indices.end(), rand);
      galois::do_all(
          galois::iterate(size_t{0}, local_sample_size), [&](size_t i) {
            // First, it doens't have duplications.
            // Second, only mark `masks` if the checking vertex is a local
            // vertex.
            if (partitioned_graph_->isLocal(h_all_indices[i])) {
              (*masks)[partitioned_graph_->getLID(h_all_indices[i])] = 1;
            }
          });
    }
  }

  void SetLocalMasksRandomly() {
    // allocate the memory for the local masks
    global_training_mask_.resize(partitioned_graph_->globalSize());
    local_training_mask_.resize(partitioned_graph_->size());
    local_validation_mask_.resize(partitioned_graph_->size());
    local_testing_mask_.resize(partitioned_graph_->size());

    auto& net                   = galois::runtime::getSystemNetworkInterface();
    global_training_count_      = partitioned_graph_->globalSize() / 4;
    size_t global_testing_count = global_training_count_ / 2;
    size_t num_local_training_samples   = global_training_count_ / net.Num;
    size_t num_local_testing_samples    = global_testing_count / net.Num;
    size_t num_local_validating_samples = num_local_testing_samples;
    global_training_mask_range_         = {.begin = 0,
                                           .end   = global_training_count_,
                                           .size  = global_training_count_};
    global_testing_mask_range_          = {.begin = 0,
                                           .end   = global_training_count_,
                                           .size  = global_training_count_};
    global_validation_mask_range_       = {.begin = 0,
                                           .end   = global_training_count_,
                                           .size  = global_training_count_};

    incomplete_masks_ = true;
    DistributedRandomSampling(num_local_training_samples,
                              &local_training_mask_);
    DistributedRandomSampling(num_local_testing_samples, &local_testing_mask_);
    DistributedRandomSampling(num_local_validating_samples,
                              &local_validation_mask_);
  }

  //! Read masks of local nodes only for training, validation, and testing
  void ReadLocalMasks(const std::string& dataset_name) {
    // allocate the memory for the local masks
    global_training_mask_.resize(partitioned_graph_->globalSize());
    local_training_mask_.resize(partitioned_graph_->size());
    local_validation_mask_.resize(partitioned_graph_->size());
    local_testing_mask_.resize(partitioned_graph_->size());

    if (dataset_name == "reddit") {
      global_training_count_ = 153431;

      // TODO reddit is hardcode handled at the moment; better way to not do
      // this?
      global_training_mask_range_ = {.begin = 0, .end = 153431, .size = 153431};
      global_validation_mask_range_ = {
          .begin = 153431, .end = 153431 + 23831, .size = 23831};
      global_testing_mask_range_ = {
          .begin = 177262, .end = 177262 + 55703, .size = 55703};

      // training
      for (size_t i = global_training_mask_range_.begin;
           i < global_training_mask_range_.end; i++) {
        if (partitioned_graph_->isLocal(i)) {
          local_training_mask_[partitioned_graph_->getLID(i)] = 1;
        }
        global_training_mask_[i] = 1;
      }

      // validation
      for (size_t i = global_validation_mask_range_.begin;
           i < global_validation_mask_range_.end; i++) {
        if (partitioned_graph_->isLocal(i)) {
          local_validation_mask_[partitioned_graph_->getLID(i)] = 1;
        }
      }

      // testing
      for (size_t i = global_testing_mask_range_.begin;
           i < global_testing_mask_range_.end; i++) {
        if (partitioned_graph_->isLocal(i)) {
          local_testing_mask_[partitioned_graph_->getLID(i)] = 1;
        }
      }
    } else if (dataset_name == "ogbn-papers100M-remap") {
      global_training_count_ = 1207178;

      global_training_mask_range_ = {
          .begin = 0, .end = 1207178, .size = 1207178};
      global_validation_mask_range_ = {
          .begin = 1207178, .end = 1207178 + 125264, .size = 125264};
      global_testing_mask_range_ = {
          .begin = 1332442, .end = 1332442 + 214337, .size = 214337};
      // training
      for (size_t i = global_training_mask_range_.begin;
           i < global_training_mask_range_.end; i++) {
        if (partitioned_graph_->isLocal(i)) {
          local_training_mask_[partitioned_graph_->getLID(i)] = 1;
        }
        global_training_mask_[i] = 1;
      }
      // validation
      for (size_t i = global_validation_mask_range_.begin;
           i < global_validation_mask_range_.end; i++) {
        if (partitioned_graph_->isLocal(i)) {
          local_validation_mask_[partitioned_graph_->getLID(i)] = 1;
        }
      }
      // testing
      for (size_t i = global_testing_mask_range_.begin;
           i < global_testing_mask_range_.end; i++) {
        if (partitioned_graph_->isLocal(i)) {
          local_testing_mask_[partitioned_graph_->getLID(i)] = 1;
        }
      }
      valid_other_ = FindOtherMask();
      GALOIS_LOG_ASSERT(valid_other_ <= 109513177);
    } else {
      size_t valid_train = ReadLocalMasksFromFile(dataset_name, "train",
                                                  &global_training_mask_range_,
                                                  &local_training_mask_);
      size_t valid_val   = ReadLocalMasksFromFile(dataset_name, "val",
                                                  &global_validation_mask_range_,
                                                  &local_validation_mask_);
      size_t valid_test  = ReadLocalMasksFromFile(dataset_name, "test",
                                                  &global_testing_mask_range_,
                                                  &local_testing_mask_);
      valid_other_       = FindOtherMask();
      // the "other" set of nodes that don't fall into any classification
      if (galois::runtime::getSystemNetworkInterface().ID == 0) {
        galois::gInfo("Valid # training nodes is ", valid_train);
        galois::gInfo("Valid # validation nodes is ", valid_val);
        galois::gInfo("Valid # test nodes is ", valid_test);
        galois::gInfo("Valid # other nodes is ", valid_other_);
      }
    }
  }

  //! Initializes the norm factors using the entire graph's topology for global
  //! degree access
  void InitNormFactor() {
    GALOIS_LOG_VERBOSE("[{}] Initializing norm factors", host_id_);
    global_degrees_.resize(partitioned_graph_->size(), 0.0);
    global_train_degrees_.resize(partitioned_graph_->size(), 0.0);
    CalculateFullNormFactor();
#ifdef GALOIS_ENABLE_GPU
    if (device_personality == DevicePersonality::GPU_CUDA) {
      gpu_memory_.InitNormFactor(partitioned_graph_->size());
    }
#endif
  }

  //! Used if ranges for a mask are complete (if in range, it's part of mask).
  bool
  IsValidForPhaseCompleteRange(const unsigned lid,
                               const galois::GNNPhase current_phase) const {
    // only use ranges if they're complete
    // convert to gid first
    size_t gid = partitioned_graph_->getGID(lid);

    // select range to use based on phase
    const GNNRange* range_to_use;
    switch (current_phase) {
    case GNNPhase::kTrain:
      range_to_use = &global_training_mask_range_;
      break;
    case GNNPhase::kValidate:
      range_to_use = &global_validation_mask_range_;
      break;
    case GNNPhase::kTest:
      range_to_use = &global_testing_mask_range_;
      break;
    case GNNPhase::kOther:
      GALOIS_LOG_FATAL("no range for other");
      break;
    default:
      GALOIS_LOG_FATAL("Invalid phase used");
      range_to_use = nullptr;
    }

    // if within range, it is valid
    // there is an assumption here that ranges are contiguous; may not
    // necessarily be the case in all inputs in which case using the mask is
    // required (but less cache efficient)
    if (range_to_use->begin <= gid && gid < range_to_use->end) {
      return true;
    } else {
      return false;
    }
  }

  //! Used if ranges for a mask are incomplete, meaning I actually have to
  //! check the mask.
  bool IsValidForPhaseMasked(const unsigned lid,
                             const galois::GNNPhase current_phase) const {
    // select mask to use based on phase
    const GNNMask* mask_to_use;
    switch (current_phase) {
    case GNNPhase::kTrain:
      mask_to_use = &local_training_mask_;
      break;
    case GNNPhase::kValidate:
      mask_to_use = &local_validation_mask_;
      break;
    case GNNPhase::kTest:
      mask_to_use = &local_testing_mask_;
      break;
    case GNNPhase::kOther:
      if (valid_other_ == 0) {
        return false;
      }
      mask_to_use = &other_mask_;
      break;
    case GNNPhase::kBatch:
      mask_to_use = &local_minibatch_mask_;
      break;
    default:
      GALOIS_LOG_FATAL("Invalid phase used");
      mask_to_use = nullptr;
    }
    return (*mask_to_use)[lid];
  }

  //////////////////////////////////////////////////////////////////////////////
  // Accuracy
  //////////////////////////////////////////////////////////////////////////////

  float GetGlobalAccuracyCPU(PointerWithSize<GNNFloat> predictions,
                             GNNPhase phase, bool sampling) {
    galois::StatTimer global_accuracy_timer("GetGlobalAccuracy");
    galois::StatTimer global_accuracy_for_singleclass_timer(
        "GetGlobalAccuracyForSingleClass");
    galois::StatTimer global_accuracy_for_multiclass_timer(
        "GetGlobalAccuracyForMultiClass");
    global_accuracy_timer.start();
    float accuracy{0};
    if (is_single_class_label()) {
      global_accuracy_for_singleclass_timer.start();
      auto accuracy_result =
          GetGlobalAccuracyCPUSingle(predictions, phase, sampling);
      accuracy = accuracy_result.first / accuracy_result.second;
      global_accuracy_for_singleclass_timer.stop();
    } else {
      global_accuracy_for_multiclass_timer.start();
      accuracy = GetGlobalAccuracyCPUMulti(predictions, phase, sampling);
      global_accuracy_for_multiclass_timer.stop();
    }
    global_accuracy_timer.stop();
    return accuracy;
  }

  std::pair<float, float>
  GetGlobalAccuracyCPUSingle(PointerWithSize<GNNFloat> predictions,
                             GNNPhase phase, bool) {
    // check owned nodes' accuracy
    num_correct_.reset();
    total_checked_.reset();

#if 0
    std::cout << "single accuracy print:\n";
    for (int i = *begin_owned(); i < *end_owned(); ++i) {
      if (!IsValidForPhase(i, GNNPhase::kBatch)) {
        continue;
      }
      //std::cout << subgraph_->SIDToLID(i) << ", " << galois::MaxIndex(num_label_classes_, &predictions[i * num_label_classes_]) <<
      std::cout << "accuracy:" << subgraph_->SIDToLID(i) << ", " <<
      predictions[i * num_label_classes_] << ", " <<
      predictions[i * num_label_classes_ + 1] << ", " <<
      predictions[i * num_label_classes_ + 2] << ", " <<
      predictions[i * num_label_classes_ + 3] << ", " <<
      predictions[i * num_label_classes_ + 4] << "-> " <<
      galois::MaxIndex(num_label_classes_, &predictions[i * num_label_classes_]) <<
      " vs " << GetSingleClassLabel(i) << "\n";
    }
#endif
    galois::do_all(
        // will only loop over sampled nodes if sampling is on
        galois::iterate(begin_owned(), end_owned()),
        // this is possibly the subgraph id
        [&](const unsigned node_id) {
          if (IsValidForPhase(node_id, phase)) {
            total_checked_ += 1;
            // get prediction by getting max
            // note the use of node_id here: lid only used to check original
            // labels
            size_t predicted_label =
                galois::MaxIndex(num_label_classes_,
                                 &(predictions[node_id * num_label_classes_]));
            // check against ground truth and track accordingly
            // TODO static cast used here is dangerous
            if (predicted_label ==
                static_cast<size_t>(GetSingleClassLabel(node_id))) {
              num_correct_ += 1;
            }
          }
        },
        // steal on as some threads may have nothing to work on
        galois::steal());

    size_t global_correct = num_correct_.reduce();
    size_t global_checked = total_checked_.reduce();

    GALOIS_LOG_DEBUG("Sub: {}, Accuracy: {} / {}", use_subgraph_,
                     global_correct, global_checked);
    return std::make_pair(static_cast<float>(global_correct),
                          static_cast<float>(global_checked));
  }

  float GetGlobalAccuracyCPUMulti(PointerWithSize<GNNFloat> predictions,
                                  GNNPhase phase, bool sampling) {
    const GNNLabel* full_ground_truth = GetMultiClassLabel(0);
    assert(predictions.size() == (num_label_classes_ * size()));

    size_t global_true_positive  = 0;
    size_t global_true_negative  = 0;
    size_t global_false_positive = 0;
    size_t global_false_negative = 0;
    size_t global_f1_score       = 0;

    // per class check
    for (size_t label_class = 0; label_class < num_label_classes_;
         label_class++) {
      local_true_positive_.reset();
      local_true_negative_.reset();
      local_false_positive_.reset();
      local_false_negative_.reset();

      // loop through all *owned* nodes (do not want to overcount)
      galois::do_all(
          galois::iterate(begin_owned(), end_owned()),
          [&](const unsigned lid) {
            if (IsValidForPhase(lid, phase)) {
              if (sampling) {
                if (phase == GNNPhase::kTrain && !IsInSampledGraph(lid)) {
                  return;
                }
              }

              size_t label_index = lid * num_label_classes_ + label_class;

              GNNLabel true_label = full_ground_truth[label_index];
              GNNLabel prediction_is_positive =
                  (predictions[label_index] > 0.5) ? 1 : 0;

              if (true_label && prediction_is_positive) {
                local_true_positive_ += 1;
              } else if (true_label && !prediction_is_positive) {
                local_false_negative_ += 1;
              } else if (!true_label && prediction_is_positive) {
                local_false_positive_ += 1;
              } else if (!true_label && !prediction_is_positive) {
                local_true_negative_ += 1;
              } else {
                // all cases should be covered with clauses above, so it should
                // NEVER get here; adding it here just for sanity purposes
                GALOIS_LOG_FATAL(
                    "Logic error with true label and prediction label");
              }
            }
            total_checked_ += 1;
          },
          galois::steal(), galois::loopname("GlobalMultiAccuracy"));

      // reduce from accumulators across all hosts for this particular class
      size_t class_true_positives  = local_true_positive_.reduce();
      size_t class_false_positives = local_false_positive_.reduce();
      size_t class_true_negatives  = local_true_negative_.reduce();
      size_t class_false_negatives = local_false_negative_.reduce();

      // add to global counts
      global_true_positive += class_true_positives;
      global_false_positive += class_false_positives;
      global_true_negative += class_true_negatives;
      global_false_negative += class_false_negatives;

      // calculate precision, recall, and f1 score for this class
      // ternery op used to avoid division by 0
      double class_precision =
          (class_true_positives + class_true_negatives) > 0
              ? static_cast<double>(class_true_positives) /
                    (class_true_positives + class_false_positives)
              : 0.0;
      double class_recall =
          (class_true_positives + class_false_negatives) > 0
              ? static_cast<double>(class_true_positives) /
                    (class_true_positives + class_false_negatives)
              : 0.0;
      double class_f1_score = (class_precision + class_recall) > 0
                                  ? (2.0 * (class_precision * class_recall)) /
                                        (class_precision + class_recall)
                                  : 0.0;

      global_f1_score += class_f1_score;
    } // end label class loop

    // GALOIS_LOG_WARN("{} {} {} {}", global_true_positive,
    // global_true_negative, global_false_positive, global_false_negative);

    // double global_f1_macro_score = global_f1_score / num_label_classes_;

    // micro = considers all classes for precision/recall
    double global_micro_precision =
        (global_true_positive + global_true_negative) > 0
            ? static_cast<double>(global_true_positive) /
                  (global_true_positive + global_false_positive)
            : 0.0;
    double global_micro_recall =
        (global_true_positive + global_false_negative) > 0
            ? static_cast<double>(global_true_positive) /
                  (global_true_positive + global_false_negative)
            : 0.0;

    double global_f1_micro_score =
        (global_micro_precision + global_micro_recall) > 0
            ? (2.0 * (global_micro_precision * global_micro_recall)) /
                  (global_micro_precision + global_micro_recall)
            : 0.0;

    return global_f1_micro_score;
  }

  void increment_evilPhase() {
    ++galois::runtime::evilPhase;
    if (galois::runtime::evilPhase >=
        static_cast<uint32_t>(std::numeric_limits<int64_t>::max())) {
      galois::runtime::evilPhase = 1;
    }
  }

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
  std::vector<galois::DynamicBitSet> edge_sample_status_;
  // TODO use a char maybe? unlikely anyone will go over 2^8 layers...
  //! What timestep a node was added to sampled set; used to determine
  //! size of subgraph at each layer
  galois::LargeArray<unsigned> sample_node_timestamps_;
  //! Count of how many masters are in each layer in a sampled subgraph.
  std::vector<unsigned> sample_master_offsets_;
  //! Count of how many mirrors are in each layer in a sampled subgraph.
  std::vector<unsigned> sample_mirror_offsets_;
  //! Definitely sampled nodes
  galois::DynamicBitSet definitely_sampled_nodes_;

  std::vector<galois::GAccumulator<uint32_t>> master_offset_accum_;
  std::vector<galois::GAccumulator<uint32_t>> mirror_offset_accum_;
  //! In a subgraph, all layer 0 masters are made the prefix of SIDs; other
  //! masters that are not layer 0 will be scattered elsewhere. This bitset
  //! tracks which of those SIDs are the masters.
  //! This is required for master masking in certain layers in distributed
  //! execution to avoid recomputation of certain gradients.
  galois::DynamicBitSet non_layer_zero_masters_;

  //! Indicates newly sampled nodes (for distributed synchronization of sampling
  //! status
  galois::DynamicBitSet new_sampled_nodes_;
  //! If edge is sampled at any point, mark this
  galois::DynamicBitSet sampled_edges_;

  //////////////////////////////////////////////////////////////////////////////

  // TODO maybe revisit this and use an actual bitset
  size_t global_training_count_;
  //! Bitset indicating which nodes are training nodes (global)
  GNNMask global_training_mask_;
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

  bool use_subgraph_{false};
  bool use_subgraph_view_{false};
  bool subgraph_choose_all_{false};

  std::unique_ptr<MinibatchGenerator> train_batcher_;
  std::unique_ptr<MinibatchGenerator> test_batcher_;

  std::vector<uint32_t> node_remapping_;

  // True if a WMD graph is being used otherwise false
  bool use_wmd_{false};

  //////////////////////////////////////////////////////////////////////////////
  // GPU things
  //////////////////////////////////////////////////////////////////////////////

#ifdef GALOIS_ENABLE_GPU
  struct CUDA_Context* cuda_ctx_;
  //! Object that holds all GPU allocated pointers to memory related to graphs.
  GNNGraphGPUAllocations gpu_memory_;
  //! Call this to setup GPU memory for this graph: allocates necessary GPU
  //! memory and copies things over
  void InitGPUMemory() {
    // create int casted CSR
    uint64_t* e_index_ptr = partitioned_graph_->row_start_ptr();
    uint32_t* e_dest_ptr  = partitioned_graph_->edge_dst_ptr();

    // + 1 because first element is 0 in BLAS CSRs
    std::vector<int> e_index(partitioned_graph_->size() + 1);
    std::vector<int> e_dest(partitioned_graph_->sizeEdges());

    // set in parallel
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), partitioned_graph_->size() + 1),
        [&](size_t index) {
          if (index != 0) {
            if (e_index_ptr[index - 1] >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
              GALOIS_LOG_FATAL("{} is too big a number for int arrays on GPUs",
                               e_index_ptr[index - 1]);
            }
            e_index[index] = static_cast<int>(e_index_ptr[index - 1]);
          } else {
            e_index[index] = 0;
          }
        },
        galois::loopname("GPUEdgeIndexConstruction"));
    galois::do_all(
        galois::iterate(static_cast<size_t>(0),
                        partitioned_graph_->sizeEdges()),
        [&](size_t edge) {
          if (e_dest_ptr[edge] >
              static_cast<size_t>(std::numeric_limits<int>::max())) {
            GALOIS_LOG_FATAL("{} is too big a number for int arrays on GPUs",
                             e_dest_ptr[edge]);
          }

          e_dest[edge] = static_cast<int>(e_dest_ptr[edge]);
        },
        galois::loopname("GPUEdgeDestConstruction"));

    gpu_memory_.SetGraphTopology(e_index, e_dest);
    e_index.clear();
    e_dest.clear();

    gpu_memory_.SetFeatures(local_node_features_, node_feature_length_);
    gpu_memory_.SetLabels(local_ground_truth_labels_);
    gpu_memory_.SetMasks(local_training_mask_, local_validation_mask_,
                         local_testing_mask_);
    gpu_memory_.AllocAggregateBitset(partitioned_graph_->size());
    gpu_memory_.SetGlobalTrainDegrees(global_train_degrees_);
    gpu_memory_.SetGlobalDegrees(global_degrees_);
  }

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

  std::vector<char> mark_sampled_nodes_;

  bool use_timer_{true};
};

} // namespace graphs
} // namespace galois

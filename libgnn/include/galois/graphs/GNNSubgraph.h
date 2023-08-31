#include "galois/graphs/GNNGraph.h"

#include <limits>

// Note no header guard or anything like that; this file is meant to be
// included in the middle of GNNGraph class declaration as a class in a class
class GNNSubgraph {
public:
  using GraphNode    = typename LC_CSR_CSC_Graph<VTy, ETy>::GraphNode;
  using NodeIterator = boost::counting_iterator<size_t>;
  using EdgeIterator = typename LC_CSR_CSC_Graph<VTy, ETy>::edge_iterator;

  //! Allocates space for the lid to sid map
  GNNSubgraph(size_t main_graph_size) {
    lid_to_subgraph_id_.create(main_graph_size,
                               std::numeric_limits<uint32_t>::max());
    // the subgraph to original graph maps are allocated on demand in gstl
    // vectors since those change every epoch
    subgraph_mirrors_.resize(galois::runtime::getSystemNetworkInterface().Num);
  }
  //! Given sampled bits set on gnn_graph, builds an explicit subgraph
  //! for the sampled bits
  size_t BuildSubgraph(GNNGraph<VTy, ETy>& gnn_graph,
                       size_t num_sampled_layers) {
    galois::StatTimer timer("BuildSubgraph", kRegionName);
    TimerStart(&timer);
    for (auto& vec : subgraph_mirrors_) {
      vec.clear();
    }
    CreateSubgraphMapping(gnn_graph, num_sampled_layers);
    if (num_subgraph_nodes_ == 0) {
      return 0;
    }
    DegreeCounting(gnn_graph);
    EdgeCreation(gnn_graph);
    NodeFeatureCreation(gnn_graph);
    // loop over each node, grab out/in edges, construct them in LC_CSR_CSC
    // no edge data, just topology
    TimerStop(&timer);
    return num_subgraph_nodes_;
  }

  size_t BuildSubgraphView(GNNGraph<VTy, ETy>& gnn_graph,
                           size_t num_sampled_layers) {
    galois::StatTimer timer("BuildSubgraphView", kRegionName);
    TimerStart(&timer);
    CreateSubgraphMapping(gnn_graph, num_sampled_layers);
    NodeFeatureCreation(gnn_graph);
    TimerStop(&timer);
    return num_subgraph_nodes_;
  }

  galois::PODResizeableArray<GNNFeature>& GetLocalFeatures() {
    return subgraph_node_features_;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Nodes
  //////////////////////////////////////////////////////////////////////////////

  uint32_t size() { return num_subgraph_nodes_; }
  NodeIterator begin() const { return NodeIterator(0); }
  NodeIterator end() const { return NodeIterator(num_subgraph_nodes_); }

  NodeIterator begin_owned() const { return NodeIterator(0); }
  NodeIterator end_owned() const {
    return NodeIterator(subgraph_master_boundary_);
  }

  uint32_t SIDToLID(uint32_t sid) const { return subgraph_id_to_lid_[sid]; }
  uint32_t LIDToSID(uint32_t lid) const { return lid_to_subgraph_id_[lid]; }

  //////////////////////////////////////////////////////////////////////////////
  // Edge iteration and destination
  //////////////////////////////////////////////////////////////////////////////

  EdgeIterator edge_begin(GraphNode n) {
    return underlying_graph_.edge_begin(n);
  }
  EdgeIterator edge_end(GraphNode n) { return underlying_graph_.edge_end(n); }
  GraphNode GetEdgeDest(EdgeIterator out_edge_iterator) {
    return underlying_graph_.getEdgeDst(out_edge_iterator);
  };
  galois::runtime::iterable<
      galois::NoDerefIterator<typename GNNDistGraph::edge_iterator>>
  edges(GraphNode n) {
    return internal::make_no_deref_range(edge_begin(n), edge_end(n));
  }

  EdgeIterator in_edge_begin(GraphNode n) {
    return underlying_graph_.in_edge_begin(n);
  }
  EdgeIterator in_edge_end(GraphNode n) {
    return underlying_graph_.in_edge_end(n);
  }
  GraphNode GetInEdgeDest(EdgeIterator in_edge_iterator) {
    return underlying_graph_.getInEdgeDst(in_edge_iterator);
  };
  galois::runtime::iterable<
      galois::NoDerefIterator<typename GNNDistGraph::edge_iterator>>
  in_edges(GraphNode n) {
    return internal::make_no_deref_range(in_edge_begin(n), in_edge_end(n));
  }

  size_t GetLocalDegree(GraphNode n) {
    return std::distance(edge_begin(n), edge_end(n));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Edge sampling status check
  //////////////////////////////////////////////////////////////////////////////

  bool OutEdgeSampled(EdgeIterator out_edge_iterator, size_t layer_num,
                      const GNNGraph<VTy, ETy>& original_graph) {
    return original_graph.IsEdgeSampledOriginalGraph(
        subedge_to_original_edge_[*out_edge_iterator], layer_num);
  }
  bool InEdgeSampled(EdgeIterator in_edge_iterator, size_t layer_num,
                     const GNNGraph<VTy, ETy>& original_graph) {
    // note that original IsEdgeSampled is called because this object stores the
    // original edge already
    return original_graph.IsEdgeSampledOriginalGraph(
        in_subedge_to_original_edge_[*in_edge_iterator], layer_num);
  }

  //////////////////////////////////////////////////////////////////////////////

  galois::LargeArray<uint32_t>* GetLIDToSIDPointer() {
    return &lid_to_subgraph_id_;
  }
  void EnableTimers() { use_timer_ = true; }
  void DisableTimers() { use_timer_ = false; }

  std::vector<std::vector<size_t>>& GetSubgraphMirrors() {
    return subgraph_mirrors_;
  }

private:
  bool use_timer_{true};
  void TimerStart(galois::StatTimer* t) {
    if (use_timer_)
      t->start();
  }
  void TimerStop(galois::StatTimer* t) {
    if (use_timer_)
      t->stop();
  }

  // TODO signature cleanup
  //! Creates subgraph ID mapping from the number of sampled nodes from the
  //! original graph. Should be done every epoch when sampled graph changes.
  void CreateSubgraphMapping(GNNGraph<VTy, ETy>& gnn_graph, size_t) {
    galois::StatTimer timer("SIDMapping", kRegionName);
    TimerStart(&timer);

    assert(gnn_graph.size() == lid_to_subgraph_id_.size());
    // clear all mappings
    galois::ParallelSTL::fill(lid_to_subgraph_id_.begin(),
                              lid_to_subgraph_id_.end(),
                              std::numeric_limits<uint32_t>::max());

    galois::GAccumulator<uint32_t> subgraph_count;
    subgraph_count.reset();
    galois::do_all(galois::iterate(gnn_graph.begin(), gnn_graph.end()),
                   [&](uint32_t node_id) {
                     if (gnn_graph.IsActiveInSubgraph(node_id)) {
                       subgraph_count += 1;
                     }
                   });
    num_subgraph_nodes_ = subgraph_count.reduce();
    // if no subgraph, get out
    if (num_subgraph_nodes_ == 0) {
      subgraph_master_boundary_ = 0;
      TimerStop(&timer);
      return;
    }

    // checking sanity
    // galois::do_all(galois::iterate(gnn_graph.begin(), gnn_graph.end()),
    //               [&](uint32_t node_id) {
    //                 if (gnn_graph.IsInSampledGraph(node_id) &&
    //                 !gnn_graph.IsActiveInSubgraph(node_id)) {
    //                  // check if any edges are active
    //                  for (auto a = gnn_graph.edge_begin(node_id); a !=
    //                  gnn_graph.edge_end(node_id);a++) {
    //                    if (gnn_graph.IsEdgeSampledAny(a)) {
    //                      galois::gWarn("ERROR node ", node_id);
    //                    }
    //                  }
    //                  for (auto a = gnn_graph.in_edge_begin(node_id); a !=
    //                  gnn_graph.in_edge_end(node_id);a++) {
    //                    if (gnn_graph.IsInEdgeSampledAny(a)) {
    //                      galois::gWarn("ERROR in node ", node_id);
    //                    }
    //                  }
    //                 }
    //               });

    if (subgraph_id_to_lid_.size() < num_subgraph_nodes_) {
      // allocate a bit more than necessary to avoid a big realloc
      // if node value changes slightly later
      subgraph_id_to_lid_.resize(num_subgraph_nodes_ * 1.02);
    }

    // bitset to mark if a master is outside the "master only" boundary
    // and not contiguous; needed to mask out non-masters
    galois::DynamicBitSet& non_layer_zero_masters =
        gnn_graph.GetNonLayerZeroMasters();
    // init the bitset as necessary
    if (non_layer_zero_masters.size() < num_subgraph_nodes_) {
      non_layer_zero_masters.resize(num_subgraph_nodes_);
    } else {
      non_layer_zero_masters.ParallelReset();
    }

    std::vector<unsigned>& master_offsets = gnn_graph.GetMasterOffsets();
    std::vector<unsigned>& mirror_offsets = gnn_graph.GetMirrorOffsets();

    ResetSIDThreadOffsets(master_offsets.size());

    // compute offsets for each layer
    galois::PODResizeableArray<unsigned> layer_offsets;
    layer_offsets.resize(master_offsets.size() - 1);
    for (unsigned i = 0; i < layer_offsets.size(); i++) {
      layer_offsets[i] = master_offsets[i] + mirror_offsets[i];
      if (i > 0) {
        // prefix summing
        layer_offsets[i] += layer_offsets[i - 1];
      }
    }

    // all nodes before this SID are master nodes in layer 0;
    // NOTE: there are master nodes past this boundary that will
    // not be covered by a begin_owned loop, which may cause problems down
    // the line; this is handled by the bitset above
    subgraph_master_boundary_ = master_offsets[0];

    size_t last_owned_node = *(gnn_graph.end_owned());
    // compute amount of work each thread needs to do
    galois::on_each([&](size_t thread_id, size_t num_threads) {
      unsigned start_node;
      unsigned end_node;
      // this thread always has a set number of nodes to run; this is it
      std::tie(start_node, end_node) = galois::block_range(
          size_t{0}, gnn_graph.size(), thread_id, num_threads);
      // these arrays track how much work will need to be done by this
      // thread
      galois::PODResizeableArray<unsigned>& my_offsets =
          sid_thread_offsets_[thread_id];
      galois::PODResizeableArray<unsigned>& my_mirror_offsets =
          subgraph_mirror_offsets_[thread_id];

      for (size_t local_node_id = start_node; local_node_id < end_node;
           local_node_id++) {
        // only bother if node was active
        if (gnn_graph.IsActiveInSubgraph(local_node_id)) {
          unsigned node_timestamp =
              gnn_graph.SampleNodeTimestamp(local_node_id);
          // TODO(loc) this check shouldn't even be necessary; active in
          // subgraph implies added at somepoint
          if (node_timestamp != std::numeric_limits<unsigned>::max()) {
            // tracks how many nodes for each timestamp this node will
            // work with by incrementing this
            my_offsets[node_timestamp]++;

            if (local_node_id >= last_owned_node) {
              // this is a mirror node; get the host that the master is located
              // on and increment this thread's mirror node count for that host
              uint32_t node_gid = gnn_graph.GetGID(local_node_id);
              my_mirror_offsets[gnn_graph.GetHostID(node_gid)]++;
            }
          } else {
            GALOIS_LOG_WARN("shouldn't ever get here right?");
          }
        }
      }
    });

    // prefix sum the threads
    galois::do_all(galois::iterate(size_t{0}, master_offsets.size()),
                   [&](size_t layer_num) {
                     for (size_t thread_id = 1;
                          thread_id < galois::getActiveThreads(); thread_id++) {
                       sid_thread_offsets_[thread_id][layer_num] +=
                           sid_thread_offsets_[thread_id - 1][layer_num];
                     }
                   });

    for (unsigned i = 0; i < master_offsets.size() - 1; i++) {
      if (i > 0) {
        GALOIS_LOG_VASSERT(
            sid_thread_offsets_[galois::getActiveThreads() - 1][i] +
                    layer_offsets[i - 1] ==
                (layer_offsets[i]),
            "layer {} wrong {} vs correct {}", i,
            sid_thread_offsets_[galois::getActiveThreads() - 1][i],
            layer_offsets[i]);
      } else {
        GALOIS_LOG_VASSERT(
            sid_thread_offsets_[galois::getActiveThreads() - 1][i] ==
                (layer_offsets[i]),
            "layer {} wrong {} vs correct {}", i,
            sid_thread_offsets_[galois::getActiveThreads() - 1][i],
            layer_offsets[i]);
      }
    }

    // last element of prefix sum needs to equal the correct layer offset
    galois::do_all(
        galois::iterate(uint32_t{0},
                        galois::runtime::getSystemNetworkInterface().Num),
        [&](size_t host_num) {
          // for each host, get prefix sum of each thread's mirrors
          for (size_t thread_id = 1; thread_id < galois::getActiveThreads();
               thread_id++) {
            subgraph_mirror_offsets_[thread_id][host_num] +=
                subgraph_mirror_offsets_[thread_id - 1][host_num];
          }
        });

    // allocate the mirror space; last element of prefix sum is total size
    for (unsigned host_num = 0;
         host_num < galois::runtime::getSystemNetworkInterface().Num;
         host_num++) {
      if (galois::runtime::getSystemNetworkInterface().ID == host_num) {
        continue;
      }
      subgraph_mirrors_[host_num].resize(
          subgraph_mirror_offsets_[galois::getActiveThreads() - 1][host_num]);
    }

    galois::on_each([&](size_t thread_id, size_t num_threads) {
      unsigned start_node;
      unsigned end_node;
      std::tie(start_node, end_node) = galois::block_range(
          size_t{0}, gnn_graph.size(), thread_id, num_threads);

      galois::PODResizeableArray<unsigned>& current_thread_offset =
          thread_id != 0 ? sid_thread_offsets_[thread_id - 1]
                         : thread_zero_work_;
      galois::PODResizeableArray<unsigned>& my_mirror_offsets =
          thread_id != 0 ? subgraph_mirror_offsets_[thread_id - 1]
                         : thread_zero_mirror_offsets_;

      for (size_t local_node_id = start_node; local_node_id < end_node;
           local_node_id++) {
        if (gnn_graph.IsActiveInSubgraph(local_node_id)) {
          unsigned node_timestamp =
              gnn_graph.SampleNodeTimestamp(local_node_id);
          if (node_timestamp != std::numeric_limits<unsigned>::max()) {
            uint32_t sid_to_use;
            if (node_timestamp != 0) {
              sid_to_use = layer_offsets[node_timestamp - 1] +
                           current_thread_offset[node_timestamp]++;
              if (local_node_id < last_owned_node) {
                // master node that is not in layer 0 (i.e. node_timestamp != 0)
                non_layer_zero_masters.set(sid_to_use);
              }
            } else {
              // node timestamp == 0; no layer offset needed because offset
              // is 0
              sid_to_use = current_thread_offset[node_timestamp]++;
            }

            // this is a mirror
            if (local_node_id >= last_owned_node) {
              // XXX(loc) mirror offsets
              uint32_t node_gid = gnn_graph.GetGID(local_node_id);
              size_t my_offset =
                  my_mirror_offsets[gnn_graph.GetHostID(node_gid)]++;

              if (my_offset >
                  subgraph_mirrors_[gnn_graph.GetHostID(node_gid)].size())
                GALOIS_LOG_FATAL(
                    "{} {}", my_offset,
                    subgraph_mirrors_[gnn_graph.GetHostID(node_gid)].size());

              subgraph_mirrors_[gnn_graph.GetHostID(node_gid)][my_offset] =
                  node_gid;
            }

            subgraph_id_to_lid_[sid_to_use]    = local_node_id;
            lid_to_subgraph_id_[local_node_id] = sid_to_use;
          } else {
            GALOIS_LOG_WARN("shouldn't ever get here right?");
          }
        }
      }
    });

    TimerStop(&timer);
  }

  //! reset sid thread offsets used for parallel SID mapping creation
  void ResetSIDThreadOffsets(size_t num_layers) {
    if (!sid_thread_offsets_.size()) {
      sid_thread_offsets_.resize(galois::getActiveThreads());
      galois::on_each([&](size_t thread_id, size_t) {
        sid_thread_offsets_[thread_id].resize(num_layers);
      });
    }

    if (!subgraph_mirror_offsets_.size()) {
      subgraph_mirror_offsets_.resize(galois::getActiveThreads());
      galois::on_each([&](size_t thread_id, size_t) {
        subgraph_mirror_offsets_[thread_id].resize(
            galois::runtime::getSystemNetworkInterface().Num);
      });
    }

    galois::do_all(
        galois::iterate(size_t{0}, sid_thread_offsets_.size()), [&](size_t i) {
          galois::PODResizeableArray<unsigned>& arr = sid_thread_offsets_[i];
          std::fill(arr.begin(), arr.end(), 0);
          galois::PODResizeableArray<unsigned>& arr2 =
              subgraph_mirror_offsets_[i];
          std::fill(arr2.begin(), arr2.end(), 0);
        });

    if (thread_zero_work_.size() < num_layers) {
      thread_zero_work_.resize(num_layers);
    }
    if (thread_zero_mirror_offsets_.size() <
        galois::runtime::getSystemNetworkInterface().Num) {
      thread_zero_mirror_offsets_.resize(
          galois::runtime::getSystemNetworkInterface().Num);
    }
    galois::ParallelSTL::fill(thread_zero_work_.begin(),
                              thread_zero_work_.end(), 0);
    galois::ParallelSTL::fill(thread_zero_mirror_offsets_.begin(),
                              thread_zero_mirror_offsets_.end(), 0);
  }

  //! Counts in and out degrees of all sampled nodes in the graph
  void DegreeCounting(const GNNGraph<VTy, ETy>& gnn_graph) {
    galois::StatTimer timer("DegreeCounting", kRegionName);
    TimerStart(&timer);

    if (local_subgraph_out_degrees_.size() < num_subgraph_nodes_) {
      local_subgraph_out_degrees_.resize(num_subgraph_nodes_ * 1.02);
    }

    if (local_subgraph_in_degrees_.size() < num_subgraph_nodes_) {
      local_subgraph_in_degrees_.resize(num_subgraph_nodes_ * 1.02);
    }

    galois::do_all(
        galois::iterate(begin(), end()),
        [&](uint32_t subgraph_id) {
          uint32_t node_id     = subgraph_id_to_lid_[subgraph_id];
          uint32_t out_degrees = 0;
          for (auto out_edge_iter : gnn_graph.edges(node_id)) {
            if (gnn_graph.IsEdgeSampledAny(out_edge_iter)) {
              out_degrees++;
            }
          }
          local_subgraph_out_degrees_[subgraph_id] = out_degrees;

          uint32_t in_degrees = 0;
          for (auto in_edge_iter : gnn_graph.in_edges(node_id)) {
            if (gnn_graph.IsInEdgeSampledAny(in_edge_iter)) {
              in_degrees++;
            }
          }
          local_subgraph_in_degrees_[subgraph_id] = in_degrees;
        },
        galois::loopname("DegreeCountingDoAll"), galois::steal());

    TimerStop(&timer);
  }

  //! Creates edges
  void EdgeCreation(const GNNGraph<VTy, ETy>& gnn_graph) {
    galois::StatTimer timer("EdgeConstruction", kRegionName);
    TimerStart(&timer);
    // galois::DGAccumulator<uint32_t> empty_masters;
    // galois::DGAccumulator<uint32_t> empty_mirrors;
    // empty_masters.reset();
    // empty_mirrors.reset();

    // galois::DGAccumulator<uint32_t> total_sn;
    // total_sn.reset();
    // total_sn += num_subgraph_nodes_;
    // size_t global_sub_size = total_sn.reduce();

    // prefix sum over subgraph degrees from previous phase to get starting
    // points
    for (size_t i = 1; i < num_subgraph_nodes_; i++) {
      // if (local_subgraph_out_degrees_[i] == 0 &&
      //    local_subgraph_in_degrees_[i] == 0) {
      //  if (i < subgraph_master_boundary_) {
      //    empty_masters += 1;
      //  } else {
      //    if (gnn_graph.GetNonLayerZeroMasters().test(i)) {
      //      empty_masters += 1;
      //    } else {
      //      empty_mirrors += 1;
      //    }
      //  }
      //}
      local_subgraph_out_degrees_[i] += local_subgraph_out_degrees_[i - 1];
      local_subgraph_in_degrees_[i] += local_subgraph_in_degrees_[i - 1];
    }

    // uint32_t emaster = empty_masters.reduce();
    // uint32_t emirror = empty_mirrors.reduce();
    // if (gnn_graph.host_id() == 0) {
    //  galois::gInfo("Empty masters percent is ", emaster /
    //  (float)global_sub_size,
    //                " ", emaster, " ", global_sub_size);
    //  galois::gInfo("Empty mirrors percent is ", emirror /
    //  (float)global_sub_size,
    //                " ", emirror, " ", global_sub_size);
    //}

    // allocate then set node endpoints
    num_subgraph_edges_ = local_subgraph_out_degrees_[num_subgraph_nodes_ - 1];

    galois::StatTimer alloc_time("EdgeCreationAlloc", kRegionName);
    TimerStart(&alloc_time);
    underlying_graph_.DeallocateOnly();
    underlying_graph_.allocateFrom(num_subgraph_nodes_, num_subgraph_edges_);
    underlying_graph_.CSCAllocate();
    TimerStop(&alloc_time);

    galois::gInfo(gnn_graph.host_prefix(), "Subgraph nodes and edges are ",
                  num_subgraph_nodes_, " ", num_subgraph_edges_);

    galois::do_all(galois::iterate(uint32_t{0}, num_subgraph_nodes_),
                   [&](uint32_t subgraph_id) {
                     underlying_graph_.fixEndEdge(
                         subgraph_id, local_subgraph_out_degrees_[subgraph_id]);
                     underlying_graph_.FixEndInEdge(
                         subgraph_id, local_subgraph_in_degrees_[subgraph_id]);
                   });
    if (subedge_to_original_edge_.size() < num_subgraph_edges_) {
      subedge_to_original_edge_.resize(num_subgraph_edges_ * 1.02);
    }
    if (in_subedge_to_original_edge_.size() < num_subgraph_edges_) {
      in_subedge_to_original_edge_.resize(num_subgraph_edges_ * 1.02);
    }

    // save edges + save reference to layer sample status
    galois::do_all(
        galois::iterate(begin(), end()),
        [&](uint32_t subgraph_id) {
          uint32_t node_id = subgraph_id_to_lid_[subgraph_id];
          assert(subgraph_id != std::numeric_limits<uint32_t>::max());
          uint32_t out_location = 0;
          uint32_t in_location  = 0;
          if (subgraph_id != 0) {
            out_location = local_subgraph_out_degrees_[subgraph_id - 1];
            in_location  = local_subgraph_in_degrees_[subgraph_id - 1];
          }

          for (auto out_edge_iter : gnn_graph.edges(node_id)) {
            if (gnn_graph.IsEdgeSampledAny(out_edge_iter)) {
              assert(
                  lid_to_subgraph_id_[gnn_graph.GetEdgeDest(out_edge_iter)] !=
                  std::numeric_limits<uint32_t>::max());
              subedge_to_original_edge_[out_location] = *out_edge_iter;

              underlying_graph_.constructEdge(
                  out_location++,
                  lid_to_subgraph_id_[gnn_graph.GetEdgeDest(out_edge_iter)]);
            }
          }

          for (auto in_edge_iter : gnn_graph.in_edges(node_id)) {
            if (gnn_graph.IsInEdgeSampledAny(in_edge_iter)) {
              in_subedge_to_original_edge_[in_location] =
                  *(gnn_graph.InEdgeToOutEdge(in_edge_iter));
              underlying_graph_.ConstructInEdge(
                  in_location++,
                  lid_to_subgraph_id_[gnn_graph.GetInEdgeDest(in_edge_iter)]);
            }
          }
          assert(out_location == local_subgraph_out_degrees_[subgraph_id]);
          assert(in_location == local_subgraph_in_degrees_[subgraph_id]);
        },
        galois::loopname("EdgeCreationDoAll"), galois::steal());
    TimerStop(&timer);
  }

  //! Copies over relevant features of the nodes
  void NodeFeatureCreation(GNNGraph<VTy, ETy>& gnn_graph) {
    galois::StatTimer timer("NodeFeatureCreation", kRegionName);
    TimerStart(&timer);
    size_t feat_length = gnn_graph.node_feature_length();
    subgraph_node_features_.resize(feat_length * num_subgraph_nodes_);

    galois::do_all(
        galois::iterate(begin(), end()), [&](size_t subgraph_node_id) {
          size_t local_id = subgraph_id_to_lid_[subgraph_node_id];
          std::memcpy(
              &(subgraph_node_features_[subgraph_node_id * feat_length]),
              &((gnn_graph.GetLocalFeatures().data())[local_id * feat_length]),
              feat_length * sizeof(GNNFeature));
        });
    TimerStop(&timer);
  }

  static const constexpr char* kRegionName = "GNNSubgraph";

  bool inductive_subgraph_{false};

  // name is self explanatory
  LC_CSR_CSC_Graph<char, void> underlying_graph_;
  // size vars
  uint32_t num_subgraph_nodes_;
  uint32_t num_subgraph_edges_;
  uint32_t subgraph_master_boundary_;
  //! Features corresponding only to this subgraph; copied from main graph
  //! (in other words, redundant; would be nice if there was a way to
  //! fake contiguous memory
  galois::PODResizeableArray<GNNFeature> subgraph_node_features_;
  //! Dense array mapping local ids to subgraph id (not space efficient)
  galois::LargeArray<uint32_t> lid_to_subgraph_id_;
  //! Map subgraph ids back to local graph ids
  //! gstl vector because this will get resized every epoch (LargeArray
  //! is for static)
  galois::PODResizeableArray<uint32_t> subgraph_id_to_lid_;
  // intermediate degrees used for edge construction
  galois::PODResizeableArray<uint32_t> local_subgraph_out_degrees_;
  galois::PODResizeableArray<uint32_t> local_subgraph_in_degrees_;
  //! Maps from subgraph out-edge id to original graph edge id (used to check if
  //! edge exists in particular layer)
  galois::PODResizeableArray<uint32_t> subedge_to_original_edge_;
  //! Maps from subgraph in-edge id to original graph edge id (used to check if
  //! edge exists in particular layer)
  galois::PODResizeableArray<uint32_t> in_subedge_to_original_edge_;

  //! Mirror mappings for Gluon for subgraph
  // std::vector<std::vector<size_t>> subgraph_mirrors_;
  std::vector<std::vector<size_t>> subgraph_mirrors_;

  //! Offsets to use for
  std::vector<galois::PODResizeableArray<unsigned>> sid_thread_offsets_;
  std::vector<galois::PODResizeableArray<unsigned>> subgraph_mirror_offsets_;
  galois::PODResizeableArray<unsigned> thread_zero_work_;
  galois::PODResizeableArray<unsigned> thread_zero_mirror_offsets_;
};

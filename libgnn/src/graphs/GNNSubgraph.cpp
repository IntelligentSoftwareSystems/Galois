#include "galois/graphs/GNNGraph.h"
#include <limits>

size_t galois::graphs::GNNGraph::GNNSubgraph::BuildSubgraph(
    GNNGraph& gnn_graph, size_t num_sampled_layers) {
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

size_t galois::graphs::GNNGraph::GNNSubgraph::BuildSubgraphView(
    GNNGraph& gnn_graph, size_t num_sampled_layers) {
  galois::StatTimer timer("BuildSubgraphView", kRegionName);
  TimerStart(&timer);
  CreateSubgraphMapping(gnn_graph, num_sampled_layers);
  NodeFeatureCreation(gnn_graph);
  TimerStop(&timer);
  return num_subgraph_nodes_;
}

// TODO signature cleanup
void galois::graphs::GNNGraph::GNNSubgraph::CreateSubgraphMapping(
    GNNGraph& gnn_graph, size_t) {
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
        unsigned node_timestamp = gnn_graph.SampleNodeTimestamp(local_node_id);
        // TODO(loc) this check shouldn't even be necessary; active in subgraph
        // implies added at somepoint
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
        thread_id != 0 ? sid_thread_offsets_[thread_id - 1] : thread_zero_work_;
    galois::PODResizeableArray<unsigned>& my_mirror_offsets =
        thread_id != 0 ? subgraph_mirror_offsets_[thread_id - 1]
                       : thread_zero_mirror_offsets_;

    for (size_t local_node_id = start_node; local_node_id < end_node;
         local_node_id++) {
      if (gnn_graph.IsActiveInSubgraph(local_node_id)) {
        unsigned node_timestamp = gnn_graph.SampleNodeTimestamp(local_node_id);
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

// TODO optimize further?
void galois::graphs::GNNGraph::GNNSubgraph::DegreeCounting(
    const GNNGraph& gnn_graph) {
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

// TODO optimize further?
void galois::graphs::GNNGraph::GNNSubgraph::EdgeCreation(
    const GNNGraph& gnn_graph) {
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

  // prefix sum over subgraph degrees from previous phase to get starting points
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
            assert(lid_to_subgraph_id_[gnn_graph.GetEdgeDest(out_edge_iter)] !=
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

void galois::graphs::GNNGraph::GNNSubgraph::NodeFeatureCreation(
    GNNGraph& gnn_graph) {
  galois::StatTimer timer("NodeFeatureCreation", kRegionName);
  TimerStart(&timer);
  size_t feat_length = gnn_graph.node_feature_length();
  subgraph_node_features_.resize(feat_length * num_subgraph_nodes_);

  galois::do_all(galois::iterate(begin(), end()), [&](size_t subgraph_node_id) {
    size_t local_id = subgraph_id_to_lid_[subgraph_node_id];
    std::memcpy(
        &(subgraph_node_features_[subgraph_node_id * feat_length]),
        &((gnn_graph.GetLocalFeatures().data())[local_id * feat_length]),
        feat_length * sizeof(GNNFeature));
  });
  TimerStop(&timer);
}

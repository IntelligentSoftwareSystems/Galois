#include "galois/graphs/GNNGraph.h"
#include <limits>

size_t galois::graphs::GNNGraph::GNNSubgraph::BuildSubgraph(
    GNNGraph& gnn_graph, size_t num_sampled_layers) {
  galois::StatTimer timer("BuildSubgraph", kRegionName);
  TimerStart(&timer);
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
  std::fill(lid_to_subgraph_id_.begin(), lid_to_subgraph_id_.end(),
            std::numeric_limits<uint32_t>::max());

  galois::GAccumulator<uint32_t> subgraph_count;
  subgraph_count.reset();
  galois::do_all(galois::iterate(gnn_graph.begin(), gnn_graph.end()),
                 [&](uint32_t node_id) {
                   if (gnn_graph.IsInSampledGraph(node_id)) {
                     subgraph_count += 1;
                   }
                 });
  num_subgraph_nodes_ = subgraph_count.reduce();
  if (subgraph_id_to_lid_.size() < num_subgraph_nodes_) {
    subgraph_id_to_lid_.resize(num_subgraph_nodes_ * 1.02);
  }

  galois::DynamicBitSet& non_layer_zero_masters =
      gnn_graph.GetNonLayerZeroMasters();
  std::vector<unsigned>& master_offsets = gnn_graph.GetMasterOffsets();
  std::vector<unsigned>& mirror_offsets = gnn_graph.GetMirrorOffsets();

  // init the bitset as necessary
  if (non_layer_zero_masters.size() < num_subgraph_nodes_) {
    non_layer_zero_masters.resize(num_subgraph_nodes_);
  } else {
    non_layer_zero_masters.reset();
  }

  // compute offsets for each layer
  uint32_t layer_zero_offset = 0;
  galois::PODResizeableArray<unsigned> layer_offsets;
  layer_offsets.resize(master_offsets.size() - 1);
  for (unsigned i = 0; i < layer_offsets.size(); i++) {
    layer_offsets[i] = master_offsets[i] + mirror_offsets[i];
    if (i > 0) {
      // prefix summing
      layer_offsets[i] += layer_offsets[i - 1];
    }
  }

  // split into 2 parts: masters, then everything else
  size_t last_owned_node = *(gnn_graph.end_owned());
  galois::gInfo(last_owned_node);
  for (size_t local_node_id = 0; local_node_id < last_owned_node;
       local_node_id++) {
    unsigned node_timestamp = gnn_graph.SampleNodeTimestamp(local_node_id);
    if (node_timestamp != std::numeric_limits<unsigned>::max()) {
      uint32_t sid_to_use;
      if (node_timestamp != 0) {
        sid_to_use = layer_offsets[node_timestamp - 1]++;
        // master that won't be in prefix needs to be marked
        non_layer_zero_masters.set(sid_to_use);
      } else {
        sid_to_use = layer_zero_offset++;
      }
      subgraph_id_to_lid_[sid_to_use]    = local_node_id;
      lid_to_subgraph_id_[local_node_id] = sid_to_use++;
    }
  }

  // all nodes before this SID are master nodes in layer 0;
  // NOTE: there are master nodes past this boundary that will
  // not be covered by a begin_owned loop, which may cause problems down
  // the line
  subgraph_master_boundary_ = master_offsets[0];

  // everything else; none of these are master nodes
  for (size_t local_node_id = last_owned_node; local_node_id < gnn_graph.size();
       local_node_id++) {
    unsigned node_timestamp = gnn_graph.SampleNodeTimestamp(local_node_id);
    if (node_timestamp != std::numeric_limits<unsigned>::max()) {
      uint32_t sid_to_use;
      if (node_timestamp != 0) {
        sid_to_use = layer_offsets[node_timestamp - 1]++;
      } else {
        sid_to_use = layer_zero_offset++;
      }
      subgraph_id_to_lid_[sid_to_use]    = local_node_id;
      lid_to_subgraph_id_[local_node_id] = sid_to_use++;
    }
  }

  GALOIS_LOG_ASSERT(layer_offsets.back() == num_subgraph_nodes_);
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

  // prefix sum over subgraph degrees from previous phase to get starting points
  for (size_t i = 1; i < num_subgraph_nodes_; i++) {
    local_subgraph_out_degrees_[i] += local_subgraph_out_degrees_[i - 1];
    local_subgraph_in_degrees_[i] += local_subgraph_in_degrees_[i - 1];
  }

  // allocate then set node endpoints
  num_subgraph_edges_ = local_subgraph_out_degrees_[num_subgraph_nodes_ - 1];

  galois::StatTimer alloc_time("EdgeCreationAlloc", kRegionName);
  TimerStart(&alloc_time);
  underlying_graph_.DeallocateOnly();
  underlying_graph_.allocateFrom(num_subgraph_nodes_, num_subgraph_edges_);
  underlying_graph_.CSCAllocate();
  TimerStop(&alloc_time);

  galois::gInfo("subgraph nodes and edges are ", num_subgraph_nodes_, " ", num_subgraph_edges_);

  galois::DGAccumulator<uint32_t> empty_masters;
  galois::DGAccumulator<uint32_t> empty_mirrors;
  empty_masters.reset();
  empty_mirrors.reset();

  galois::do_all(galois::iterate(uint32_t{0}, num_subgraph_nodes_),
                 [&](uint32_t subgraph_id) {
                   if (local_subgraph_out_degrees_[subgraph_id] == 0 &&
                       local_subgraph_in_degrees_[subgraph_id] == 0) {
                     if (subgraph_id < subgraph_master_boundary_) {
                       empty_masters += 1;
                     } else {
                       if (gnn_graph.GetNonLayerZeroMasters().test(subgraph_id)) {
                         empty_masters += 1;
                       } else {
                         empty_mirrors += 1;
                       }
                     }
                   }
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
  uint32_t emaster = empty_masters.reduce();
  uint32_t emirror = empty_mirrors.reduce();
  galois::gInfo("empty masters percent is ", emaster / (float)num_subgraph_nodes_, " ", emaster);
  galois::gInfo("empty mirrors percent is ", emirror / (float)num_subgraph_nodes_, " ", emirror);

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

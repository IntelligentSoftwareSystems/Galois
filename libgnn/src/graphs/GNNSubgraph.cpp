#include "galois/graphs/GNNGraph.h"
#include <limits>

size_t
galois::graphs::GNNGraph::GNNSubgraph::BuildSubgraph(GNNGraph& gnn_graph) {
  galois::StatTimer timer("BuildSubgraph", kRegionName);
  timer.start();
  CreateLocalToSubgraphMapping(gnn_graph);
  DegreeCounting(gnn_graph);
  EdgeCreation(gnn_graph);
  NodeFeatureCreation(gnn_graph);
  // loop over each node, grab out/in edges, construct them in LC_CSR_CSC
  // no edge data, just topology
  timer.stop();
  return num_subgraph_nodes_;
}

void galois::graphs::GNNGraph::GNNSubgraph::CreateLocalToSubgraphMapping(
    const GNNGraph& gnn_graph) {
  galois::StatTimer timer("LIDToSIDMapping", kRegionName);
  timer.start();

  assert(gnn_graph.size() == lid_to_subgraph_id_.size());
  // clear all mappings
  std::fill(lid_to_subgraph_id_.begin(), lid_to_subgraph_id_.end(),
            std::numeric_limits<uint32_t>::max());
  // TODO(loc) depending on overhead, can parallelize this with a prefix sum
  // serial loop over LIDs to construct lid -> subgraph id mapping
  uint32_t current_sid = 0;

  // split into 2 parts: masters, then mirrors
  size_t last_owned_node = *(gnn_graph.end_owned());
  for (size_t local_node_id = 0; local_node_id < last_owned_node;
       local_node_id++) {
    if (gnn_graph.IsInSampledGraph(local_node_id)) {
      // TODO should bound check the SID to max uint32_t
      // note: if SID is max uint32t, then it's not valid
      lid_to_subgraph_id_[local_node_id] = current_sid++;
    }
  }

  // all nodes before this SID are master nodes
  subgraph_master_boundary_ = current_sid;

  for (size_t local_node_id = last_owned_node; local_node_id < gnn_graph.size();
       local_node_id++) {
    if (gnn_graph.IsInSampledGraph(local_node_id)) {
      // TODO should bound check the SID to max uint32_t
      // note: if SID is max uint32t, then it's not valid
      lid_to_subgraph_id_[local_node_id] = current_sid++;
    }
  }
  galois::gDebug("Numbered sampled nodes for subgraph construction is ",
                 current_sid);

  num_subgraph_nodes_ = current_sid;

  timer.stop();
}

// TODO optimize further?
void galois::graphs::GNNGraph::GNNSubgraph::DegreeCounting(
    const GNNGraph& gnn_graph) {
  galois::StatTimer timer("DegreeCounting", kRegionName);
  timer.start();

  subgraph_id_to_lid_.resize(num_subgraph_nodes_);
  subgraph_out_degrees_.resize(num_subgraph_nodes_);
  subgraph_in_degrees_.resize(num_subgraph_nodes_);

  galois::do_all(
      galois::iterate(gnn_graph.begin(), gnn_graph.end()),
      [&](uint32_t node_id) {
        if (gnn_graph.IsInSampledGraph(node_id)) {
          uint32_t subgraph_id             = lid_to_subgraph_id_[node_id];
          subgraph_id_to_lid_[subgraph_id] = node_id;

          uint32_t out_degrees = 0;
          for (auto out_edge_iter : gnn_graph.edges(node_id)) {
            if (gnn_graph.IsEdgeSampledAny(out_edge_iter)) {
              out_degrees++;
            }
          }
          subgraph_out_degrees_[subgraph_id] = out_degrees;

          uint32_t in_degrees = 0;
          for (auto in_edge_iter : gnn_graph.in_edges(node_id)) {
            if (gnn_graph.IsInEdgeSampledAny(in_edge_iter)) {
              in_degrees++;
            }
          }
          subgraph_in_degrees_[subgraph_id] = in_degrees;
          // galois::gDebug("Local ID ", node_id, " SID ", subgraph_id, " out ",
          //               out_degrees, " in ", in_degrees);
        }
      },
      galois::steal());

  timer.stop();
}

// TODO optimize further?
void galois::graphs::GNNGraph::GNNSubgraph::EdgeCreation(
    const GNNGraph& gnn_graph) {
  galois::StatTimer timer("EdgeConstruction", kRegionName);
  timer.start();

  // prefix sum over subgraph degrees from previous phase to get starting points
  for (size_t i = 1; i < num_subgraph_nodes_; i++) {
    subgraph_out_degrees_[i] += subgraph_out_degrees_[i - 1];
    subgraph_in_degrees_[i] += subgraph_in_degrees_[i - 1];
  }

  // allocate then set node endpoints
  num_subgraph_edges_ = subgraph_out_degrees_.back();
  underlying_graph_.DeallocateOnly();
  underlying_graph_.allocateFrom(num_subgraph_nodes_, num_subgraph_edges_);
  underlying_graph_.CSCAllocate();
  galois::do_all(galois::iterate(uint32_t{0}, num_subgraph_nodes_),
                 [&](uint32_t subgraph_id) {
                   underlying_graph_.fixEndEdge(
                       subgraph_id, subgraph_out_degrees_[subgraph_id]);
                   underlying_graph_.FixEndInEdge(
                       subgraph_id, subgraph_in_degrees_[subgraph_id]);
                 });
  subedge_to_original_edge_.resize(num_subgraph_edges_);
  in_subedge_to_original_edge_.resize(num_subgraph_edges_);

  // save edges + save reference to layer sample status
  galois::do_all(
      galois::iterate(gnn_graph.begin(), gnn_graph.end()),
      [&](uint32_t node_id) {
        if (gnn_graph.IsInSampledGraph(node_id)) {
          uint32_t subgraph_id = lid_to_subgraph_id_[node_id];

          uint32_t out_location = 0;
          uint32_t in_location  = 0;
          if (subgraph_id != 0) {
            out_location = subgraph_out_degrees_[subgraph_id - 1];
            in_location  = subgraph_in_degrees_[subgraph_id - 1];
          }

          for (auto out_edge_iter : gnn_graph.edges(node_id)) {
            if (gnn_graph.IsEdgeSampledAny(out_edge_iter)) {
              subedge_to_original_edge_[out_location] = *out_edge_iter;
              underlying_graph_.constructEdge(
                  out_location++, gnn_graph.GetEdgeDest(out_edge_iter));
            }
          }

          for (auto in_edge_iter : gnn_graph.in_edges(node_id)) {
            if (gnn_graph.IsInEdgeSampledAny(in_edge_iter)) {
              in_subedge_to_original_edge_[in_location] =
                  *(gnn_graph.InEdgeToOutEdge(in_edge_iter));
              underlying_graph_.ConstructInEdge(
                  in_location++, gnn_graph.GetInEdgeDest(in_edge_iter));
            }
          }
          assert(out_location == subgraph_out_degrees_[subgraph_id]);
          assert(in_location == subgraph_in_degrees_[subgraph_id]);
        }
      },
      galois::steal());
  timer.stop();
}

void galois::graphs::GNNGraph::GNNSubgraph::NodeFeatureCreation(
    GNNGraph& gnn_graph) {
  galois::StatTimer timer("NodeFeatureCreation", kRegionName);
  timer.start();
  size_t feat_length = gnn_graph.node_feature_length();
  // assumes everything is already setup
  subgraph_node_features_.resize(feat_length * num_subgraph_nodes_);

  galois::do_all(galois::iterate(begin(), end()), [&](size_t subgraph_node_id) {
    size_t local_id = subgraph_id_to_lid_[subgraph_node_id];
    std::memcpy(
        &(subgraph_node_features_[subgraph_node_id * feat_length]),
        &((gnn_graph.GetLocalFeatures().data())[local_id * feat_length]),
        feat_length * sizeof(GNNFeature));
    // for (unsigned i = 0; i < feat_length; i++) {
    //  galois::gPrint(feat_length * sizeof(GNNFeature) , " ", subgraph_node_id,
    //  " local id " , local_id, " feat at ", i, " is ",
    //  subgraph_node_features_[subgraph_node_id * feat_length + i], " ",
    //  gnn_graph.GetLocalFeatures()[local_id * feat_length + i], "\n");
    //}
  });
  timer.stop();
}

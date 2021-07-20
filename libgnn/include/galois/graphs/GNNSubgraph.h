// Note no header guard or anything like that; this file is meant to be
// included in the middle of GNNGraph class declaration as a class in a class
class GNNSubgraph {
public:
  using GraphNode    = LC_CSR_CSC_Graph<char, void>::GraphNode;
  using NodeIterator = boost::counting_iterator<size_t>;
  using EdgeIterator = LC_CSR_CSC_Graph<char, void>::edge_iterator;

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
  size_t BuildSubgraph(GNNGraph& gnn_graph, size_t num_sampled_layers);

  size_t BuildSubgraphView(GNNGraph& gnn_graph, size_t num_sampled_layers);

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
      galois::NoDerefIterator<GNNDistGraph::edge_iterator>>
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
      galois::NoDerefIterator<GNNDistGraph::edge_iterator>>
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
                      const GNNGraph& original_graph) {
    return original_graph.IsEdgeSampledOriginalGraph(
        subedge_to_original_edge_[*out_edge_iterator], layer_num);
  }
  bool InEdgeSampled(EdgeIterator in_edge_iterator, size_t layer_num,
                     const GNNGraph& original_graph) {
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
  void CreateSubgraphMapping(GNNGraph& gnn_graph, size_t);

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
  void DegreeCounting(const GNNGraph& gnn_graph);
  //! Creates edges
  void EdgeCreation(const GNNGraph& gnn_graph);
  //! Copies over relevant features of the nodes
  void NodeFeatureCreation(GNNGraph& gnn_graph);

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

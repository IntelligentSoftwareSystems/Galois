template<typename Graph>
std::vector<typename Graph::GraphNode> getDegSortedNodes(Graph& g) {
  std::vector<typename Graph::GraphNode> nodes;
  nodes.reserve(std::distance(g.begin(), g.end()));
  std::copy(g.begin(), g.end(), std::back_insert_iterator<decltype(nodes)>(nodes));
  galois::ParallelSTL::sort(nodes.begin(), nodes.end(), 
                            [&g](const typename Graph::GraphNode& lhs, 
                                 const typename  Graph::GraphNode& rhs) {
                              return std::distance(g.edge_begin(lhs), g.edge_end(lhs))
                                > std::distance(g.edge_begin(rhs), g.edge_end(rhs));
                            });
  return nodes;
}

template<typename Graph>
std::vector<typename Graph::GraphNode> getPRSortedNodes(Graph& g) {
  std::vector<typename Graph::GraphNode> nodes;
  nodes.reserve(std::distance(g.begin(), g.end()));
  std::copy(g.begin(), g.end(), std::back_insert_iterator<decltype(nodes)>(nodes));
  galois::ParallelSTL::sort(nodes.begin(), nodes.end(), 
                            [&g](const typename Graph::GraphNode& lhs, 
                                 const typename  Graph::GraphNode& rhs) {
                              return g.getData(lhs).getPageRank() > g.getData(rhs).getPageRank();
                            });
  return nodes;
}


struct PPRAsyncRsd {
  static constexpr double alpha = 0.99;
  static constexpr double tolerance = 0.00001;

  struct LNode {
    float value;
    float residual; // tracking residual
    bool flag; // tracking if it is in the worklist
    bool seedset;
    void init() { value = 0.0; flag = true; seedset = false; residual = 0.0; }
    float getPageRank(int x = 0) const { return value; }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "PPRAsyncRsd"; }

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
    galois::graphs::readGraph(graph, filename); 
  }

  struct Process2 {
    Graph& graph;
     
    Process2(Graph& g): graph(g) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      LNode& sdata = graph.getData(src);
      if (sdata.residual < tolerance){
        sdata.flag = false;
        return;
      }

      // the node is processed
      sdata.flag = false;
      double sum = 0;
      for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sum += ddata.value / std::distance(graph.edge_begin(dst), graph.edge_end(dst));
      }
      float value = alpha * sum + (sdata.seedset ? (1.0 - alpha) : 0.0);
      float diff = std::fabs(value - sdata.value);

      if (diff >= tolerance) {
        sdata.value = value;
        //std::cout << src << " " << sdata.value << "\n";
        // for each out-going neighbors
        for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
	  LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
	  ddata.residual += sdata.residual*alpha/std::distance(graph.edge_begin(src), graph.edge_end(src)); // update residual
	  // if the node is not in the worklist and the residual is greater than tolerance
	  if(!ddata.flag && ddata.residual>=tolerance) {
	    ddata.flag = true;
            ctx.push(dst);
	  } 
        }
        sdata.residual = 0.0; // update residual
      } // enf of if
      
    } // end of operator
  };

  double computeResidual(Graph& graph,  GNode src) {
    double retval = 0;
    for (auto jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj){
      GNode dst = graph.getEdgeDst(jj);
      retval += 1.0/std::distance(graph.edge_begin(dst), graph.edge_end(dst));
    }
    return retval * alpha * (1.0 - alpha);
  }

  std::vector<GNode> cluster_from_sweep(Graph& g) {
    // sparserow* G, sparsevec& p, 
    // std::vector<mwIndex>& cluster, double *outcond, double* outvolume,
    // double *outcut)
    // now we have to do the sweep over p in sorted order by value
    auto nodes = getPRSortedNodes(g);
    struct ldata {
      unsigned volume;
      int cutsize;
    };
    
    std::vector<unsigned> rank(nodes.size());
    std::vector<ldata> data(nodes.size());

    size_t i = 0;
    for(auto n : nodes)
      rank[n] = i++;
    
    unsigned total_degree = 0;

    i = 0;
    for (auto n : nodes) {
      int deg = std::distance(g.edge_begin(n), g.edge_end(n));
      total_degree += deg;
      int change = deg;
      for (auto ni = g.edge_begin(n), ne = g.edge_end(n); ni != ne; ++ni) {
        if (rank[g.getEdgeDst(ni)] < rank[n])
          change -= 2;
      }
      //assert(change > 0 || -change < curcutsize);
      data[i].cutsize = change;
      data[i].volume = deg;
      ++i;
    }
    std::partial_sum(data.begin(), data.end(), data.begin(), [] (const ldata& lhs, const ldata& rhs) -> ldata { return {lhs.volume + rhs.volume, lhs.cutsize + rhs.cutsize}; });
    std::vector<double> conductance(data.size());
    std::transform(data.begin(), data.end(), conductance.begin(),
                   [total_degree] (const ldata& d) -> double {
                     if (d.volume == 0 || total_degree == d.volume)
                       return  1.0;
                     else 
                       return double(d.cutsize)/double(std::min(d.volume, total_degree-d.volume));
                   });

    // for (int i = 0; i < 10; ++i )
    //   std::cout << conductance[i] << " ";
    // std::cout << "\n";

    auto m = std::min_element(conductance.begin(), conductance.end());
    assert(m != conductance.end());
    double mincond = *m;
    //printf("mincond=%f mincondind=%i\n", mincond, mincondind);
    size_t num = std::distance(conductance.begin(), m);
    std::vector<GNode> cluster;
    for (i = 0; i <= num; ++i)
      cluster.push_back(nodes[i]);
    return cluster;
  }



  void operator()(Graph& graph, GNode seed) {
    for (auto ii = graph.edge_begin(seed), ee = graph.edge_end(seed);
         ii != ee; ++ii) {
      auto n = graph.getEdgeDst(ii);
      auto& d = graph.getData(n);
      d.residual = computeResidual(graph,n);
      d.seedset = true;
      d.value = 1 - alpha;
    }
    graph.getData(seed).residual = computeResidual(graph, seed);
    graph.getData(seed).seedset = true;
    graph.getData(seed).value = 1 - alpha;

    typedef galois::WorkList::dChunkedFIFO<16> WL;
    galois::for_each_local(graph, Process2(graph), galois::wl<WL>());
  }
};


//! Find k seeds, in degree order which are at least on hop
template<typename Graph>
std::vector<typename Graph::GraphNode> findPPRSeeds(Graph& g, unsigned k) {
  std::vector<typename Graph::GraphNode> nodes = getDegSortedNodes(g);
  std::set<typename Graph::GraphNode> marks;
  std::vector<typename Graph::GraphNode> retval;
  auto nodeI = nodes.begin();
  while (k) {
    --k;
    while (marks.count(*nodeI))
      ++nodeI;
    auto n = *nodeI++;
    retval.push_back(n);
    marks.insert(n);
    for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii)
      marks.insert(g.getEdgeDst(ii));
  }
  return retval;
}

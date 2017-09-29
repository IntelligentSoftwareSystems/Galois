typedef unsigned int Dist;
static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max() - 1;


struct UpdateRequest {
  GNode n;
  Dist w;
  UpdateRequest(const GNode& N, Dist W): n(N), w(W) {}
  UpdateRequest(): n(), w(0) {}
};



struct UpdateRequestIndexer {
  unsigned int operator()(const UpdateRequest& val) const {
    unsigned int t = val.w >> stepShift;
    return t;
  }
};

template<bool useOne>
struct not_consistent {
  Graph& g;
  std::atomic<bool>& refb;
  not_consistent(Graph& g, std::atomic<bool>& refb) : g(g), refb(refb) {}

  template<bool useOneL, typename iiTy>
  Dist getEdgeWeight(iiTy ii, typename std::enable_if<useOneL>::type* = nullptr) const {
    return 1;
  }

  template<bool useOneL, typename iiTy>
  Dist getEdgeWeight(iiTy ii, typename std::enable_if<!useOneL>::type* = nullptr) const {
    return g.getEdgeData(ii);
  }

  void operator()(typename Graph::GraphNode n) const {
    Dist dist = g.getData(n);
    if (dist == DIST_INFINITY)
      return;
    
    for (auto ii : g.edges(n)) {
      Dist ddist = g.getData(g.getEdgeDst(ii));
      Dist w = getEdgeWeight<useOne>(ii);
      if (ddist > dist + w) {
        std::cout << ddist << " " << dist + w << " " << n << " " << g.getEdgeDst(ii) << "\n"; // XXX
        refb = true;
        // return;
      }
    }
  }
};

struct max_dist {
  Graph& g;
  galois::GReduceMax<Dist>& m;

  max_dist(Graph& g, galois::GReduceMax<Dist>& m) : g(g), m(m) {}

  void operator()(typename Graph::GraphNode n) const {
    Dist d = g.getData(n);
    if (d == DIST_INFINITY)
      return;
    m.update(d);
  }
};

template<bool useOne>
bool verify(Graph& graph, GNode source) {
  if (graph.getData(source) != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }

  std::atomic<size_t> notVisited(0);
  galois::do_all(graph, [&notVisited, &graph] (GNode n) { if (graph.getData(n) >= DIST_INFINITY) ++notVisited; });
  if (notVisited)
    std::cerr << notVisited << " unvisited nodes; this is an error if the graph is strongly connected\n";

  std::atomic<bool> not_c;
  galois::do_all(graph, not_consistent<useOne>(graph, not_c));
  if (not_c) {
    std::cerr << "node found with incorrect distance\n";
    return false;
  }

  galois::GReduceMax<Dist> m;
  galois::do_all(graph, max_dist(graph, m));
  std::cout << "max dist: " << m.reduce() << "\n";
  
  return true;
}

#include <iostream>
#include <cstdlib>

template <typename Graph, typename _DistLabel, ptrdiff_t EDGE_TILE_SIZE=256> 
struct BFS_SSSP {

  using Dist = _DistLabel;

  constexpr static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max()/2 - 1;

  using GNode = typename Graph::GraphNode;
  using EI = typename Graph::edge_iterator;

  struct UpdateRequest {
    GNode n;
    Dist w;
    UpdateRequest(const GNode& N, Dist W): n(N), w(W) {}
    UpdateRequest(): n(), w(0) {}
  };

  struct UpdateRequestIndexer {
    unsigned shift;

    unsigned int operator()(const UpdateRequest& val) const {
      unsigned int t = val.w >> shift;
      return t;
    }
  };

  struct DistEdgeTile {
    Dist dist;
    EI beg;
    EI end;
  };

  struct DistEdgeTileMaker {
    Dist dist;

    template <typename EI>
    DistEdgeTile operator () (const EI& beg, const EI& end) const {
      return DistEdgeTile {dist, beg, end};
    }
  };

  template <typename WL, typename TileMaker>
  static void pushEdgeTiles(WL& wl, EI beg, const EI end , const TileMaker& f) {
    assert(beg <= end);

    if ((end - beg) > EDGE_TILE_SIZE) {
      for (; beg + EDGE_TILE_SIZE < end;) {
        auto ne = beg + EDGE_TILE_SIZE;
        assert(ne < end);
        wl.push_back( f(beg, ne) );
        beg = ne;
      }
    }
    
    if ((end - beg) > 0) {
      wl.push_back( f(beg, end) );
    }
  }

  template <typename WL, typename TileMaker>
  static void pushEdgeTiles(WL& wl, Graph& graph, GNode src, const TileMaker& f) {
    auto beg = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
    const auto end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);

    pushEdgeTiles(wl, beg, end, f);

  }

  template <typename WL, typename TileMaker>
  static void pushEdgeTilesParallel(WL& wl, Graph& graph, GNode src , const TileMaker& f) {

    auto beg = graph.edge_begin(src);
    const auto end = graph.edge_end(src);

    if ((end - beg) > EDGE_TILE_SIZE) {

      galois::on_each(
          [&] (const unsigned tid, const unsigned numT) {

            auto p = galois::block_range(beg, end, tid, numT);

            auto b = p.first;
            const auto e = p.second;

            pushEdgeTiles(wl, b, e, f);
          }, galois::loopname("Init-Tiling"));


    } else if ((end - beg) > 0) {
      wl.push_back( f(beg, end) );
    }
  }

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
        auto dst = g.getEdgeDst(ii);
        Dist ddist = g.getData(dst);
        Dist w = getEdgeWeight<useOne>(ii);
        if (ddist > dist + w) {
          std::cout << "Wrong label: " <<  ddist << ", on node: " << dst << ", correct label (from pred): " << dist + w << "\n"; // XXX
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
  static bool verify(Graph& graph, GNode source) {
    if (graph.getData(source) != 0) {
      std::cerr << "source has non-zero dist value\n";
      return false;
    }

    std::atomic<size_t> notVisited(0);
    galois::do_all(galois::iterate(graph), 
        [&notVisited, &graph] (GNode n) { 
          if (graph.getData(n) >= DIST_INFINITY) 
            ++notVisited; 
          });

    if (notVisited)
      std::cerr << notVisited << " unvisited nodes; this is an error if the graph is strongly connected\n";

    std::atomic<bool> not_c;
    galois::do_all(galois::iterate(graph), 
        not_consistent<useOne>(graph, not_c));

    if (not_c) {
      std::cerr << "node found with incorrect distance\n";
      return false;
    }

    galois::GReduceMax<Dist> m;
    galois::do_all(galois::iterate(graph), 
        max_dist(graph, m));

    std::cout << "max dist: " << m.reduce() << "\n";
    
    return true;
  }
};

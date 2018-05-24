#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <random>
#include <deque>
#include <string>
#include <limits>
#include <iostream>

#include "HybridBFS.h"
#ifdef GALOIS_USE_EXP
#include "LigraAlgo.h"
#include "GraphLabAlgo.h"
#endif
#include "BFS.h"

static const char* name = "Diameter Estimation";
static const char* desc = "Estimates the diameter of a graph";
static const char* url = 0;

//****** Command Line Options ******
enum Algo {
  graphlab,
  ligra,
  ligraChi,
  pickK,
  simple
};

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));
static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<unsigned int> numCandidates("numCandidates", cll::desc("Number of candidates to use for pickK algorithm"), cll::init(5));
cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::simple, "simple", "Simple pseudo-peripheral algorithm (default)"),
      clEnumValN(Algo::pickK, "pickK", "Pick K candidates"),
#ifdef USE_EXP
      clEnumValN(Algo::ligra, "ligra", "Use Ligra programming model"),
      clEnumValN(Algo::ligraChi, "ligraChi", "Use Ligra and GraphChi programming model"),
      clEnumValN(Algo::graphlab, "graphlab", "Use GraphLab programming model"),
#endif
      clEnumValEnd), cll::init(Algo::simple));

template<typename Graph>
struct min_degree {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;
  min_degree(Graph& g): graph(g) { }

  galois::optional<GNode> operator()(const galois::optional<GNode>& a, const galois::optional<GNode>& b) const {
    if (!a) return b;
    if (!b) return a;
    if (std::distance(graph.edge_begin(*a), graph.edge_end(*a))
        < std::distance(graph.edge_begin(*b), graph.edge_end(*b)))
      return a;
    else
      return b;
  }
};

template<typename Graph>
struct order_by_degree {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;
  order_by_degree(Graph& g): graph(g) { }

  bool operator()(const GNode& a, const GNode& b) const {
    return std::distance(graph.edge_begin(a), graph.edge_end(a))
        < std::distance(graph.edge_begin(b), graph.edge_end(b));
  }
};

//! Collect nodes with dist == d
template<typename Graph>
struct collect_nodes_with_dist {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;
  galois::InsertBag<GNode>& bag;
  Dist dist;
  collect_nodes_with_dist(Graph& g, galois::InsertBag<GNode>& b, Dist d): graph(g), bag(b), dist(d) { }

  void operator()(const GNode& n) const {
    if (graph.getData(n).dist == dist)
      bag.push(n);
  }
};

template<typename Graph>
struct has_dist {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;
  Dist dist;
  has_dist(Graph& g, Dist d): graph(g), dist(d) { }
  galois::optional<GNode> operator()(const GNode& a) const {
    if (graph.getData(a).dist == dist)
      return galois::optional<GNode>(a);
    return galois::optional<GNode>();
  }
};

template<typename Graph>
struct CountLevels {

  Graph& graph;
  //! [Define GReducible]
  CountLevels(Graph& g): graph(g) { }
  
  //! [Use GReducible in parallel]
  
  const gstl::Vector<size_t>& count() {
    galois::GVectorPerItemReduce<size_t, std::plus<size_t> > reducer;

    galois::do_all(galois::iterate(graph),
        [&] (typename Graph::GraphNode n) const {
          Dist d = graph.getData(n).dist;
          if (d == DIST_INFINITY)
            return;
          reducer.update(d, 1);
        });

    return reducer.reduce();
  }
};

template<typename Algo>
void resetGraph(typename Algo::Graph& g) {
  galois::do_all(g, typename Algo::Initialize(g));
}

template<typename Graph>
void readInOutGraph(Graph& graph) {
  using namespace galois::graphs;
  if (symmetricGraph) {
    galois::graphs::readGraph(graph, filename);
  } else if (transposeGraphName.size()) {
    galois::graphs::readGraph(graph, filename, transposeGraphName);
  } else {
    GALOIS_DIE("Graph type not supported");
  }
}

/**
 * The eccentricity of vertex v, ecc(v), is the greatest distance from v to any vertex.
 * A peripheral vertex v is one whose distance from some other vertex u is the
 * diameter of the graph: \exists u : dist(v, u) = D. A pseudo-peripheral vertex is a 
 * vertex v that satisfies: \forall u : dist(v, u) = ecc(v) ==> ecc(v) = ecc(u).
 *
 * Simple pseudo-peripheral algorithm:
 *  1. Choose v
 *  2. Among the vertices dist(v, u) = ecc(v), select u with minimal degree
 *  3. If ecc(u) > ecc(v) then
 *       v = u and go to step 2
 *     otherwise
 *       u is a pseudo-peripheral vertex
 */
struct SimpleAlgo {
  typedef HybridBFS<SNode,Dist> BFS;
  typedef BFS::Graph Graph;
  typedef Graph::GraphNode GNode;
  typedef std::pair<size_t,GNode> Result;

  void readGraph(Graph& graph) { readInOutGraph(graph); }

  struct Initialize {
    Graph& graph;
    Initialize(Graph& g): graph(g) { }
    void operator()(GNode n) const {
      graph.getData(n).dist = DIST_INFINITY;
    }
  };

  Result search(Graph& graph, GNode start) {
    BFS bfs;

    bfs(graph, start);
    CountLevels<Graph> cl(graph);
    const auto& counts = cl.count();

    size_t ecc = counts.size() - 1;
    //size_t maxWidth = *std::max_element(counts.begin(), counts.end());
    GNode candidate = *galois::ParallelSTL::map_reduce(graph.begin(), graph.end(),
        has_dist<Graph>(graph, ecc), galois::optional<GNode>(), min_degree<Graph>(graph));
    resetGraph<SimpleAlgo>(graph);
    return Result(ecc, candidate);
  }

  size_t operator()(Graph& graph, GNode source) {
    Result v = search(graph, source);
    while (true) {
      Result u = search(graph, v.second);
      std::cout << "ecc(v) = " << v.first << " ecc(u) = " << u.first << "\n";
      bool better = u.first > v.first;
      if (!better)
        break;
      v = u;
    }
    return v.first;
  }
};

/**
 * A more complicated pseudo-peripheral algorithm. Designed for finding pairs
 * of nodes with small maximum width between them, which is useful for matrix
 * reordering. Include it here for completeness.
 *
 * Let the width of vertex v be the maximum number of nodes with the same
 * distance from v.
 *
 * Unlike the simple one, instead of picking a minimal degree candidate u,
 * select among some number of candidates U. Here, we select the top n
 * lowest degree nodes who do not share neighborhoods.
 *
 * If there exists a vertex u such that ecc(u) > ecc(v) proceed as in the
 * simple algorithm. 
 *
 * Otherwise, select the u that has least maximum width.
 */
struct PickKAlgo {
  struct LNode: public SNode {
    bool done;
  };

  typedef HybridBFS<LNode,Dist> BFS;
  typedef BFS::Graph Graph;
  typedef Graph::GraphNode GNode;

  void readGraph(Graph& graph) { readInOutGraph(graph); }

  struct Initialize {
    Graph& graph;
    Initialize(Graph& g): graph(g) { }
    void operator()(GNode n) const {
      graph.getData(n).dist = DIST_INFINITY;
      graph.getData(n).done = false;
    }
  };
  
  std::deque<GNode> select(Graph& graph, unsigned topn, size_t dist) {
    galois::InsertBag<GNode> bag;
    galois::do_all(graph, collect_nodes_with_dist<Graph>(graph, bag, dist));

    // Incrementally sort nodes until we find least N who are not neighbors
    // of each other
    std::deque<GNode> nodes;
    std::deque<GNode> result;
    std::copy(bag.begin(), bag.end(), std::back_inserter(nodes));
    size_t cur = 0;
    size_t size = nodes.size();
    size_t delta = topn * 5;

    for (std::deque<GNode>::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ) {
      std::deque<GNode>::iterator mi = ii;
      if (cur + delta < size) {
        std::advance(mi, delta);
        cur += delta;
      } else {
        mi = ei;
        cur = size;
      }

      std::partial_sort(ii, mi, ei, order_by_degree<Graph>(graph));

      for (std::deque<GNode>::iterator jj = ii; jj != mi; ++jj) {
        GNode n = *jj;

        // Ignore marked neighbors
        if (graph.getData(n).done)
          continue;

        result.push_back(n);
        
        if (result.size() == topn) {
          return result;
        }

        // Mark neighbors
        for (auto nn : graph.edges(n)) 
          graph.getData(graph.getEdgeDst(nn)).done = true;
      }

      ii = mi;
    }

    return result;
  }

  struct Result {
    GNode source;
    std::deque<GNode> candidates;
    size_t maxWidth;
    size_t ecc;
  };

  Result search(Graph& graph, const GNode& start, size_t limit, bool computeCandidates) {
    BFS bfs;
    Result res;

    bfs(graph, start);
    CountLevels<Graph> cl(graph);
    const auto& counts = cl.count();

    res.source = start;
    res.ecc = counts.size() - 1;
    res.maxWidth = *std::max_element(counts.begin(), counts.end());

    if (limit == static_cast<size_t>(-1) || res.maxWidth < limit) {
      if (computeCandidates)
        res.candidates = select(graph, numCandidates, res.ecc);
    }

    resetGraph<PickKAlgo>(graph);
    return res;
  }

  size_t operator()(Graph& graph, GNode source) {
    galois::optional<size_t> terminal;

    Result v = search(graph, source, ~0, true);

    while (true) {
      std::cout 
        << "(ecc(v), max_width) =" 
        << " (" << v.ecc << ", " << v.maxWidth << ")"
        << " (ecc(u), max_width(u)) =";

      size_t last = ~0;
      for (auto ii = v.candidates.begin(), ei = v.candidates.end(); ii != ei; ++ii) {
        Result u = search(graph, *ii, last, false);

        std::cout << " (" << u.ecc << ", " << u.maxWidth << ")";

        if (u.maxWidth >= last) {
          continue;
        } else if (u.ecc > v.ecc) {
          v = u;
          terminal = galois::optional<size_t>();
          break;
        } else if (u.maxWidth < last) {
          last = u.maxWidth;
          terminal = galois::optional<size_t>(u.ecc);
        }
      }

      std::cout << "\n";

      if (terminal)
        break;
      v = search(graph, v.source, ~0, true);
    }

    return *terminal;
  }
};

template<typename Algo>
void initialize(Algo& algo,
    typename Algo::Graph& graph,
    typename Algo::Graph::GraphNode& source) {

  algo.readGraph(graph);
  std::cout << "Read " << graph.size() << " nodes\n";

  if (startNode >= graph.size()) {
    std::cerr 
      << "failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
  
  typename Algo::Graph::iterator it = graph.begin();
  std::advance(it, startNode);
  source = *it;
}


template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Algo algo;
  Graph graph;
  GNode source;

  initialize(algo, graph, source);

  //galois::preAlloc((numThreads + (graph.size() * sizeof(SNode) * 2) / galois::runtime::MM::hugePageSize)*8);
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  T.start();
  resetGraph<Algo>(graph);
  size_t diameter = algo(graph, source);
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");

  std::cout << "Estimated diameter: " << diameter << "\n";
}

int main(int argc, char **argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::simple: run<SimpleAlgo>(); break;
    case Algo::pickK: run<PickKAlgo>(); break;
#ifdef GALOIS_USE_EXP
    case Algo::ligra: run<LigraDiameter<false> >(); break;
    case Algo::ligraChi: run<LigraDiameter<true> >(); break;
    case Algo::graphlab: run<GraphLabDiameter<true> >(); break;
#endif
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}

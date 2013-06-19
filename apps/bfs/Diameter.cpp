/** Computing/Estimating diameter of a graph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Algorithms for estimating the diameter (longest shortest path) of a graph.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/config.h"
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#ifdef GALOIS_USE_EXP
#include <boost/mpl/if.hpp>
#include "Galois/Graph/OCGraph.h"
#include "Galois/Graph/GraphNodeBag.h"
#include "Galois/DomainSpecificExecutors.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "HybridBFS.h"

#include GALOIS_CXX11_STD_HEADER(random)
#include <deque>
#include <string>
#include <limits>
#include <iostream>

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
static cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::simple, "simple", "Simple pseudo-peripheral algorithm (default)"),
      clEnumValN(Algo::pickK, "pickK", "Pick K candidates"),
      clEnumValN(Algo::ligra, "ligra", "Use Ligra programming model"),
      clEnumValN(Algo::ligraChi, "ligraChi", "Use Ligra and GraphChi programming model"),
      clEnumValN(Algo::graphlab, "graphlab", "Use GraphLab programming model"),
      clEnumValEnd), cll::init(Algo::simple));

typedef unsigned int Dist;
static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max() - 1;

//! Standard data type on nodes
struct SNode {
  Dist dist;
};

template<typename Graph>
struct min_degree {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;
  min_degree(Graph& g): graph(g) { }

  Galois::optional<GNode> operator()(const Galois::optional<GNode>& a, const Galois::optional<GNode>& b) const {
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
  Galois::InsertBag<GNode>& bag;
  Dist dist;
  collect_nodes_with_dist(Graph& g, Galois::InsertBag<GNode>& b, Dist d): graph(g), bag(b), dist(d) { }

  void operator()(const GNode& n) {
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
  Galois::optional<GNode> operator()(const GNode& a) const {
    if (graph.getData(a).dist == dist)
      return Galois::optional<GNode>(a);
    return Galois::optional<GNode>();
  }
};

template<typename Graph>
struct CountLevels {
  Graph& graph;
  std::deque<size_t> counts;
  
  CountLevels(Graph& g): graph(g) { }

  void operator()(typename Graph::GraphNode n) {
    Dist d = graph.getData(n).dist;
    if (d == DIST_INFINITY)
      return;
    if (counts.size() <= d)
      counts.resize(d + 1);
    ++counts[d];
  }

  // Reduce function
  template<typename G>
  void operator()(CountLevels<G>& a, CountLevels<G>& b) {
    if (a.counts.size() < b.counts.size())
      a.counts.resize(b.counts.size());
    std::transform(b.counts.begin(), b.counts.end(), a.counts.begin(), a.counts.begin(), std::plus<size_t>());
  }

  std::deque<size_t> count() {
    return Galois::Runtime::do_all_impl(Galois::Runtime::makeLocalRange(graph), *this, *this).counts;
  }
};

template<typename Algo>
void resetGraph(typename Algo::Graph& g) {
  Galois::do_all_local(g, typename Algo::Initialize(g));
}

template<typename Graph>
void readInOutGraph(Graph& graph) {
  using namespace Galois::Graph;
  if (symmetricGraph) {
    Galois::Graph::readGraph(graph, filename);
  } else if (transposeGraphName.size()) {
    Galois::Graph::readGraph(graph, filename, transposeGraphName);
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
    void operator()(GNode n) {
      graph.getData(n).dist = DIST_INFINITY;
    }
  };

  Result search(Graph& graph, GNode start) {
    BFS bfs;

    bfs(graph, start);
    CountLevels<Graph> cl(graph);
    std::deque<size_t> counts = cl.count();

    size_t ecc = counts.size() - 1;
    //size_t maxWidth = *std::max_element(counts.begin(), counts.end());
    GNode candidate = *Galois::ParallelSTL::map_reduce(graph.begin(), graph.end(),
        has_dist<Graph>(graph, ecc), Galois::optional<GNode>(), min_degree<Graph>(graph));
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
    void operator()(GNode n) {
      graph.getData(n).dist = DIST_INFINITY;
      graph.getData(n).done = false;
    }
  };
  
  std::deque<GNode> select(Graph& graph, unsigned topn, size_t dist) {
    Galois::InsertBag<GNode> bag;
    Galois::do_all_local(graph, collect_nodes_with_dist<Graph>(graph, bag, dist));

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
        for (Graph::edge_iterator nn = graph.edge_begin(n), en = graph.edge_end(n); nn != en; ++nn)
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
    std::deque<size_t> counts = cl.count();

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
    Galois::optional<size_t> terminal;

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
          terminal = Galois::optional<size_t>();
          break;
        } else if (u.maxWidth < last) {
          last = u.maxWidth;
          terminal = Galois::optional<size_t>(u.ecc);
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

#ifdef GALOIS_USE_EXP
static void bitwise_or(std::vector<std::vector<bool> >& v1, const std::vector<std::vector<bool> >& v2) {
  while (v1.size() < v2.size())
    v1.emplace_back();

  for (size_t a = 0; a < v1.size(); ++a) {
    while (v1[a].size() < v2[a].size()) {
      v1[a].push_back(false);
    }
    for (size_t i = 0; i < v2[a].size(); ++i) {
      v1[a][i] = v1[a][i] || v2[a][i];
    }
  }
}

template<bool UseHashed>
struct GraphLabAlgo {
  struct LNode {
    std::vector<std::vector<bool> > bitmask1;
    std::vector<std::vector<bool> > bitmask2;
    bool odd_iteration;

    LNode(): odd_iteration(false) { }
  };

  typedef typename Galois::Graph::LC_CSR_Graph<LNode,void>
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;

  void readGraph(Graph& graph) { readInOutGraph(graph); }

  struct Initialize {
    Graph& graph;
    Galois::optional<std::mt19937> gen;
#if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_REAL_DISTRIBUTION)
    std::uniform_real_distribution<float> dist;
#else
    std::uniform_real<float> dist;
#endif

    Initialize(Graph& g): graph(g) { }

    size_t hash_value() {
      if (!gen) {
        gen = std::mt19937();
        gen->seed(Galois::Runtime::LL::getTID());
      }
      size_t ret = 0;
      while (dist(*gen) < 0.5) {
        ret++;
      }
      return ret;
    }

    void initHashed(LNode& data) {
      for (size_t i = 0; i < 10; ++i) {
        size_t hash_val = hash_value();

        std::vector<bool> mask1(hash_val + 2, 0);
        mask1[hash_val] = 1;
        data.bitmask1.push_back(mask1);
        std::vector<bool> mask2(hash_val + 2, 0);
        mask2[hash_val] = 1;
        data.bitmask2.push_back(mask2);
      }
    }

    void initExact(LNode& data, size_t id) {
      std::vector<bool> mask1(id + 2, 0);
      mask1[id] = 1;
      data.bitmask1.push_back(mask1);
      std::vector<bool> mask2(id + 2, 0);
      mask2[id] = 1;
      data.bitmask2.push_back(mask2);
    }

    void operator()(GNode n) {
      LNode& data = graph.getData(n, Galois::MethodFlag::NONE);
      if (UseHashed)
        initHashed(data);
      else
        initExact(data, n);
    }
  };

  struct Program {
    struct gather_type {
      std::vector<std::vector<bool> > bitmask;
      gather_type() { }
      explicit gather_type(const std::vector<std::vector<bool> > & in_b) {
        for(size_t i=0;i<in_b.size();++i){
          bitmask.push_back(in_b[i]);
        }
      }

      gather_type& operator+=(const gather_type& other) {
        bitwise_or(bitmask, other.bitmask);
        return *this;
      }
    };
    typedef Galois::GraphLab::EmptyMessage message_type;

    typedef std::pair<GNode,message_type> WorkItem;
    typedef int tt_needs_gather_out_edges;
    
    void gather(Graph& graph, GNode node, GNode src, GNode dst, gather_type& gather, typename Graph::edge_data_reference) {
      LNode& sdata = graph.getData(node, Galois::MethodFlag::NONE);
      LNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
      if (sdata.odd_iteration) {
        bitwise_or(gather.bitmask, ddata.bitmask2);
        //gather += gather_type(ddata.bitmask2);
      } else {
        bitwise_or(gather.bitmask, ddata.bitmask1);
        //gather += gather_type(ddata.bitmask1);
      }
    }

    void apply(Graph& graph, GNode node, const gather_type& total) {
      LNode& data = graph.getData(node, Galois::MethodFlag::NONE);
      if (data.odd_iteration) {
        if (total.bitmask.size() > 0)
          bitwise_or(data.bitmask1, total.bitmask);
        data.odd_iteration = false;
      } else {
        if (total.bitmask.size() > 0)
          bitwise_or(data.bitmask2, total.bitmask);
        data.odd_iteration = true;
      }
    }

    void init(Graph& graph, GNode node, const message_type& msg) { }
    bool needsScatter(Graph& graph, GNode node) { return false; }
    void scatter(Graph& graph, GNode node, GNode src, GNode dst,
        Galois::GraphLab::Context<Graph,Program>& ctx, typename Graph::edge_data_reference) { }
  };

  struct count_exact_visited {
    Graph& graph;
    count_exact_visited(Graph& g): graph(g) { }
    size_t operator()(GNode n) {
      LNode& data = graph.getData(n);
      size_t count = 0;
      for (size_t i = 0; i < data.bitmask1[0].size(); ++i)
        if (data.bitmask1[0][i])
          count++;
      return count;
    }
  };

  struct count_hashed_visited {
    Graph& graph;
    count_hashed_visited(Graph& g): graph(g) { }

    size_t approximate_pair_number(const std::vector<std::vector<bool> >& bitmask) {
      float sum = 0.0;
      for (size_t a = 0; a < bitmask.size(); ++a) {
        for (size_t i = 0; i < bitmask[a].size(); ++i) {
          if (bitmask[a][i] == 0) {
            sum += (float) i;
            break;
          }
        }
      }
      return (size_t) (pow(2.0, sum / (float) (bitmask.size())) / 0.77351);
    }

    size_t operator()(GNode n) {
      LNode& data = graph.getData(n);
      return approximate_pair_number(data.bitmask1);
    }
  };

  size_t operator()(Graph& graph, const GNode& source) {
    size_t previous_count = 0;
    size_t diameter = 0;
    for (size_t iter = 0; iter < 100; ++iter) {
      //Galois::GraphLab::executeSync(graph, graph, Program());
      Galois::GraphLab::SyncEngine<Graph,Program> engine(graph, Program());
      engine.execute();

      Galois::do_all(graph.begin(), graph.end(), [&](GNode n) {
        LNode& data = graph.getData(n);
        if (data.odd_iteration == false) {
          data.bitmask2 = data.bitmask1;
        } else {
          data.bitmask1 = data.bitmask2;
        }
      });

      size_t current_count;
      if (UseHashed)
        current_count = Galois::ParallelSTL::map_reduce(graph.begin(), graph.end(),
            count_hashed_visited(graph), (size_t) 0, std::plus<size_t>());
      else
        current_count = Galois::ParallelSTL::map_reduce(graph.begin(), graph.end(),
            count_exact_visited(graph), (size_t) 0, std::plus<size_t>());

      std::cout << iter + 1 << "-th hop: " << current_count << " vertex pairs are reached\n";
      if (iter > 0 && (float) current_count < (float) previous_count * (1.0 + 0.0001)) {
        diameter = iter;
        std::cout << "Converged.\n";
        break;
      }
      previous_count = current_count;
    }

    return diameter;
  }
};

template<bool UseGraphChi>
struct LigraAlgo: public Galois::LigraGraphChi::ChooseExecutor<UseGraphChi>  {
  typedef int Visited;

  struct LNode:  public SNode {
    Visited visited[2];
  };

  typedef typename Galois::Graph::LC_CSR_Graph<LNode,void>
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type InnerGraph;
  typedef typename boost::mpl::if_c<UseGraphChi,
          Galois::Graph::OCImmutableEdgeGraph<LNode,void>,
          Galois::Graph::LC_InOut_Graph<InnerGraph> >::type
          Graph;
  typedef typename Graph::GraphNode GNode;

  void readGraph(Graph& graph) {
    readInOutGraph(graph); 
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct Initialize {
    Graph& graph;
    Initialize(Graph& g): graph(g) { }
    void operator()(GNode n) {
      LNode& data = graph.getData(n, Galois::MethodFlag::NONE);
      data.dist = DIST_INFINITY;
      data.visited[0] = data.visited[1] = 0;
    }
  };

  struct EdgeOperator {
    LigraAlgo* self;
    int cur;
    int next;
    Dist newDist;
    EdgeOperator(LigraAlgo* s, int c, int n, Dist d): self(s), cur(c), next(n), newDist(d) { }

    template<typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode) { return true; }

    template<typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src, typename GTy::GraphNode dst, typename GTy::edge_data_reference) {
      LNode& sdata = graph.getData(src, Galois::MethodFlag::NONE);
      LNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
      Visited toWrite = sdata.visited[cur] | ddata.visited[cur];

      if (toWrite != ddata.visited[cur]) {
        while (true) {
          Visited old = ddata.visited[next];
          Visited newV = old | toWrite;
          if (old == newV)
            break;
          if (__sync_bool_compare_and_swap(&ddata.visited[next], old, newV))
            break;
        }
        Dist oldDist = ddata.dist;
        if (ddata.dist != newDist)
          return __sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist);
      }
      return false;
    }
  };

  struct Update {
    LigraAlgo* self;
    Graph& graph;
    int cur;
    int next;
    Update(LigraAlgo* s, Graph& g, int c, int n): self(s), graph(g), cur(c), next(n) { 
    }
    void operator()(size_t id) {
      LNode& data = graph.getData(graph.nodeFromId(id), Galois::MethodFlag::NONE);
      data.visited[next] |= data.visited[cur];
    }
  };

  size_t operator()(Graph& graph, const GNode& source) {
    Galois::GraphNodeBagPair<> bags(graph.size());

    if (startNode != 0)
      std::cerr << "Warning: Ignoring user-requested start node\n";
    Dist newDist = 0;
    unsigned sampleSize = std::min(graph.size(), sizeof(Visited) * 8);
    unsigned count = 0;
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      LNode& data = graph.getData(*ii);
      data.dist = 0;
      data.visited[1] = (Visited)1 << count;
      bags.next().push(graph.idFromNode(*ii), graph.size());

      if (++count >= sampleSize)
        break;
    }
    
    while (!bags.next().empty()) {
      bags.swap();
      newDist++;
      int cur = newDist & 1;
      int next = (newDist + 1) & 1;
      Galois::do_all_local(bags.cur(), Update(this, graph, cur, next));
      this->outEdgeMap(memoryLimit, graph, EdgeOperator(this, cur, next, newDist), bags.cur(), bags.next(), false);
    }

    return newDist - 1;
  }
};
#endif

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
  GNode source, report;

  initialize(algo, graph, source);

  //Galois::preAlloc((numThreads + (graph.size() * sizeof(SNode) * 2) / Galois::Runtime::MM::pageSize)*8);
  Galois::reportPageAlloc("MeminfoPre");

  Galois::StatTimer T;
  T.start();
  resetGraph<Algo>(graph);
  size_t diameter = algo(graph, source);
  T.stop();
  
  Galois::reportPageAlloc("MeminfoPost");

  std::cout << "Estimated diameter: " << diameter << "\n";
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::simple: run<SimpleAlgo>(); break;
    case Algo::pickK: run<PickKAlgo>(); break;
#ifdef GALOIS_USE_EXP
    case Algo::ligra: run<LigraAlgo<false> >(); break;
    case Algo::ligraChi: run<LigraAlgo<true> >(); break;
    case Algo::graphlab: run<GraphLabAlgo<true> >(); break;
#endif
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}

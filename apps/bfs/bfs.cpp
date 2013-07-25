/** Breadth-first search -*- C++ -*-
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
 * Breadth-first search.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Graph/TypeTraits.h"
#ifdef GALOIS_USE_EXP
#include <boost/mpl/if.hpp>
#include "Galois/PriorityScheduling.h"
#include "Galois/Graph/OCGraph.h"
#include "Galois/Graph/GraphNodeBag.h"
#include "Galois/DomainSpecificExecutors.h"
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "HybridBFS.h"

#include <string>
#include <deque>
#include <sstream>
#include <limits>
#include <iostream>


static const char* name = "Breadth-first Search";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm";
static const char* url = "breadth_first_search";

//****** Command Line Options ******
enum Algo {
  async,
  barrier,
  barrierWithCas,
  barrierWithInline,
  deterministic,
  deterministicDisjoint,
  graphlab,
  highCentrality,
  hybrid,
  ligra,
  ligraChi,
  serial
};

enum DetAlgo {
  none,
  base,
  disjoint
};

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));
static cll::opt<bool> useDetBase("detBase", cll::desc("Deterministic"));
static cll::opt<bool> useDetDisjoint("detDisjoint", cll::desc("Deterministic with disjoint optimization"));
static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", cll::desc("Node to report distance to"), cll::init(1));
static cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::async, "async", "Asynchronous"),
      clEnumValN(Algo::barrier, "barrier", "Parallel optimized with barrier (default)"),
      clEnumValN(Algo::barrierWithCas, "barrierWithCas", "Use compare-and-swap to update nodes"),
      clEnumValN(Algo::deterministic, "detBase", "Deterministic"),
      clEnumValN(Algo::deterministicDisjoint, "detDisjoint", "Deterministic with disjoint optimization"),
      clEnumValN(Algo::graphlab, "graphlab", "Use GraphLab programming model"),
      clEnumValN(Algo::highCentrality, "highCentrality", "Optimization for graphs with many shortest paths"),
      clEnumValN(Algo::hybrid, "hybrid", "Hybrid of barrier and high centrality algorithms"),
      clEnumValN(Algo::ligraChi, "ligraChi", "Use Ligra and GraphChi programming model"),
      clEnumValN(Algo::ligra, "ligra", "Use Ligra programming model"),
      clEnumValN(Algo::serial, "serial", "Serial"),
#ifdef GALOIS_USE_EXP
      clEnumValN(Algo::barrierWithInline, "barrierWithInline", "Optimized with inlined workset"),
#endif
      clEnumValEnd), cll::init(Algo::barrier));

typedef unsigned int Dist;
static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max() - 1;

//! Standard data type on nodes
struct SNode {
  Dist dist;
};

template<typename Graph, typename Enable = void>
struct not_consistent {
  not_consistent(Graph& g) { }

  bool operator()(typename Graph::GraphNode n) const { return false; }
};

template<typename Graph>
struct not_consistent<Graph, typename std::enable_if<!Galois::Graph::is_segmented<Graph>::value>::type> {
  Graph& g;
  not_consistent(Graph& g): g(g) { }

  bool operator()(typename Graph::GraphNode n) const {
    Dist dist = g.getData(n).dist;
    if (dist == DIST_INFINITY)
      return false;

    for (typename Graph::edge_iterator ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii) {
      Dist ddist = g.getData(g.getEdgeDst(ii)).dist;
      if (ddist > dist + 1) {
	return true;
      }
    }
    return false;
  }
};

template<typename Graph>
struct not_visited {
  Graph& g;

  not_visited(Graph& g): g(g) { }

  bool operator()(typename Graph::GraphNode n) const {
    return g.getData(n).dist >= DIST_INFINITY;
  }
};

template<typename Graph>
struct max_dist {
  Graph& g;
  Galois::GReduceMax<Dist>& m;

  max_dist(Graph& g, Galois::GReduceMax<Dist>& m): g(g), m(m) { }

  void operator()(typename Graph::GraphNode n) const {
    Dist d = g.getData(n).dist;
    if (d == DIST_INFINITY)
      return;
    m.update(d);
  }
};

template<typename Graph>
bool verify(Graph& graph, typename Graph::GraphNode source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  namespace pstl = Galois::ParallelSTL;

  size_t notVisited = pstl::count_if(graph.begin(), graph.end(), not_visited<Graph>(graph));
  if (notVisited) {
    std::cerr << notVisited << " unvisited nodes; this is an error if the graph is strongly connected\n";
  }

  bool consistent = pstl::find_if(graph.begin(), graph.end(), not_consistent<Graph>(graph)) == graph.end();
  if (!consistent) {
    std::cerr << "node found with incorrect distance\n";
    return false;
  }

  Galois::GReduceMax<Dist> m;
  Galois::do_all(graph.begin(), graph.end(), max_dist<Graph>(graph, m));
  std::cout << "max dist: " << m.reduce() << "\n";
  
  return true;
}

template<typename Graph>
struct Initialize {
  Graph& g;
  Initialize(Graph& g): g(g) { }
  void operator()(typename Graph::GraphNode n) {
    g.getData(n).dist = DIST_INFINITY;
  }
};

template<typename Algo>
void initialize(Algo& algo,
    typename Algo::Graph& graph,
    typename Algo::Graph::GraphNode& source,
    typename Algo::Graph::GraphNode& report) {

  algo.readGraph(graph);
  std::cout << "Read " << graph.size() << " nodes\n";

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr 
      << "failed to set report: " << reportNode 
      << "or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
  
  typename Algo::Graph::iterator it = graph.begin();
  std::advance(it, startNode);
  source = *it;
  it = graph.begin();
  std::advance(it, reportNode);
  report = *it;
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

//! Serial BFS using optimized flags based off asynchronous algo
struct SerialAlgo {
  typedef Galois::Graph::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Serial"; }
  void readGraph(Graph& graph) { Galois::Graph::readGraph(graph, filename); }

  void operator()(Graph& graph, const GNode source) const {
    std::deque<GNode> wl;
    graph.getData(source).dist = 0;
    wl.push_back(source);

    while (!wl.empty()) {
      GNode n = wl.front();
      wl.pop_front();

      SNode& data = graph.getData(n, Galois::MethodFlag::NONE);

      Dist newDist = data.dist + 1;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

        if (newDist < ddata.dist) {
          ddata.dist = newDist;
          wl.push_back(dst);
        }
      }
    }
  }
};

//! Galois BFS using optimized flags
struct AsyncAlgo {
  typedef Galois::Graph::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type
    ::with_numa_alloc<true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Asynchronous"; }
  void readGraph(Graph& graph) { Galois::Graph::readGraph(graph, filename); }

  typedef std::pair<GNode, Dist> WorkItem;

  struct Indexer: public std::unary_function<WorkItem,Dist> {
    Dist operator()(const WorkItem& val) const {
      return val.second;
    }
  };

  struct Process {
    typedef int tt_does_not_need_aborts;

    Graph& graph;
    Process(Graph& g): graph(g) { }

    void operator()(WorkItem& item, Galois::UserContext<WorkItem>& ctx) const {
      GNode n = item.first;

      Dist newDist = item.second;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

        Dist oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            ctx.push(WorkItem(dst, newDist + 1));
            break;
          }
        }
      }
    }
  };

  void operator()(Graph& graph, const GNode& source) const {
    using namespace Galois::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<Indexer,dChunk> OBIM;
    
    graph.getData(source).dist = 0;

    Galois::for_each<OBIM>(WorkItem(source, 1), Process(graph));
  }
};

#ifdef GALOIS_USE_EXP
template<bool UseGraphChi>
struct LigraAlgo: public Galois::LigraGraphChi::ChooseExecutor<UseGraphChi> {
  typedef typename Galois::Graph::LC_CSR_Graph<SNode,void>
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type InnerGraph;
  typedef typename boost::mpl::if_c<UseGraphChi,
          Galois::Graph::OCImmutableEdgeGraph<SNode,void>,
          Galois::Graph::LC_InOut_Graph<InnerGraph>>::type
          Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return UseGraphChi ? "LigraChi" : "Ligra"; }

  void readGraph(Graph& graph) { 
    readInOutGraph(graph);
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct EdgeOperator {
    Dist newDist;
    EdgeOperator(Dist d): newDist(d) { }

    template<typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode n) { 
      return graph.getData(n, Galois::MethodFlag::NONE).dist == DIST_INFINITY;
    }

    template<typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src, typename GTy::GraphNode dst, typename GTy::edge_data_reference) {
      SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

      Dist oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          return false;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          return true;
        }
      }
      return false;
    }
  };

  void operator()(Graph& graph, const GNode& source) {
    Galois::GraphNodeBagPair<> bags(graph.size());

    Dist newDist = 1;
    graph.getData(source).dist = 0;

    this->outEdgeMap(memoryLimit, graph, EdgeOperator(newDist), source, bags.next());
    
    while (!bags.next().empty()) {
      bags.swap();
      newDist++;
      this->outEdgeMap(memoryLimit, graph, EdgeOperator(newDist), bags.cur(), bags.next(), false);
    }
  }
};

struct GraphLabAlgo {
  typedef typename Galois::Graph::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type
    ::with_numa_alloc<true>::type InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  void readGraph(Graph& graph) {
    readInOutGraph(graph);
  }

  std::string name() const { return "GraphLab"; }

  struct Program {
    typedef size_t gather_type;

    struct message_type {
      size_t value;
      message_type(): value(std::numeric_limits<size_t>::max()) { }
      explicit message_type(size_t v): value(v) { }
      message_type& operator+=(const message_type& other) {
        value = std::min<size_t>(value, other.value);
        return *this;
      }
    };

    typedef int tt_needs_scatter_out_edges;

  private:
    size_t received_dist;
    bool changed;

  public:
    Program(): received_dist(DIST_INFINITY), changed(false) { }

    void init(Graph& graph, GNode node, const message_type& msg) {
      received_dist = msg.value;
    }

    void apply(Graph& graph, GNode node, const gather_type&) {
      changed = false;
      SNode& sdata = graph.getData(node, Galois::MethodFlag::NONE);
      if (sdata.dist > received_dist) {
        changed = true;
        sdata.dist = received_dist;
      }
    }

    bool needsScatter(Graph& graph, GNode node) {
      return changed;
    }

    void gather(Graph& graph, GNode node, GNode src, GNode dst, gather_type&, typename Graph::edge_data_reference) { }

    void scatter(Graph& graph, GNode node, GNode src, GNode dst,
        Galois::GraphLab::Context<Graph,Program>& ctx, typename Graph::edge_data_reference) {
      SNode& sdata = graph.getData(node, Galois::MethodFlag::NONE);
      Dist newDist = sdata.dist + 1;

      if (graph.getData(dst, Galois::MethodFlag::NONE).dist > newDist) {
        ctx.push(dst, message_type(newDist));
      }
    }
  };

  void operator()(Graph& graph, const GNode& source) {
    Galois::GraphLab::SyncEngine<Graph,Program> engine(graph, Program());
    engine.signal(source, Program::message_type(0));
    engine.execute();
  }
};
#endif

/**
 * Alternate between processing outgoing edges or incoming edges. Best for
 * graphs that have many redundant shortest paths.
 *
 * S. Beamer, K. Asanovic and D. Patterson. Direction-optimizing breadth-first
 * search. In Supercomputing. 2012.
 */
struct HighCentralityAlgo {
  typedef Galois::Graph::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type 
    ::with_numa_alloc<true>::type InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;
  
  std::string name() const { return "High Centrality"; }

  void readGraph(Graph& graph) { readInOutGraph(graph); }

  struct CountingBag {
    Galois::InsertBag<GNode> wl;
    Galois::GAccumulator<size_t> count;

    void clear() {
      wl.clear();
      count.reset();
    }

    bool empty() { return wl.empty(); }
    size_t size() { return count.reduce(); }
  };

  CountingBag bags[2];

  struct ForwardProcess {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    Graph& graph;
    CountingBag* next;
    Dist newDist;
    ForwardProcess(Graph& g, CountingBag* n, int d): graph(g), next(n), newDist(d) { }

    void operator()(const GNode& n, Galois::UserContext<GNode>&) {
      (*this)(n);
    }

    void operator()(const Graph::edge_iterator& it, Galois::UserContext<Graph::edge_iterator>&) {
      (*this)(it);
    }

    void operator()(const Graph::edge_iterator& ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

      Dist oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          return;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          next->wl.push(dst);
          next->count += 1
            + std::distance(graph.edge_begin(dst, Galois::MethodFlag::NONE),
              graph.edge_end(dst, Galois::MethodFlag::NONE));
          break;
        }
      }
    }

    void operator()(const GNode& n) {
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        (*this)(ii);
      }
    }
  };

  struct BackwardProcess {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    Graph& graph;
    CountingBag* next;
    Dist newDist; 
    BackwardProcess(Graph& g, CountingBag* n, int d): graph(g), next(n), newDist(d) { }

    void operator()(const GNode& n, Galois::UserContext<GNode>&) {
      (*this)(n);
    }

    void operator()(const GNode& n) {
      SNode& sdata = graph.getData(n, Galois::MethodFlag::NONE);
      if (sdata.dist <= newDist)
        return;

      for (Graph::in_edge_iterator ii = graph.in_edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.in_edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = graph.getInEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

        if (ddata.dist + 1 == newDist) {
          sdata.dist = newDist;
          next->wl.push(n);
          next->count += 1
            + std::distance(graph.edge_begin(n, Galois::MethodFlag::NONE),
              graph.edge_end(n, Galois::MethodFlag::NONE));
          break;
        }
      }
    }
  };

  void operator()(Graph& graph, const GNode& source) {
    using namespace Galois::WorkList;
    typedef dChunkedLIFO<256> WL;
    int next = 0;
    Dist newDist = 1;
    graph.getData(source).dist = 0;
    Galois::for_each(graph.out_edges(source, Galois::MethodFlag::NONE).begin(), 
        graph.out_edges(source, Galois::MethodFlag::NONE).end(),
        ForwardProcess(graph, &bags[next], newDist));
    while (!bags[next].empty()) {
      size_t nextSize = bags[next].size();
      int cur = next;
      next = (cur + 1) & 1;
      newDist++;
      std::cout << nextSize << " " << (nextSize > graph.sizeEdges() / 20) << "\n";
      if (nextSize > graph.sizeEdges() / 20)
        Galois::do_all_local(graph, BackwardProcess(graph, &bags[next], newDist));
      else
        Galois::for_each_local<WL>(bags[cur].wl, ForwardProcess(graph, &bags[next], newDist));
      bags[cur].clear();
    }
  }
};

//! BFS using optimized flags and barrier scheduling 
template<typename WL, bool useCas>
struct BarrierAlgo {
  typedef Galois::Graph::LC_CSR_Graph<SNode,void>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;
  typedef std::pair<GNode,Dist> WorkItem;

  std::string name() const { return "Barrier"; }
  void readGraph(Graph& graph) { Galois::Graph::readGraph(graph, filename); }

  struct Process {
    typedef int tt_does_not_need_aborts;

    Graph& graph;
    Process(Graph& g): graph(g) { }

    void operator()(const WorkItem& item, Galois::UserContext<WorkItem>& ctx) const {
      GNode n = item.first;

      Dist newDist = item.second;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

        Dist oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          if (!useCas || __sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            if (!useCas)
              ddata.dist = newDist;
            ctx.push(WorkItem(dst, newDist + 1));
            break;
          }
        }
      }
    }
  };

  void operator()(Graph& graph, const GNode& source) const {
    graph.getData(source).dist = 0;
    Galois::for_each<WL>(WorkItem(source, 1), Process(graph));
  }
};

struct HybridAlgo: public HybridBFS<SNode,Dist> {
  std::string name() const { return "Hybrid"; }

  void readGraph(Graph& graph) { readInOutGraph(graph); }
};

template<DetAlgo Version>
struct DeterministicAlgo {
  typedef Galois::Graph::LC_CSR_Graph<SNode,void>
    ::template with_numa_alloc<true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Deterministic"; }
  void readGraph(Graph& graph) { Galois::Graph::readGraph(graph, filename); }

  typedef std::pair<GNode,int> WorkItem;

  struct Process {
    typedef int tt_needs_per_iter_alloc; // For LocalState
    static_assert(Galois::needs_per_iter_alloc<Process>::value, "Oops");

    Graph& graph;

    Process(Graph& g): graph(g) { }

    struct LocalState {
      typedef typename Galois::PerIterAllocTy::rebind<GNode>::other Alloc;
      typedef std::deque<GNode,Alloc> Pending;
      Pending pending;
      LocalState(Process& self, Galois::PerIterAllocTy& alloc): pending(alloc) { }
    };
    typedef LocalState GaloisDeterministicLocalState;
    static_assert(Galois::has_deterministic_local_state<Process>::value, "Oops");

    uintptr_t galoisDeterministicId(const WorkItem& item) const {
      return item.first;
    }
    static_assert(Galois::has_deterministic_id<Process>::value, "Oops");

    void build(const WorkItem& item, typename LocalState::Pending* pending) const {
      GNode n = item.first;

      Dist newDist = item.second;
      
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::ALL);

        Dist oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          pending->push_back(dst);
          break;
        }
      }
    }

    void modify(const WorkItem& item, Galois::UserContext<WorkItem>& ctx, typename LocalState::Pending* ppending) const {
      Dist newDist = item.second;
      bool useCas = false;

      for (typename LocalState::Pending::iterator ii = ppending->begin(), ei = ppending->end(); ii != ei; ++ii) {
        GNode dst = *ii;
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

        Dist oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          if (!useCas || __sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            if (!useCas)
              ddata.dist = newDist;
            ctx.push(WorkItem(dst, newDist + 1));
            break;
          }
        }
      }
    }

    void operator()(const WorkItem& item, Galois::UserContext<WorkItem>& ctx) const {
      typename LocalState::Pending* ppending;
      if (Version == DetAlgo::disjoint) {
        bool used;
        LocalState* localState = (LocalState*) ctx.getLocalState(used);
        ppending = &localState->pending;
        if (used) {
          modify(item, ctx, ppending);
          return;
        }
      }
      if (Version == DetAlgo::disjoint) {
        build(item, ppending);
      } else {
        typename LocalState::Pending pending(ctx.getPerIterAlloc());
        build(item, &pending);
        graph.getData(item.first, Galois::MethodFlag::WRITE); // Failsafe point
        modify(item, ctx, &pending);
      }
    }
  };

  void operator()(Graph& graph, const GNode& source) const {
#ifdef GALOIS_USE_EXP
    typedef Galois::WorkList::BulkSynchronousInline<> WL;
#else
    typedef Galois::WorkList::BulkSynchronous<Galois::WorkList::dChunkedLIFO<256> > WL;
#endif
    graph.getData(source).dist = 0;

    switch (Version) {
      case DetAlgo::none: Galois::for_each<WL>(WorkItem(source, 1), Process(graph)); break; 
      case DetAlgo::base: Galois::for_each_det(WorkItem(source, 1), Process(graph)); break;
      case DetAlgo::disjoint: Galois::for_each_det(WorkItem(source, 1), Process(graph)); break;
      default: std::cerr << "Unknown algorithm " << int(Version) << "\n"; abort();
    }
  }
};

template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Algo algo;
  Graph graph;
  GNode source, report;

  initialize(algo, graph, source, report);

  Galois::preAlloc(numThreads + (3*graph.size() * sizeof(typename Graph::node_data_type)) / Galois::Runtime::MM::pageSize);
  Galois::reportPageAlloc("MeminfoPre");

  Galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  Galois::do_all_local(graph, Initialize<typename Algo::Graph>(graph));
  algo(graph, source);
  T.stop();
  
  Galois::reportPageAlloc("MeminfoPost");

  std::cout << "Node " << reportNode << " has distance " << graph.getData(report).dist << "\n";

  if (!skipVerify) {
    if (verify(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      std::cerr << "Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  using namespace Galois::WorkList;
  typedef BulkSynchronous<dChunkedLIFO<256> > BSWL;

#ifdef GALOIS_USE_EXP
  typedef BulkSynchronousInline<> BSInline;
#else
  typedef BSWL BSInline;
#endif
  if (useDetDisjoint)
    algo = Algo::deterministicDisjoint;
  else if (useDetBase)
    algo = Algo::deterministic;

  Galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::serial: run<SerialAlgo>(); break;
    case Algo::async: run<AsyncAlgo>();  break;
    case Algo::barrier: run<BarrierAlgo<BSWL,false> >(); break;
    case Algo::barrierWithCas: run<BarrierAlgo<BSWL,true> >(); break;
    case Algo::barrierWithInline: run<BarrierAlgo<BSInline,false> >(); break;
    case Algo::highCentrality: run<HighCentralityAlgo>(); break;
    case Algo::hybrid: run<HybridAlgo>(); break;
#ifdef GALOIS_USE_EXP
    case Algo::ligra: run<LigraAlgo<false> >(); break;
    case Algo::ligraChi: run<LigraAlgo<true> >(); break;
    case Algo::graphlab: run<GraphLabAlgo>(); break;
#endif
    case Algo::deterministic: run<DeterministicAlgo<DetAlgo::base> >(); break;
    case Algo::deterministicDisjoint: run<DeterministicAlgo<DetAlgo::disjoint> >(); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}

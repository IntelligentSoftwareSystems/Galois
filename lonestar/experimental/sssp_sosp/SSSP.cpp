/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <deque>
#include <set>

#include "SSSP.h"
#include "GraphLabAlgo.h"
#include "LigraAlgo.h"

namespace cll = llvm::cl;

static const char* name = "Single Source Shortest Path";
static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using a modified chaotic iteration algorithm";
static const char* url = "single_source_shortest_path";

enum Algo { async, asyncWithCas, asyncPP, graphlab, ligra, ligraChi, serial };

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string>
    transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph",
                                     cll::desc("Input graph is symmetric"));
static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));
static cll::opt<unsigned int>
    reportNode("reportNode", cll::desc("Node to report distance to"),
               cll::init(1));
static cll::opt<int> stepShift("delta",
                               cll::desc("Shift value for the deltastep"),
                               cll::init(10));
cll::opt<unsigned int>
    memoryLimit("memoryLimit",
                cll::desc("Memory limit for out-of-core algorithms (in MB)"),
                cll::init(~0U));
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::async, "async", "Asynchronous"),
                clEnumValN(Algo::asyncPP, "asyncPP", "Async, CAS, push-pull"),
                clEnumValN(Algo::asyncWithCas, "asyncWithCas",
                           "Use compare-and-swap to update nodes"),
                clEnumValN(Algo::serial, "serial", "Serial"),
                clEnumValN(Algo::graphlab, "graphlab",
                           "Use GraphLab programming model"),
                clEnumValN(Algo::ligraChi, "ligraChi",
                           "Use Ligra and GraphChi programming model"),
                clEnumValN(Algo::ligra, "ligra", "Use Ligra programming model"),
                clEnumValEnd),
    cll::init(Algo::asyncWithCas));

static const bool trackWork = true;
static galois::Statistic* BadWork;
static galois::Statistic* WLEmptyWork;

template <typename Graph>
struct not_visited {
  Graph& g;

  not_visited(Graph& g) : g(g) {}

  bool operator()(typename Graph::GraphNode n) const {
    return g.getData(n).dist >= DIST_INFINITY;
  }
};

template <typename Graph, typename Enable = void>
struct not_consistent {
  not_consistent(Graph& g) {}

  bool operator()(typename Graph::GraphNode n) const { return false; }
};

template <typename Graph>
struct not_consistent<Graph,
                      typename std::enable_if<
                          !galois::graphs::is_segmented<Graph>::value>::type> {
  Graph& g;
  not_consistent(Graph& g) : g(g) {}

  bool operator()(typename Graph::GraphNode n) const {
    Dist dist = g.getData(n).dist;
    if (dist == DIST_INFINITY)
      return false;

    for (auto ii : g.edges(n)) {
      Dist ddist = g.getData(g.getEdgeDst(ii)).dist;
      Dist w     = g.getEdgeData(ii);
      if (ddist > dist + w) {
        // std::cout << ddist << " " << dist + w << " " << n << " " <<
        // g.getEdgeDst(ii) << "\n"; // XXX
        return true;
      }
    }
    return false;
  }
};

template <typename Graph>
struct max_dist {
  Graph& g;
  galois::GReduceMax<Dist>& m;

  max_dist(Graph& g, galois::GReduceMax<Dist>& m) : g(g), m(m) {}

  void operator()(typename Graph::GraphNode n) const {
    Dist d = g.getData(n).dist;
    if (d == DIST_INFINITY)
      return;
    m.update(d);
  }
};

template <typename UpdateRequest>
struct UpdateRequestIndexer
    : public std::unary_function<UpdateRequest, unsigned int> {
  unsigned int operator()(const UpdateRequest& val) const {
    unsigned int t = val.w >> stepShift;
    return t;
  }
};

template <typename Graph>
bool verify(Graph& graph, typename Graph::GraphNode source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  namespace pstl = galois::ParallelSTL;

  size_t notVisited =
      pstl::count_if(graph.begin(), graph.end(), not_visited<Graph>(graph));
  if (notVisited) {
    std::cerr << notVisited
              << " unvisited nodes; this is an error if the "
                 "graph is strongly connected\n";
  }

  bool consistent = pstl::find_if(graph.begin(), graph.end(),
                                  not_consistent<Graph>(graph)) == graph.end();
  if (!consistent) {
    std::cerr << "node found with incorrect distance\n";
    return false;
  }

  galois::GReduceMax<Dist> m;
  galois::do_all(graph.begin(), graph.end(), max_dist<Graph>(graph, m));
  std::cout << "max dist: " << m.reduce() << "\n";

  return true;
}

template <typename Algo>
void initialize(Algo& algo, typename Algo::Graph& graph,
                typename Algo::Graph::GraphNode& source,
                typename Algo::Graph::GraphNode& report) {

  algo.readGraph(graph);
  std::cout << "Read " << graph.size() << " nodes\n";

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  typename Algo::Graph::iterator it = graph.begin();
  std::advance(it, startNode);
  source = *it;
  it     = graph.begin();
  std::advance(it, reportNode);
  report = *it;
}

template <typename Graph>
void readInOutGraph(Graph& graph) {
  using namespace galois::graphs;
  if (symmetricGraph) {
    //! [Reading a graph]
    galois::graphs::readGraph(graph, filename);
    //! [Reading a graph]
  } else if (transposeGraphName.size()) {
    galois::graphs::readGraph(graph, filename, transposeGraphName);
  } else {
    GALOIS_DIE("Graph type not supported");
  }
}

struct SerialAlgo {
  //! [Define LC_CSR_Graph]
  typedef galois::graphs::LC_CSR_Graph<SNode, uint32_t>::with_no_lockable<
      true>::type Graph;
  //! [Define LC_CSR_Graph]

  typedef Graph::GraphNode GNode;
  typedef UpdateRequestCommon<GNode> UpdateRequest;

  std::string name() const { return "Serial"; }
  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}

    void operator()(Graph::GraphNode n) const {
      g.getData(n).dist = DIST_INFINITY;
    }
  };

  void operator()(Graph& graph, const GNode src) const {
    std::set<UpdateRequest, std::less<UpdateRequest>> initial;
    UpdateRequest init(src, 0);
    initial.insert(init);

    galois::Statistic counter("Iterations");

    while (!initial.empty()) {
      counter += 1;
      UpdateRequest req = *initial.begin();
      initial.erase(initial.begin());
      SNode& data = graph.getData(req.n, galois::MethodFlag::UNPROTECTED);
      if (req.w < data.dist) {
        data.dist = req.w;
        for (auto ii : graph.edges(req.n, galois::MethodFlag::UNPROTECTED)) {
          GNode dst    = graph.getEdgeDst(ii);
          Dist d       = graph.getEdgeData(ii);
          Dist newDist = req.w + d;
          if (newDist <
              graph.getData(dst, galois::MethodFlag::UNPROTECTED).dist) {
            initial.insert(UpdateRequest(dst, newDist));
          }
        }
      }
    }
  }
};

template <bool UseCas>
struct AsyncAlgo {
  typedef SNode Node;

  // ! [Define LC_InlineEdge_Graph]
  typedef galois::graphs::LC_InlineEdge_Graph<
      Node, uint32_t>::template with_out_of_line_lockable<true>::type::
      template with_compressed_node_ptr<true>::type::template with_numa_alloc<
          true>::type Graph;
  // ! [Define LC_InlineEdge_Graph]

  typedef typename Graph::GraphNode GNode;
  typedef UpdateRequestCommon<GNode> UpdateRequest;

  std::string name() const {
    return UseCas ? "Asynchronous with CAS" : "Asynchronous";
  }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(typename Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  template <typename Pusher>
  void relaxEdge(Graph& graph, Dist sdist, typename Graph::edge_iterator ii,
                 Pusher& pusher) {
    GNode dst    = graph.getEdgeDst(ii);
    Dist d       = graph.getEdgeData(ii);
    Node& ddata  = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist newDist = sdist + d;
    Dist oldDist = ddata.dist;
    if (!UseCas && newDist < oldDist) {
      ddata.dist = newDist;
      if (trackWork && oldDist != DIST_INFINITY)
        *BadWork += 1;
      pusher.push(UpdateRequest(dst, newDist));
    } else {
      while (newDist < oldDist) {
        if (ddata.dist.compare_exchange_weak(oldDist, newDist,
                                             std::memory_order_acq_rel)) {
          if (trackWork && oldDist != DIST_INFINITY)
            *BadWork += 1;
          pusher.push(UpdateRequest(dst, newDist));
        }
      }
    }
  }

  template <typename Pusher>
  void relaxNode(Graph& graph, UpdateRequest& req, Pusher& pusher) {
    const galois::MethodFlag flag =
        UseCas ? galois::MethodFlag::UNPROTECTED : galois::MethodFlag::WRITE;
    Dist sdist = graph.getData(req.n, flag).dist;

    if (req.w != sdist) {
      if (trackWork)
        *WLEmptyWork += 1;
      return;
    }

    for (auto ii : graph.edges(req.n, flag)) {
      relaxEdge(graph, sdist, ii, pusher);
    }
  }

  struct Process {
    AsyncAlgo* self;
    Graph& graph;
    Process(AsyncAlgo* s, Graph& g) : self(s), graph(g) {}
    void operator()(UpdateRequest& req,
                    galois::UserContext<UpdateRequest>& ctx) {
      self->relaxNode(graph, req, ctx);
    }
  };

  typedef galois::InsertBag<UpdateRequest> Bag;

  struct InitialProcess {
    AsyncAlgo* self;
    Graph& graph;
    Bag& bag;
    Node& sdata;
    InitialProcess(AsyncAlgo* s, Graph& g, Bag& b, Node& d)
        : self(s), graph(g), bag(b), sdata(d) {}
    void operator()(typename Graph::edge_iterator ii) const {
      self->relaxEdge(graph, sdata.dist, ii, bag);
    }
  };

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    typedef PerSocketChunkFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Chunk,
                                   10, false>
        OBIM;

    std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
    std::cout
        << "WARNING: Performance varies considerably due to delta parameter.\n";
    std::cout
        << "WARNING: Do not expect the default to be good for your graph.\n";

    Bag initial;
    graph.getData(source).dist = 0;
    galois::do_all(graph.edges(source, galois::MethodFlag::UNPROTECTED).begin(),
                   graph.edges(source, galois::MethodFlag::UNPROTECTED).end(),
                   InitialProcess(this, graph, initial, graph.getData(source)));
    galois::for_each(initial, Process(this, graph), galois::wl<OBIM>());
  }
};

struct AsyncAlgoPP {
  typedef SNode Node;

  typedef galois::graphs::LC_InlineEdge_Graph<Node, uint32_t>::
      with_out_of_line_lockable<true>::type::with_compressed_node_ptr<
          true>::type::with_numa_alloc<true>::type Graph;
  typedef Graph::GraphNode GNode;
  typedef UpdateRequestCommon<GNode> UpdateRequest;

  std::string name() const { return "Asynchronous with CAS and Push and pull"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  template <typename Pusher>
  void relaxEdge(Graph& graph, Dist& sdata, typename Graph::edge_iterator ii,
                 Pusher& pusher) {
    GNode dst    = graph.getEdgeDst(ii);
    Dist d       = graph.getEdgeData(ii);
    Node& ddata  = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist newDist = sdata + d;
    Dist oldDist;
    if (newDist < (oldDist = ddata.dist)) {
      do {
        if (ddata.dist.compare_exchange_weak(oldDist, newDist)) {
          if (trackWork && oldDist != DIST_INFINITY)
            *BadWork += 1;
          pusher.push(UpdateRequest(dst, newDist));
          break;
        }
      } while (newDist < oldDist);
    } else {
      sdata = std::min(oldDist + d, sdata);
    }
  }

  struct Process {
    AsyncAlgoPP* self;
    Graph& graph;
    Process(AsyncAlgoPP* s, Graph& g) : self(s), graph(g) {}

    void operator()(UpdateRequest& req,
                    galois::UserContext<UpdateRequest>& ctx) {
      const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
      Node& sdata                   = graph.getData(req.n, flag);
      Dist sdist                    = sdata.dist;

      if (req.w != sdist) {
        if (trackWork)
          *WLEmptyWork += 1;
        return;
      }

      for (auto ii : graph.edges(req.n, flag)) {
        self->relaxEdge(graph, sdist, ii, ctx);
      }

      // //try doing a pull
      // Dist oldDist;
      // while (sdist < (oldDist = *psdist)) {
      //   if (__sync_bool_compare_and_swap(psdist, oldDist, sdist)) {
      //     req.w = sdist;
      //     operator()(req, ctx);
      //   }
      // }
    }
  };

  typedef galois::InsertBag<UpdateRequest> Bag;

  struct InitialProcess {
    AsyncAlgoPP* self;
    Graph& graph;
    Bag& bag;
    InitialProcess(AsyncAlgoPP* s, Graph& g, Bag& b)
        : self(s), graph(g), bag(b) {}
    void operator()(Graph::edge_iterator ii) const {
      Dist d = 0;
      self->relaxEdge(graph, d, ii, bag);
    }
  };

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    typedef ChunkFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Chunk,
                                   10, false>
        OBIM;

    std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
    std::cout
        << "WARNING: Performance varies considerably due to delta parameter.\n";
    std::cout
        << "WARNING: Do not expect the default to be good for your graph.\n";

    Bag initial;
    graph.getData(source).dist = 0;
    galois::do_all(graph.edges(source, galois::MethodFlag::UNPROTECTED).begin(),
                   graph.edges(source, galois::MethodFlag::UNPROTECTED).end(),
                   InitialProcess(this, graph, initial));
    galois::for_each(initial, Process(this, graph), galois::wl<OBIM>());
  }
};

namespace galois {
namespace DEPRECATED {
template <>
struct does_not_need_aborts<AsyncAlgo<true>::Process>
    : public boost::true_type {};
} // namespace DEPRECATED
} // namespace galois

static_assert(
    galois::DEPRECATED::does_not_need_aborts<AsyncAlgo<true>::Process>::value,
    "Oops");

template <typename Algo>
void run(bool prealloc = true) {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Algo algo;
  Graph graph;
  GNode source, report;

  initialize(algo, graph, source, report);

  size_t approxNodeData = graph.size() * 64;
  // size_t approxEdgeData = graph.sizeEdges() * sizeof(typename
  // Graph::edge_data_type) * 2;
  if (prealloc)
    galois::preAlloc(numThreads +
                     approxNodeData / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  galois::do_all(graph, typename Algo::Initialize(graph));
  algo(graph, source);
  T.stop();

  galois::reportPageAlloc("MeminfoPost");
  galois::runtime::reportNumaAlloc("NumaPost");

  std::cout << "Node " << reportNode << " has distance "
            << graph.getData(report).dist << "\n";

  if (!skipVerify) {
    if (verify(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("Verification failed");
    }
  }
}

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (trackWork) {
    BadWork     = new galois::Statistic("BadWork");
    WLEmptyWork = new galois::Statistic("EmptyWork");
  }

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
  case Algo::serial:
    run<SerialAlgo>();
    break;
  case Algo::async:
    run<AsyncAlgo<false>>();
    break;
  case Algo::asyncWithCas:
    run<AsyncAlgo<true>>();
    break;
  case Algo::asyncPP:
    run<AsyncAlgoPP>();
    break;
    // Fixme: ligra still asumes gcc sync builtins
    //  case Algo::ligra:
    //    run<LigraAlgo<false>>();
    //    break;
    //  case Algo::ligraChi:
    //    run<LigraAlgo<true>>(false);
    //    break;
  case Algo::graphlab:
    run<GraphLabAlgo>();
    break;
  default:
    std::cerr << "Unknown algorithm\n";
    abort();
  }
  T.stop();

  if (trackWork) {
    delete BadWork;
    delete WLEmptyWork;
  }

  return 0;
}

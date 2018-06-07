/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
#include "galois/UserContext.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#ifdef GALOIS_USE_EXP
#include "galois/worklists/Partitioned.h"
#endif
#include "llvm/Support/CommandLine.h"

#include "LC_PartitionedInlineEdge_Graph.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <regex>
#include <iostream>
#include <deque>
#include <set>

namespace cll = llvm::cl;

static const char* name = "Single Source Shortest Path";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified chaotic iteration algorithm";
static const char* url = "single_source_shortest_path";

enum Algo {
  normal,
  part,
  part2,
  part3,
  part4,
  part5,
  part6,
  part7
};

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> partsname(cll::Positional, cll::desc("<part base graph>"));
static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", cll::desc("Node to report distance to"), cll::init(1));
static cll::opt<unsigned int> deltaShift("delta", cll::desc("Shift value for the delta step"), cll::init(10));
static cll::opt<unsigned int> blockShift("block", cll::desc("Shift value for the block"), cll::init(10));
static cll::opt<unsigned int> partitionShift("part", cll::desc("Shift value for the partition"), cll::init(10));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::part, "part", "Graph partitioned by source nodes"),
      clEnumValN(Algo::part2, "part2", "Graph partitioned by source nodes"),
      clEnumValN(Algo::part3, "part3", "Graph partitioned by source nodes"),
      clEnumValN(Algo::part4, "part4", "Graph partitioned by source nodes"),
      clEnumValN(Algo::part5, "part5", "Graph partitioned by source nodes"),
      clEnumValN(Algo::part6, "part6", "Graph partitioned by source nodes"),
      clEnumValN(Algo::part7, "part7", "Graph partitioned by source nodes"),
      clEnumValN(Algo::normal, "normal", ""),
      clEnumValEnd), cll::init(Algo::normal));

static int blockStep;
static const bool trackBadWork = true;
static galois::Statistic* BadWork;
static galois::Statistic* WLEmptyWork;

typedef unsigned int Dist;
static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max() - 1;

template<typename GrNode>
struct UpdateRequestCommon {
  GrNode n;
  Dist w;

  UpdateRequestCommon(const GrNode& N, Dist W): n(N), w(W) {}
  
  UpdateRequestCommon(): n(), w(0) {}

  bool operator>(const UpdateRequestCommon& rhs) const {
    if (w > rhs.w) return true;
    if (w < rhs.w) return false;
    return n > rhs.n;
  }

  bool operator<(const UpdateRequestCommon& rhs) const {
    if (w < rhs.w) return true;
    if (w > rhs.w) return false;
    return n < rhs.n;
  }

  bool operator!=(const UpdateRequestCommon& other) const {
    if (w != other.w) return true;
    return n != other.n;
  }

  uintptr_t getID() const {
    return reinterpret_cast<uintptr_t>(n);
  }
};

struct SNode {
  Dist dist;
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
struct not_consistent {
  Graph& g;
  not_consistent(Graph& g): g(g) { }

  bool operator()(typename Graph::GraphNode n) const {
    Dist dist = g.getData(n).dist;
    if (dist == DIST_INFINITY)
      return false;

    for (typename Graph::edge_iterator ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii) {
      Dist ddist = g.getData(g.getEdgeDst(ii)).dist;
      Dist w = g.getEdgeData(ii);
      if (ddist > dist + w) {
        //std::cout << ddist << " " << dist + w << " " << n << " " << g.getEdgeDst(ii) << "\n"; // XXX
	return true;
      }
    }
    return false;
  }
};

template<typename Graph>
struct max_dist {
  Graph& g;
  galois::GReduceMax<Dist>& m;

  max_dist(Graph& g, galois::GReduceMax<Dist>& m): g(g), m(m) { }

  void operator()(typename Graph::GraphNode n) const {
    Dist d = g.getData(n).dist;
    if (d == DIST_INFINITY)
      return;
    m.update(d);
  }
};

template<typename UpdateRequest>
struct UpdateRequestIndexer: public std::unary_function<UpdateRequest, unsigned int> {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w >> deltaShift;
    return t;
  }
};

template<typename Graph>
bool verify(Graph& graph, typename Graph::GraphNode source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  namespace pstl = galois::ParallelSTL;

  size_t notVisited = pstl::count_if(graph.begin(), graph.end(), not_visited<Graph>(graph));
  if (notVisited) {
    std::cerr << notVisited << " unvisited nodes; this is an error if the graph is strongly connected\n";
  }

  bool consistent = pstl::find_if(graph.begin(), graph.end(), not_consistent<Graph>(graph)) == graph.end();
  if (!consistent) {
    std::cerr << "node found with incorrect distance\n";
    return false;
  }

  galois::GReduceMax<Dist> m;
  galois::do_all(graph.begin(), graph.end(), max_dist<Graph>(graph, m));
  std::cout << "max dist: " << m.reduce() << "\n";
  
  return true;
}

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
      << " or failed to set source: " << startNode << "\n";
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

template<bool WithPartitioning>
struct AsyncAlgo {
  typedef SNode Node;

  typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::template with_out_of_line_lockable<true>
    ::template with_compressed_node_ptr<true>
    ::template with_numa_alloc<!WithPartitioning>
    Graph;
  typedef typename Graph::GraphNode GNode;
  typedef UpdateRequestCommon<GNode> UpdateRequest;
  typedef galois::InsertBag<UpdateRequest> Bag;
  typedef std::deque<Bag> Bags;

  std::string name() const { return WithPartitioning ? "partitioned" : "not partitioned"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(typename Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  // Operator without partitioning
  template<typename Pusher>
  void relaxEdge(Graph& graph, Node& sdata, typename Graph::edge_iterator ii, Pusher& pusher) {
    GNode dst = graph.getEdgeDst(ii);
    Dist d = graph.getEdgeData(ii);
    Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist newDist = sdata.dist + d;
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        pusher.push(UpdateRequest(dst, newDist));
        break;
      }
    }
  }

  // Operator with partitioning
  void relaxEdge(Graph& graph, Node& sdata, typename Graph::edge_iterator ii, int scale, Bags& bags) {
    GNode dst = graph.getEdgeDst(ii);
    // int id = graph.idFromNode(dst) / scale;
    int id = graph.idFromNode(dst) >> partitionShift;

    Dist d = graph.getEdgeData(ii);
    Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist newDist = sdata.dist + d;
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        bags[id].push(UpdateRequest(dst, newDist));
        break;
      }
    }
  }

  // Operator with partitioning
  void relaxEdge(Graph& graph, Node& sdata, typename Graph::edge_iterator ii, 
      int scale, int cur, Bags& bags, galois::UserContext<UpdateRequest>& pusher) {
    GNode dst = graph.getEdgeDst(ii);
    //int id = graph.idFromNode(dst) / scale;
    int id = graph.idFromNode(dst) >> partitionShift;
    
    Dist d = graph.getEdgeData(ii);
    Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist newDist = sdata.dist + d;
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
#if 1
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        if (id == cur)
          pusher.push(UpdateRequest(dst, newDist));
        else 
          bags[id].push(UpdateRequest(dst, newDist));
        break;
      }
#else
      ddata.dist = newDist;
      if (trackBadWork && oldDist != DIST_INFINITY)
        *BadWork += 1;
      if (id == cur)
        pusher.push(UpdateRequest(dst, newDist));
      else 
        bags[id].push(UpdateRequest(dst, newDist));
      break;
#endif
    }
  }

  // Without partitioning
  template<typename Pusher>
  void relaxNode(Graph& graph, GNode src, Dist prevDist, Pusher& pusher) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Node& sdata = graph.getData(src, flag);

    if (prevDist > sdata.dist) {
      if (trackBadWork)
        *WLEmptyWork += 1;
      return;
    }

    for (typename Graph::edge_iterator ii = graph.edge_begin(src, flag), ei = graph.edge_end(src, flag); ii != ei; ++ii) {
      relaxEdge(graph, sdata, ii, pusher);
    }
  }

  // With partitioning
  void relaxNode(Graph& graph, GNode src, Dist prevDist, 
      int scale, int cur, Bags& bags, galois::UserContext<UpdateRequest>& pusher) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Node& sdata = graph.getData(src, flag);

    if (prevDist > sdata.dist) {
      if (trackBadWork)
        *WLEmptyWork += 1;
      return;
    }

    for (typename Graph::edge_iterator ii = graph.edge_begin(src, flag), ei = graph.edge_end(src, flag); ii != ei; ++ii) {
      relaxEdge(graph, sdata, ii, scale, cur, bags, pusher);
    }
  }

  struct ProcessWithPartitioning {
    typedef int tt_does_not_need_aborts;
    typedef int tt_needs_parallel_break;

    AsyncAlgo* self;
    Graph& graph;
    int scale;
    int cur;
    Bag& next;
    Bags& bags;
    int count;

    ProcessWithPartitioning(AsyncAlgo* s, Graph& g, int _scale, int _cur, Bag& _next, Bags& _bags): 
      self(s), graph(g), scale(_scale), cur(_cur), next(_next), bags(_bags), count(0) { }

    void operator()(UpdateRequest& req, galois::UserContext<UpdateRequest>& ctx) {
      self->relaxNode(graph, req.n, req.w, scale, cur, bags, ctx);
      if (++count > blockStep)
        ctx.breakLoop();
    }

    void galoisParallelBreakReceiveRemaining(const UpdateRequest& req) { next.push(req); }
    //static_assert(galois::has_parallel_break_receive_remaining<ProcessWithPartitioning>::value, "Oops!");
  };

  struct ProcessWithoutPartitioning {
    typedef int tt_does_not_need_aborts;

    AsyncAlgo* self;
    Graph& graph;

    ProcessWithoutPartitioning(AsyncAlgo* s, Graph& g, int _scale, int _cur, Bag& _next, Bags& _bags): 
      self(s), graph(g) { }
    void operator()(UpdateRequest& req, galois::UserContext<UpdateRequest>& ctx) {
      self->relaxNode(graph, req.n, req.w, ctx);
    }
  };

  typedef typename boost::mpl::if_c<WithPartitioning,ProcessWithPartitioning,ProcessWithoutPartitioning>::type Process;

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    //typedef PerThreadChunkFIFO<64> Chunk;
    //typedef dChunkedFIFO<64> Chunk;
    //typedef dChunkedFIFO<64> Chunk;
    typedef dChunkedFIFO<128> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Chunk, 10> OBIM;

    std::cout << "INFO: Using delta-step of " << (1 << deltaShift) << " (" << deltaShift << ")\n";
    if (WithPartitioning) {
      std::cout << "INFO: Using block-step of " << (1 << blockShift) << " (" << blockShift << ")\n";
      std::cout << "INFO: Using partition size of " << (1 << partitionShift) << " (" << partitionShift << ")\n";
    }

    int total = 1;
    if (WithPartitioning) 
      total = (graph.size() + (1 << partitionShift) - 1) / (1 << partitionShift);
    if (WithPartitioning) {
      std::cout << "INFO: Number of partitions " << total << "\n";
    }

    int rangeScale = 0; // XXX unused 
    int cur = 0;
    Bags bags;
    for (int i = 0; i < total; ++i)
      bags.emplace_back();

    AsyncAlgo* self = this;
    Node& sourceData = graph.getData(source);
    sourceData.dist = 0;
    galois::do_all(
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(),
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end(),
        [&](typename Graph::edge_iterator ii) {
          if (WithPartitioning)
            self->relaxEdge(graph, sourceData, ii, rangeScale, bags);
          else
            self->relaxEdge(graph, sourceData, ii, bags[0]);
        });

    galois::Statistic rounds("Rounds");
    galois::StatTimer clearTime("ClearTime");
    Bag next;
    for (int count = 0; count < total; cur = (cur + 1) % total) {
      if (bags[cur].empty()) {
        ++count;
        continue;
      } else {
        count = 0;
      }

      galois::for_each(bags[cur], Process(this, graph, rangeScale, cur, next, bags), galois::wl<OBIM>());
      rounds += 1;

      if (!WithPartitioning)
        break;

      clearTime.start();
      bags[cur].clear();
      if (!next.empty()) {
        galois::do_all(next, [&](UpdateRequest& req) { bags[cur].push(req); });
        next.clear();
      }
      clearTime.stop();
    }
  }
};

struct Algo2 {
  typedef SNode Node;

  typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::with_out_of_line_lockable<true>
    ::with_compressed_node_ptr<true>
    //::template with_numa_alloc<!WithPartitioning> XXX
    Graph;
  typedef typename Graph::GraphNode GNode;

  struct UpdateRequest: public UpdateRequestCommon<GNode> {
    int partition;
    UpdateRequest(const GNode& n, Dist w, int p): UpdateRequestCommon<GNode>(n, w), partition(p) { }
    UpdateRequest(): UpdateRequestCommon<GNode>(), partition(0) { }
  };

  typedef galois::InsertBag<UpdateRequest> Bag;

  struct Partitioner {
    int operator()(const UpdateRequest& r) { return r.partition; }
  };

  std::string name() const { return "algo2"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  template<typename Pusher>
  void relaxEdge(Graph& graph, Node& sdata, Graph::edge_iterator ii, Pusher& pusher) {
    GNode dst = graph.getEdgeDst(ii);
    // int id = graph.idFromNode(dst) / scale;
    int id = graph.idFromNode(dst) >> partitionShift;

    Dist d = graph.getEdgeData(ii);
    Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist newDist = sdata.dist + d;
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        pusher.push(UpdateRequest(dst, newDist, id));
        break;
      }
    }
  }

  template<typename Pusher>
  void relaxNode(Graph& graph, GNode src, Dist prevDist, Pusher& pusher) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Node& sdata = graph.getData(src, flag);

    if (prevDist > sdata.dist) {
      if (trackBadWork)
        *WLEmptyWork += 1;
      return;
    }

    for (Graph::edge_iterator ii = graph.edge_begin(src, flag), ei = graph.edge_end(src, flag); ii != ei; ++ii) {
      relaxEdge(graph, sdata, ii, pusher);
    }
  }

  struct Process {
    typedef int tt_does_not_need_aborts;

    Algo2* self;
    Graph& graph;

    Process(Algo2* s, Graph& g): self(s), graph(g) { }
    void operator()(UpdateRequest& req, galois::UserContext<UpdateRequest>& ctx) {
      self->relaxNode(graph, req.n, req.w, ctx);
    }
  };

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    //typedef PerThreadChunkFIFO<64> Chunk;
    //typedef dChunkedFIFO<64> Chunk;
    //typedef dChunkedFIFO<64> Chunk;
    const int blockPeriod = 10;
    const int maxValue = 128;
    typedef dChunkedFIFO<128> Chunk;
    typedef Partitioned<Partitioner, Chunk>
      ::with_block_period<blockPeriod>
      ::with_max_value<maxValue> Part;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Part, 10> OBIM;

    typedef galois::InsertBag<UpdateRequest> Bag;

    int total = (graph.size() + (1 << partitionShift) - 1) / (1 << partitionShift);

    std::cout << "INFO: Using delta-step of " << (1 << deltaShift) << " (" << deltaShift << ")\n";
    std::cout << "INFO: Using block-step of " << (1 << blockPeriod) << " (" << blockPeriod << ")\n";
    std::cout << "INFO: Using partition size of " << (1 << partitionShift) << " (" << partitionShift << ")\n";
    std::cout << "INFO: Number of partitions " << total << " (max: " << maxValue << ")\n";

    Bag bag;

    Algo2* self = this;
    Node& sourceData = graph.getData(source);
    sourceData.dist = 0;
    galois::do_all(
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(),
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end(),
        [&](Graph::edge_iterator ii) {
            self->relaxEdge(graph, sourceData, ii, bag);
        });

    galois::for_each(bag, Process(this, graph), galois::wl<OBIM>());
  }
};

struct Algo3 {
  typedef SNode Node;

  typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::with_out_of_line_lockable<true>
    ::with_compressed_node_ptr<true>
    //::template with_numa_alloc<!WithPartitioning> XXX
    Graph;
  typedef typename Graph::GraphNode GNode;

  struct UpdateRequest: public UpdateRequestCommon<GNode> {
    Graph::edge_iterator begin;

    UpdateRequest(const GNode& n, Dist w, Graph::edge_iterator b):
      UpdateRequestCommon<GNode>(n, w), begin(b) { }

    UpdateRequest(const GNode& n, Dist w): UpdateRequestCommon<GNode>(n, w) { }
  };

  typedef galois::InsertBag<UpdateRequest> Bag;

  std::string name() const { return "algo3"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  template<typename Pusher>
  void relaxEdge(Graph& graph, Node& sdata, const GNode& dst, const Dist& d, Pusher& pusher) {
    Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist newDist = sdata.dist + d;
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        pusher.push(UpdateRequest(dst, newDist));
        break;
      }
    }
  }

  template<typename Pusher1, typename Pusher2>
  void relaxNode(Graph& graph, GNode src, Dist prevDist, int curId, Graph::edge_iterator begin, Pusher1& pusher1, Pusher2& pusher2) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Node& sdata = graph.getData(src, flag);

    if (prevDist > sdata.dist) {
      if (trackBadWork)
        *WLEmptyWork += 1;
      return;
    }

    for (Graph::edge_iterator ii = begin, ei = graph.edge_end(src, flag); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      int id = graph.idFromNode(dst) >> partitionShift;
      if (id == curId) {
        relaxEdge(graph, sdata, dst, graph.getEdgeData(ii), pusher2);
      } else if (id > curId) {
        //pusher1.push(UpdateRequest(src, prevDist, ii));
        pusher1.push(UpdateRequest(src, graph.getData(src, flag).dist, ii));
        break;
      }
    }
  }

  template<bool First>
  struct Process {
    typedef int tt_does_not_need_aborts;

    Algo3* self;
    Graph& graph;
    int id;
    Bag* innerBag;
    Bag* bag;

    Process(Algo3* s, Graph& g, int i, Bag* ib, Bag* b): self(s), graph(g), id(i), innerBag(ib), bag(b) { }

    void operator()(UpdateRequest& req, galois::UserContext<UpdateRequest>&) {
      if (First)
        self->relaxNode(graph, req.n, req.w, id, graph.edge_begin(req.n, galois::MethodFlag::UNPROTECTED), *innerBag, *bag);
      else
        self->relaxNode(graph, req.n, req.w, id, req.begin, *innerBag, *bag);
    }
  };


  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    //const int blockPeriod = 10;
    const int maxValue = 128;
    typedef dChunkedFIFO<128> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Chunk, 0> OBIM;

    typedef galois::InsertBag<UpdateRequest> Bag;

    int total = (graph.size() + (1 << partitionShift) - 1) / (1 << partitionShift);

    std::cout << "INFO: Using delta-step of " << (1 << deltaShift) << " (" << deltaShift << ")\n";
    std::cout << "INFO: Using partition size of " << (1 << partitionShift) << " (" << partitionShift << ")\n";
    std::cout << "INFO: Number of partitions " << total << " (max: " << maxValue << ")\n";

    Bag bag[3];
    Bag* cur = &bag[0], *next = &bag[1], *innerNext = &bag[2];

    Algo3* self = this;
    Node& sourceData = graph.getData(source);
    sourceData.dist = 0;
    galois::do_all(
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(),
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end(),
        [&](Graph::edge_iterator ii) {
            self->relaxEdge(graph, sourceData, graph.getEdgeDst(ii), graph.getEdgeData(ii), *cur);
        });

    galois::Statistic rounds("Rounds");
    galois::StatTimer clearTime("ClearTime");
    while (!cur->empty()) {
      galois::TimeAccumulator t;
      t.start();
      for (int i = 0; i < total; ++i) {
        if (cur->empty())
          break;
        std::cout << std::distance(cur->begin(), cur->end()) << " ";
        if (i == 0)
          galois::for_each(*cur, Process<true>(this, graph, i, innerNext, next), galois::wl<OBIM>());
        else
          galois::for_each(*cur, Process<false>(this, graph, i, innerNext, next), galois::wl<OBIM>());
        clearTime.start();
        cur->clear();
        clearTime.stop();
        std::swap(cur, innerNext);
      }
      t.stop();
      std::cout << "Time " << t.get() << "\n";
      clearTime.start();
      cur->clear();
      clearTime.stop();
      std::swap(cur, next);
      rounds += 1;
    }
  }
};

struct Algo4 {
  typedef SNode Node;

  typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::with_no_lockable<true>
    ::with_compressed_node_ptr<true>
    //::template with_numa_alloc<!WithPartitioning> XXX
    Graph;
  typedef typename Graph::GraphNode GNode;

  struct UpdateRequest {
    GNode src;
    Graph::edge_iterator begin;
    Dist w;
    int part;

    UpdateRequest(GNode s, Dist _w, Graph::edge_iterator b, int p):
      src(s), begin(b), w(_w), part(p) { }
  };

  struct Partitioner {
    int operator()(const UpdateRequest& r) { return r.part; }
  };

  typedef galois::InsertBag<UpdateRequest> Bag;

  std::string name() const { return "algo4"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  template<typename Pusher>
  void relaxEdge(Graph& graph, const GNode& src, const GNode& dst, const Dist& newDist, Pusher& pusher) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Node& ddata = graph.getData(dst, flag);
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        pusher.push(UpdateRequest(dst, newDist, graph.edge_begin(dst, flag), 0));
        return;
      }
    }
  }

  template<typename Pusher1, typename Pusher2>
  void relaxNode(Graph& graph, const UpdateRequest& req, Pusher1& pusher1, Pusher2& pusher2) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

    int srcPart = graph.idFromNode(req.src) >> partitionShift;
    const bool check = true || srcPart == req.part;
    volatile Dist* srcDist = &graph.getData(req.src, flag).dist;

    if (check && *srcDist != req.w) {
    //if (*srcDist != req.w) {
      if (trackBadWork)
        *WLEmptyWork += 1;
      return;
    }

    for (Graph::edge_iterator ii = req.begin, ei = graph.edge_end(req.src, flag); ii != ei; ++ii) {
      if (check && *srcDist != req.w) {
        if (trackBadWork)
          *WLEmptyWork += 1;
        return;
      }
      GNode dst = graph.getEdgeDst(ii);
      int dstPart = graph.idFromNode(dst) >> partitionShift;
      //int p = dst >> partitionShift;

      if (dstPart == req.part || req.part > 0) {
      //if (dstPart == req.part) {
        relaxEdge(graph, req.src, dst, req.w + graph.getEdgeData(ii), pusher2);
      } else if (dstPart > req.part) {
        pusher1.push(UpdateRequest(req.src, req.w, ii, dstPart));
        break;
      }
    }
  }

  struct Process {
    typedef int tt_does_not_need_aborts;

    Algo4* self;
    Graph& graph;
    Bag* bag;

    Process(Algo4* s, Graph& g, Bag* b): self(s), graph(g), bag(b) { }

    void operator()(const UpdateRequest& req, galois::UserContext<UpdateRequest>& ctx) {
      self->relaxNode(graph, req, ctx, ctx);
    }
  };

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    const int blockPeriod = 10;
    const int maxValue = 8;
    typedef dChunkedFIFO<128> Chunk;
    typedef Partitioned<Partitioner, Chunk>
      ::with_block_period<blockPeriod>
      ::with_max_value<maxValue> Part;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Part, 10> OBIM;

    typedef galois::InsertBag<UpdateRequest> Bag;

    int total = (graph.size() + (1 << partitionShift) - 1) / (1 << partitionShift);

    std::cout << "INFO: Using delta-step of " << (1 << deltaShift) << " (" << deltaShift << ")\n";
    std::cout << "INFO: Using partition size of " << (1 << partitionShift) << " (" << partitionShift << ")\n";
    std::cout << "INFO: Number of partitions " << total << " (max: " << maxValue << ")\n";

    Bag bag[2];
    Bag* cur = &bag[0], *next = &bag[1];

    Algo4* self = this;
    Node& sourceData = graph.getData(source);
    sourceData.dist = 0;
    galois::do_all(
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(),
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end(),
        [&](Graph::edge_iterator ii) {
            self->relaxEdge(graph, source, graph.getEdgeDst(ii), sourceData.dist + graph.getEdgeData(ii), *cur);
        });

    galois::for_each(*cur, Process(this, graph, next), galois::wl<OBIM>());
  }
};

struct Algo5 {
  typedef SNode Node;

  typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::with_no_lockable<true>
    ::with_compressed_node_ptr<true>
    //::template with_numa_alloc<!WithPartitioning> XXX
    Graph;
  typedef typename Graph::GraphNode GNode;

  struct UpdateRequest {
    GNode src;
    Graph::edge_iterator begin;
    Dist w;
    int part;

    UpdateRequest(GNode s, Dist _w, Graph::edge_iterator b, int p):
      src(s), begin(b), w(_w), part(p) { }
  };

  struct Partitioner {
    int operator()(const UpdateRequest& r) { return r.part; }
  };

  typedef galois::InsertBag<UpdateRequest> Bag;

  std::string name() const { return "algo5"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  template<typename Pusher>
  void relaxEdge(Graph& graph, const GNode& src, const GNode& dst, const Dist& newDist, Pusher& pusher) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Node& ddata = graph.getData(dst, flag);
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
#if 1
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        pusher.push(UpdateRequest(dst, newDist, graph.edge_begin(dst, flag), 0));
        return;
      }
#else
      ddata.dist = newDist;
      if (trackBadWork && oldDist != DIST_INFINITY)
        *BadWork += 1;
      pusher.push(UpdateRequest(dst, newDist, graph.edge_begin(dst, flag), 0));
      return;
#endif
    }
  }

  template<typename Pusher1, typename Pusher2>
  void relaxNode(Graph& graph, const UpdateRequest& req, Pusher1& pusher1, Pusher2& pusher2) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

    int srcPart = graph.idFromNode(req.src) >> partitionShift;
    const bool check = true || srcPart == req.part;
    volatile Dist* srcDist = &graph.getData(req.src, flag).dist;

    if (check && *srcDist != req.w) {
    //if (*srcDist != req.w) {
      if (trackBadWork)
        *WLEmptyWork += 1;
      return;
    }

    for (Graph::edge_iterator ii = req.begin, ei = graph.edge_end(req.src, flag); ii != ei; ++ii) {
      if (check && *srcDist != req.w) {
        if (trackBadWork)
          *WLEmptyWork += 1;
        return;
      }
      GNode dst = graph.getEdgeDst(ii);
      int dstPart = graph.idFromNode(dst) >> partitionShift;
      //int p = dst >> partitionShift;

      if (dstPart == req.part) {
      //if (dstPart == req.part) {
        relaxEdge(graph, req.src, dst, req.w + graph.getEdgeData(ii), pusher2);
      } else if (dstPart > req.part) {
        pusher1.push(UpdateRequest(req.src, req.w, ii, dstPart));
        break;
      }
    }
  }

  struct Process {
    typedef int tt_does_not_need_aborts;

    Algo5* self;
    Graph& graph;
    Bag* bag;

    Process(Algo5* s, Graph& g, Bag* b): self(s), graph(g), bag(b) { }

    void operator()(const UpdateRequest& req, galois::UserContext<UpdateRequest>& ctx) {
      self->relaxNode(graph, req, ctx, ctx);
    }
  };

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    //const int blockPeriod = 10;
    //const int maxValue = 8;
    typedef dChunkedFIFO<128> Chunk;
    typedef ThreadPartitioned<Partitioner, Chunk> Part;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Part, 10> OBIM;

    typedef galois::InsertBag<UpdateRequest> Bag;

    int total = (graph.size() + (1 << partitionShift) - 1) / (1 << partitionShift);

    std::cout << "INFO: Using delta-step of " << (1 << deltaShift) << " (" << deltaShift << ")\n";
    std::cout << "INFO: Using partition size of " << (1 << partitionShift) << " (" << partitionShift << ")\n";
    std::cout << "INFO: Number of partitions " << total << "\n";

    Bag bag[2];
    Bag* cur = &bag[0], *next = &bag[1];

    Algo5* self = this;
    Node& sourceData = graph.getData(source);
    sourceData.dist = 0;
    std::for_each(
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(),
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end(),
        [&](Graph::edge_iterator ii) {
            self->relaxEdge(graph, source, graph.getEdgeDst(ii), sourceData.dist + graph.getEdgeData(ii), *cur);
        });

    galois::for_each(*cur, Process(this, graph, next), galois::wl<OBIM>());
    //galois::for_each<Part>(*cur, Process(this, graph, next));
  }
};

struct Algo6 {
  typedef SNode Node;

  typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::with_no_lockable<true>
    ::with_compressed_node_ptr<true>
    //::template with_numa_alloc<!WithPartitioning> XXX
    Graph;
  typedef typename Graph::GraphNode GNode;
  typedef UpdateRequestCommon<GNode> UpdateRequest;
  typedef galois::InsertBag<UpdateRequest> Bag;

  std::deque<Graph> parts;

  std::string name() const { return "algo6"; }

  void readGraphParts(const std::string& name) {
    // partitioned files are in form xxx.gr.i.of.n
    // TODO: regex not fully supported everywhere, parse by hand for now
    int state = 0;
    size_t index = name.length();
    int numParts = 0;
    std::string base;
    for (std::string::const_reverse_iterator ii = name.rbegin(), ei = name.rend(); ii != ei; ++ii, --index) {
      if (state == 0) {
        if (isdigit(*ii)) {
          continue;
        } else {
          state = 1;
          numParts = stoi(name.substr(index));
        }
      } 
      if (state == 1 && index > 4) {
        if (*ii++ == '.' && *ii++ == 'f' && *ii++ == 'o' && *ii++ == '.') {
          state = 2;
          index -= 4;
        } else {
          GALOIS_DIE("unknown name for partitioned graph");
        }
      } 
      if (state == 2) {
        if (isdigit(*ii)) {
          continue;
        } else {
          base = name.substr(0, index);
          break;
        }
      }
    }

    for (int i = 0; i < numParts; ++i) {
      std::ostringstream os;
      os << base << i << ".of." << numParts;
      parts.emplace_back();
      galois::graphs::readGraph(parts.back(), os.str());
    }
  }

  void readGraph(Graph& graph) {
    galois::graphs::readGraph(graph, filename);
    readGraphParts(partsname);
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      g.getData(n, galois::MethodFlag::UNPROTECTED).dist = DIST_INFINITY;
    }
  };

  template<typename Pusher>
  void relaxEdge(Graph& graph, const GNode& src, const GNode& dst, const Dist& newDist, Pusher& pusher) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Node& ddata = graph.getData(dst, flag);
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
#if 1
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        pusher.push(UpdateRequest(dst, newDist));
        return;
      }
#else
      ddata.dist = newDist;
      if (trackBadWork && oldDist != DIST_INFINITY)
        *BadWork += 1;
      pusher.push(UpdateRequest(dst, newDist));
      return;
#endif
    }
  }

  template<typename Pusher1, typename Pusher2>
  void relaxNode(Graph& graph, const UpdateRequest& req, Pusher1& pusher1, Pusher2& pusher2) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

    const bool check = true;
    volatile Dist* srcDist = &graph.getData(req.n, flag).dist;

    if (check && *srcDist != req.w) {
      if (trackBadWork)
        *WLEmptyWork += 1;
      return;
    }

    for (Graph::edge_iterator ii = graph.edge_begin(req.n, flag), ei = graph.edge_end(req.n, flag); ii != ei; ++ii) {
      if (check && *srcDist != req.w) {
        if (trackBadWork)
          *WLEmptyWork += 1;
        return;
      }
      GNode dst = graph.getEdgeDst(ii);
      relaxEdge(graph, req.n, dst, req.w + graph.getEdgeData(ii), pusher2);
    }
  }

  struct Process {
    typedef int tt_does_not_need_aborts;

    Algo6* self;
    Graph& graph;
    Bag* bag;

    Process(Algo6* s, Graph& g, Bag* b): self(s), graph(g), bag(b) { }

    void operator()(const UpdateRequest& req, galois::UserContext<UpdateRequest>& ctx) {
      self->relaxNode(graph, req, ctx, ctx);
    }
  };

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    typedef dChunkedFIFO<128> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Chunk, 10> OBIM;

    typedef galois::InsertBag<UpdateRequest> Bag;

    std::cout << "INFO: Using delta-step of " << (1 << deltaShift) << " (" << deltaShift << ")\n";
    std::cout << "INFO: Number of partitions " << parts.size() << "\n";

    galois::StatTimer initTime("ExtraInitTime");
    initTime.start();
    for (Graph& g : parts)
      galois::do_all(g, Initialize(g));
    initTime.stop();

    // XXX compute dst ranges

    Bag bag[2];
    Bag* cur = &bag[0], *next = &bag[1];

    Algo6* self = this;
    GNode source0 = parts[0].nodeFromId(graph.idFromNode(source));
    Node& sourceData = parts[0].getData(source0);
    sourceData.dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source, galois::MethodFlag::UNPROTECTED),
        ei = graph.edge_end(source, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Dist newDist = graph.getEdgeData(ii);
      GNode dst0 = parts[0].nodeFromId(graph.idFromNode(dst));
      parts[0].getData(dst0).dist = newDist;
      cur->push(UpdateRequest(dst0, newDist));
    }

    while (true) {
      galois::TimeAccumulator wholeTime;
      galois::TimeAccumulator mainTime;

      wholeTime.start();
      mainTime.start();
      galois::for_each(*cur, Process(this, parts[0], next), galois::wl<OBIM>());
      mainTime.stop();
      //std::cout << "(" << std::distance(cur->begin(), cur->end()) << ", " << mainTime.get() << ") ";

      for (unsigned int i = 1; i < parts.size(); ++i) {
        cur->clear();
        // XXX for dst range add outgoing 
        galois::do_all(self->parts[i], [&](GNode nn) {
          GNode n = self->parts[i-1].nodeFromId(self->parts[i].idFromNode(nn));
          Dist newDist = self->parts[i-1].getData(n).dist;
          if (newDist != self->parts[i].getData(nn).dist) {
            self->parts[i].getData(nn).dist = newDist;
            if (self->parts[i].edge_begin(nn, galois::MethodFlag::UNPROTECTED) != self->parts[i].edge_end(nn, galois::MethodFlag::UNPROTECTED))
              cur->push(UpdateRequest(nn, newDist));
          }
        });
        mainTime.start();
        galois::for_each(*cur, Process(this, parts[i], next), galois::wl<OBIM>());
        mainTime.stop();
        //std::cout << "(" << std::distance(cur->begin(), cur->end()) << ", " << mainTime.get() << ") ";
      }

      for (int i = 0; i < 1; ++i) {
        cur->clear();
        int t = parts.size() - 1;
        galois::do_all(parts[i], [&](GNode nn) {
          GNode n = self->parts[t].nodeFromId(self->parts[i].idFromNode(nn));
          Dist newDist = self->parts[t].getData(n).dist;
          if (newDist != self->parts[i].getData(nn).dist) {
            self->parts[i].getData(nn).dist = newDist;
            if (self->parts[i].edge_begin(nn, galois::MethodFlag::UNPROTECTED) != self->parts[i].edge_end(nn, galois::MethodFlag::UNPROTECTED))
              cur->push(UpdateRequest(nn, newDist));
          }
        });
      }
      wholeTime.stop();
      std::cout << "Time " << wholeTime.get() << " " << mainTime.get() << "\n";

      // XXX Done flag
      if (cur->empty())
        break;
    }

    galois::StatTimer copyOutTime("CopyOutTime");
    copyOutTime.start();
    galois::do_all(parts[0], [&](GNode n0) {
      GNode n = graph.nodeFromId(self->parts[0].idFromNode(n0));
      graph.getData(n).dist = self->parts[0].getData(n0).dist;
    });
    copyOutTime.stop();
  }
};

struct Algo7 {
  struct Node : public SNode {
    int start;
  };

  typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::with_no_lockable<true>
    ::with_compressed_node_ptr<true>
    ::with_numa_alloc<true>
    Graph;
  typedef typename Graph::GraphNode GNode;
  typedef UpdateRequestCommon<GNode> UpdateRequest;

  struct UpdateRequest2: public UpdateRequestCommon<GNode> {
    typename Graph::edge_iterator begin;
    UpdateRequest2(GNode n, Dist w, typename Graph::edge_iterator b): UpdateRequestCommon<GNode>(n, w), begin(b) { }
  };

  std::string name() const { return "algo7"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      Node& node = g.getData(n, galois::MethodFlag::UNPROTECTED);
      node.dist = DIST_INFINITY;
      node.start = std::numeric_limits<int>::max();
    }
  };

  template<typename Pusher>
  void relaxEdge(Graph& graph, typename Graph::edge_iterator ii, const Dist& newDist, Pusher& pusher) {
    GNode dst = graph.getEdgeDst(ii);
    Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        pusher.push(UpdateRequest(dst, newDist));
        break;
      }
    }
  }

  template<typename Pusher>
  void relaxEdge2(Graph& graph, typename Graph::edge_iterator ii, const Dist& newDist, Pusher& pusher) {
    GNode dst = graph.getEdgeDst(ii);
    Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
    Dist oldDist;
    while (newDist < (oldDist = ddata.dist)) {
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        if (trackBadWork && oldDist != DIST_INFINITY)
          *BadWork += 1;
        if (ddata.start == -1)
          break;

        typename Graph::edge_iterator next = graph.edge_begin(dst, galois::MethodFlag::UNPROTECTED);
        if (ddata.start != std::numeric_limits<int>::max())
          std::advance(next, ddata.start);
        pusher.push(UpdateRequest2(dst, newDist, next));
        break;
      }
    }
  }

  template<bool UseContinue, typename Pusher1, typename Pusher2>
  void relaxNode(Graph& graph, const UpdateRequest& req, typename Graph::edge_iterator begin, Pusher1* pusher1, Pusher2& pusher2) {
  }

  struct Process {
    typedef int tt_does_not_need_aborts;

    Algo7* self;
    Graph& graph;
    galois::InsertBag<UpdateRequest2>* bag;

    Process(Algo7* s, Graph& g, galois::InsertBag<UpdateRequest2>* b): self(s), graph(g), bag(b) { }

    void updateStart(Node& sdata, int newStart) {
      int oldStart;
      while (newStart < (oldStart = sdata.start)) {
        if (__sync_bool_compare_and_swap(&sdata.start, oldStart, newStart)) {
          break;
        }
      }
    }

    void operator()(const UpdateRequest& req, galois::UserContext<UpdateRequest>& ctx) {
      const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
      Node& sdata = graph.getData(req.n, flag);
      volatile Dist* sdist = &sdata.dist;

      if (req.w != *sdist) {
        if (trackBadWork)
          *WLEmptyWork += 1;
        return;
      }

      int c = 0;
      for (typename Graph::edge_iterator ii = graph.edge_begin(req.n, flag), ei = graph.edge_end(req.n, flag); ii != ei; ++ii, ++c) {
        if (req.w != *sdist) {
          if (trackBadWork)
            *WLEmptyWork += 1;
          return;
        }
        Dist newDist = sdata.dist + graph.getEdgeData(ii);
        if ((newDist >> deltaShift) < partitionShift) {
          self->relaxEdge(graph, ii, newDist, ctx);
        } else {
          updateStart(sdata, c);
          bag->push(UpdateRequest2(req.n, *sdist, ii));
          return;
        }
      }
      updateStart(sdata, -1);
    }

    void operator()(const UpdateRequest2& req, galois::UserContext<UpdateRequest2>& ctx) {
      const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
      Node& sdata = graph.getData(req.n, flag);
      volatile Dist* sdist = &sdata.dist;

      if (req.w != *sdist) {
        if (trackBadWork)
          *WLEmptyWork += 1;
        return;
      }

      for (typename Graph::edge_iterator ii = req.begin, ei = graph.edge_end(req.n, flag); ii != ei; ++ii) {
        if (req.w != *sdist) {
          if (trackBadWork)
            *WLEmptyWork += 1;
          break;
        }
        Dist newDist = sdata.dist + graph.getEdgeData(ii);
        self->relaxEdge2(graph, ii, newDist, ctx);
      }
    }
  };

  void operator()(Graph& graph, GNode source) {
    using namespace galois::worklists;
    typedef dChunkedFIFO<128> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer<UpdateRequest>, Chunk, 10> OBIM;

    std::cout << "INFO: Using delta-step of " << (1 << deltaShift) << " (" << deltaShift << ")\n";
    std::cout << "INFO: Using partition shift of " << partitionShift << "\n";
    std::cout << "INFO: Using block shift of " << blockShift << "\n";

    galois::InsertBag<UpdateRequest> initial;
    galois::InsertBag<UpdateRequest2> next;

    Algo7* self = this;
    Node& sourceData = graph.getData(source);
    sourceData.dist = 0;
    galois::do_all(
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(),
        graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end(),
        [&](Graph::edge_iterator ii) {
            self->relaxEdge(graph, ii, sourceData.dist + graph.getEdgeData(ii), initial);
        });
    galois::for_each(initial, Process(this, graph, &next), galois::loopname("A1"), galois::wl<OBIM>());
    galois::for_each(next, Process(this, graph, nullptr), galois::loopname("A2"), galois::wl<OBIM>());
  }
};



template<typename A>
void run(bool prealloc = true) {
  typedef typename A::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  A algo;
  Graph graph;
  GNode source, report;

  initialize(algo, graph, source, report);

  size_t approxNodeData = graph.size() * 64;
  //size_t approxEdgeData = graph.sizeEdges() * sizeof(typename Graph::edge_data_type) * 2;
  if (prealloc)
    galois::preAlloc(numThreads + approxNodeData / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  galois::do_all(graph, typename A::Initialize(graph));
  algo(graph, source);
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");
  galois::runtime::reportNumaAlloc("NumaPost");

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
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (trackBadWork) {
    BadWork = new galois::Statistic("BadWork");
    WLEmptyWork = new galois::Statistic("EmptyWork");
  }

  blockStep = 1 << blockShift;

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::normal: run<AsyncAlgo<false> >(); break;
    case Algo::part: run<AsyncAlgo<true> >(); break;
    case Algo::part2: run<Algo2>(); break;
    case Algo::part3: run<Algo3>(); break;
    case Algo::part4: run<Algo4>(); break;
    case Algo::part5: run<Algo5>(); break;
    case Algo::part6: run<Algo6>(); break;
    case Algo::part7: run<Algo7>(); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  if (trackBadWork) {
    delete BadWork;
    delete WLEmptyWork;
  }

  return 0;
}

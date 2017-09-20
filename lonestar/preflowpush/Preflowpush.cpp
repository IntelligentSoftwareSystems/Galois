/** Preflow-push application -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Bag.h"
#include "Galois/Graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/iterator_adaptor.hpp>

#include <iostream>
#include <fstream>

namespace cll = llvm::cl;

const char* name = "Preflow Push";
const char* desc = "Finds the maximum flow in a network using the preflow push technique";
const char* url = "preflow_push";

enum DetAlgo {
  nondet,
  detBase,
  detDisjoint
};

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<uint32_t> sourceId(cll::Positional, cll::desc("sourceID"), cll::Required);
static cll::opt<uint32_t> sinkId(cll::Positional, cll::desc("sinkID"), cll::Required);
static cll::opt<bool> useHLOrder("useHLOrder", cll::desc("Use HL ordering heuristic"), cll::init(false));
static cll::opt<bool> useUnitCapacity("useUnitCapacity", cll::desc("Assume all capacities are unit"), cll::init(false));
static cll::opt<bool> useSymmetricDirectly("useSymmetricDirectly",
    cll::desc("Assume input graph is symmetric and has unit capacities"), cll::init(false));
static cll::opt<int> relabelInt("relabel",
    cll::desc("relabel interval: < 0 no relabeling, 0 use default interval, > 0 relabel every X iterations"), cll::init(0));
static cll::opt<DetAlgo> detAlgo(cll::desc("Deterministic algorithm:"),
    cll::values(
      clEnumVal(nondet, "Non-deterministic"),
      clEnumVal(detBase, "Base execution"),
      clEnumVal(detDisjoint, "Disjoint execution"),
      clEnumValEnd), cll::init(nondet));

/**
 * Alpha parameter the original Goldberg algorithm to control when global
 * relabeling occurs. For comparison purposes, we keep them the same as
 * before, but it is possible to achieve much better performance by adjusting
 * the global relabel frequency.
 */
const int ALPHA = 6;

/**
 * Beta parameter the original Goldberg algorithm to control when global
 * relabeling occurs. For comparison purposes, we keep them the same as
 * before, but it is possible to achieve much better performance by adjusting
 * the global relabel frequency.
 */
const int BETA = 12;

struct Node {
  uint32_t id;
  int64_t excess;
  int height;
  int current;

  Node() : excess(0), height(1), current(0) { }
};

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "("
     << "id: " << n.id
     << ", excess: " << n.excess
     << ", height: " << n.height
     << ", current: " << n.current
     << ")";
  return os;
}

typedef galois::graphs::LC_Linear_Graph<Node, int32_t>::with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode;

struct Config {
  Graph graph;
  GNode sink;
  GNode source;
  int global_relabel_interval;
  bool should_global_relabel;
  Config() : should_global_relabel(false) {}
};

Config app;

struct Indexer :std::unary_function<GNode, int> {
  int operator()(const GNode& n) const {
    return -app.graph.getData(n, galois::MethodFlag::UNPROTECTED).height;
  }
};

struct GLess :std::binary_function<GNode, GNode, bool> {
  bool operator()(const GNode& lhs, const GNode& rhs) const {
    int lv = -app.graph.getData(lhs, galois::MethodFlag::UNPROTECTED).height;
    int rv = -app.graph.getData(rhs, galois::MethodFlag::UNPROTECTED).height;
    return lv < rv;
  }
};
struct GGreater :std::binary_function<GNode, GNode, bool> {
  bool operator()(const GNode& lhs, const GNode& rhs) const {
    int lv = -app.graph.getData(lhs, galois::MethodFlag::UNPROTECTED).height;
    int rv = -app.graph.getData(rhs, galois::MethodFlag::UNPROTECTED).height;
    return lv > rv;
  }
};

void checkAugmentingPath() {
  // Use id field as visited flag
  for (Graph::iterator ii = app.graph.begin(),
      ee = app.graph.end(); ii != ee; ++ii) {
    GNode src = *ii;
    app.graph.getData(src).id = 0;
  }

  std::deque<GNode> queue;

  app.graph.getData(app.source).id = 1;
  queue.push_back(app.source);

  while (!queue.empty()) {
    GNode& src = queue.front();
    queue.pop_front();
    for (auto ii : app.graph.edges(src)) {
      GNode dst = app.graph.getEdgeDst(ii);
      if (app.graph.getData(dst).id == 0
          && app.graph.getEdgeData(ii) > 0) {
        app.graph.getData(dst).id = 1;
        queue.push_back(dst);
      }
    }
  }

  if (app.graph.getData(app.sink).id != 0) {
    assert(false && "Augmenting path exisits");
    abort();
  }
}

void checkHeights() {
  for (Graph::iterator ii = app.graph.begin(),
      ei = app.graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    int sh = app.graph.getData(src).height;
    for (auto jj : app.graph.edges(src)) {
      GNode dst = app.graph.getEdgeDst(jj);
      int64_t cap = app.graph.getEdgeData(jj);
      int dh = app.graph.getData(dst).height;
      if (cap > 0 && sh > dh + 1) {
        std::cerr << "height violated at " << app.graph.getData(src) << "\n";
        abort();
      }
    }
  }
}


Graph::edge_iterator findEdgeLog2 (Graph& g, GNode dst, Graph::edge_iterator i, Graph::edge_iterator end_i) {
  
  struct EdgeDstIter: 
    public boost::iterator_facade<
      EdgeDstIter,
      GNode,
      boost::random_access_traversal_tag,
      GNode>
  {
    using Base = boost::iterator_facade<
      EdgeDstIter, 
      GNode,
      boost::random_access_traversal_tag,
      GNode>;

    Graph* g;
    Graph::edge_iterator ei;

    EdgeDstIter (void): g (nullptr)
    {}

    EdgeDstIter (Graph* g, Graph::edge_iterator ei):
      g (g), ei (ei)
    {}

  private:

    friend boost::iterator_core_access;

    GNode dereference (void) const {
      return g->getEdgeDst (ei);
    }

    void increment (void) {
      ++ei;
    }

    void decrement (void) {
      --ei;
    }

    bool equal (const EdgeDstIter& that) const {
      assert (this->g == that.g);
      return this->ei == that.ei;
    }

    void advance (ptrdiff_t n) {
      ei += n;
    }

    ptrdiff_t distance_to (const EdgeDstIter& that) const {
      assert (this->g == that.g);

      return that.ei - this->ei;
    }

  };
                      
  EdgeDstIter ai (&g, i);
  EdgeDstIter end_ai (&g, end_i);

  auto ret = std::lower_bound (ai, end_ai, dst);

  assert (ret != end_ai);
  assert (*ret == dst);

  return ret.ei;

}

Graph::edge_iterator findEdgeLinear (Graph& g, GNode dst, Graph::edge_iterator beg_e, Graph::edge_iterator end_e) {

  auto ii = beg_e;
  for (; ii != end_e; ++ii) {
    if (g.getEdgeDst(ii) == dst)
      break;
  }
  assert(ii != end_e); // Never return the end iterator
  return ii;
}

Graph::edge_iterator findEdge(Graph& g, GNode src, GNode dst) {

  auto i = g.edge_begin (src, galois::MethodFlag::UNPROTECTED);
  auto end_i = g.edge_end (src, galois::MethodFlag::UNPROTECTED);

  if ((end_i - i) < 32) { 
    return findEdgeLinear (g, dst, i, end_i);

  } else {
    return findEdgeLog2 (g, dst, i, end_i);

  }

}

void checkConservation(Config& orig) {
  std::vector<GNode> map;
  map.resize(app.graph.size());

  // Setup ids assuming same iteration order in both graphs
  uint32_t id = 0;
  for (Graph::iterator ii = app.graph.begin(),
      ei = app.graph.end(); ii != ei; ++ii, ++id) {
    app.graph.getData(*ii).id = id;
  }
  id = 0;
  for (Graph::iterator ii = orig.graph.begin(),
      ei = orig.graph.end(); ii != ei; ++ii, ++id) {
    orig.graph.getData(*ii).id = id;
    map[id] = *ii;
  }

  // Now do some checking
  for (Graph::iterator ii = app.graph.begin(), ei = app.graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    const Node& node = app.graph.getData(src);
    uint32_t srcId = node.id;

    if (src == app.source || src == app.sink)
      continue;

    if (node.excess != 0 && node.height != (int) app.graph.size()) {
      std::cerr << "Non-zero excess at " << node << "\n";
      abort();
    }

    int64_t sum = 0;
    for (auto jj : app.graph.edges(src)) {
      GNode dst = app.graph.getEdgeDst(jj);
      uint32_t dstId = app.graph.getData(dst).id;
      int64_t ocap = orig.graph.getEdgeData(findEdge(orig.graph, map[srcId], map[dstId]));
      int64_t delta = 0;
      if (ocap > 0) 
        delta -= (ocap - app.graph.getEdgeData(jj));
      else
        delta += app.graph.getEdgeData(jj);
      sum += delta;
    }

    if (node.excess != sum) {
      std::cerr << "Not pseudoflow: " << node.excess << " != " << sum << " at " << node << "\n";
      abort();
    }
  }
}

void verify(Config& orig) {
  // FIXME: doesn't fully check result
  checkHeights();
  checkConservation(orig);
  checkAugmentingPath();
}

void reduceCapacity(const Graph::edge_iterator& ii, const GNode& src, const GNode& dst, int64_t amount) {
  Graph::edge_data_type& cap1 = app.graph.getEdgeData(ii);
  Graph::edge_data_type& cap2 = app.graph.getEdgeData(findEdge(app.graph, dst, src));
  cap1 -= amount;
  cap2 += amount;
}

template<DetAlgo version,bool useCAS=true>
struct UpdateHeights {

  struct LocalState {
    LocalState(UpdateHeights<version,useCAS>& self, galois::PerIterAllocTy& alloc) { }
  };

  typedef std::tuple<
    galois::needs_per_iter_alloc<>,
    galois::has_deterministic_local_state<LocalState>
  > function_traits;

  //struct IdFn {
  //  unsigned long operator()(const GNode& item) const {
  //    return app.graph.getData(item, galois::MethodFlag::UNPROTECTED).id;
  //  }
  //};

  /**
   * Do reverse BFS on residual graph.
   */
  void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
    if (version != nondet) {

      if (ctx.isFirstPass()) {
        for (auto ii : app.graph.edges(src, galois::MethodFlag::WRITE)) {
          GNode dst = app.graph.getEdgeDst(ii);
          int64_t rdata = app.graph.getEdgeData(findEdge(app.graph, dst, src));
          if (rdata > 0) {
            app.graph.getData(dst, galois::MethodFlag::WRITE);
          }
        }
      }

      if (version == detDisjoint && ctx.isFirstPass()) {
          return;
      } else {
        app.graph.getData(src, galois::MethodFlag::WRITE);
        ctx.cautiousPoint();
      }
    }

    for (auto ii : app.graph.edges(src, useCAS ? galois::MethodFlag::UNPROTECTED : galois::MethodFlag::WRITE)) {
      GNode dst = app.graph.getEdgeDst(ii);
      int64_t rdata = app.graph.getEdgeData(findEdge(app.graph, dst, src));
      if (rdata > 0) {
        Node& node = app.graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        int newHeight = app.graph.getData(src, galois::MethodFlag::UNPROTECTED).height + 1;
        if (useCAS) {
          int oldHeight;
          while (newHeight < (oldHeight = node.height)) {
            if (__sync_bool_compare_and_swap(&node.height, oldHeight, newHeight)) {
              ctx.push(dst);
              break;
            }
          }
        } else {
          if (newHeight < node.height) {
            node.height = newHeight;
            ctx.push(dst);
          }
        }
      }
    }
  }
};

struct ResetHeights {
  void operator()(const GNode& src) const {
    Node& node = app.graph.getData(src, galois::MethodFlag::UNPROTECTED);
    node.height = app.graph.size();
    node.current = 0;
    if (src == app.sink)
      node.height = 0;
  }
};

template<typename WLTy>
struct FindWork {
  WLTy& wl;
  FindWork(WLTy& w) : wl(w) {}

  void operator()(const GNode& src) const {
    Node& node = app.graph.getData(src, galois::MethodFlag::UNPROTECTED);
    if (src == app.sink || src == app.source || node.height >= (int) app.graph.size())
      return;
    if (node.excess > 0) 
      wl.push_back(src);
  }
};

template<typename IncomingWL>
void globalRelabel(IncomingWL& incoming) {
  typedef galois::worklists::Deterministic<> DWL;

  galois::StatTimer T1("ResetHeightsTime");
  T1.start();
  galois::do_all_local(app.graph, ResetHeights(), galois::loopname("ResetHeights"));
  T1.stop();

  galois::StatTimer T("UpdateHeightsTime");
  T.start();

  switch (detAlgo) {
    case nondet:
      galois::for_each(app.sink, UpdateHeights<nondet>(), galois::loopname("UpdateHeights"), galois::wl<galois::worklists::BulkSynchronous<>>());
      //      galois::for_each(app.sink, UpdateHeights<nondet>(), galois::loopname("UpdateHeights"));
      break;
    case detBase:
      galois::for_each(app.sink, UpdateHeights<detBase>(), 
          galois::wl<DWL>(),
          galois::loopname("UpdateHeights"));
      break;
    case detDisjoint:
      galois::for_each(app.sink, UpdateHeights<detDisjoint>(),
          galois::wl<DWL>(),
          galois::loopname("UpdateHeights"));
      break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
  T.stop();

  galois::StatTimer T2("FindWorkTime");
  T2.start();
  galois::do_all_local(app.graph, FindWork<IncomingWL>(incoming), galois::loopname("FindWork"));
  T2.stop();
}

void acquire(const GNode& src) {
  // LC Graphs have a different idea of locking
  for (auto ii : app.graph.edges(src, galois::MethodFlag::WRITE)) {
    GNode dst = app.graph.getEdgeDst(ii);
    app.graph.getData(dst, galois::MethodFlag::WRITE);
  }
}

void relabel(const GNode& src) {
  int minHeight = std::numeric_limits<int>::max();
  int minEdge = 0;

  int current = 0;
  for (auto ii : app.graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
    GNode dst = app.graph.getEdgeDst(ii);
    int64_t cap = app.graph.getEdgeData(ii);
    if (cap > 0) {
      const Node& dnode = app.graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      if (dnode.height < minHeight) {
        minHeight = dnode.height;
        minEdge = current;
      }
    }
    ++current;
  }

  assert(minHeight != std::numeric_limits<int>::max());
  ++minHeight;

  Node& node = app.graph.getData(src, galois::MethodFlag::UNPROTECTED);
  if (minHeight < (int) app.graph.size()) {
    node.height = minHeight;
    node.current = minEdge;
  } else {
    node.height = app.graph.size();
  }
}

bool discharge(const GNode& src, galois::UserContext<GNode>& ctx) {
  //Node& node = app.graph.getData(src, galois::MethodFlag::WRITE);
  Node& node = app.graph.getData(src, galois::MethodFlag::UNPROTECTED);
  //int prevHeight = node.height;
  bool relabeled = false;

  if (node.excess == 0 || node.height >= (int) app.graph.size()) {
    return false;
  }

  while (true) {
    //galois::MethodFlag flag = relabeled ? galois::MethodFlag::UNPROTECTED : galois::MethodFlag::WRITE;
    galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    bool finished = false;
    int current = node.current;
    Graph::edge_iterator
      ii = app.graph.edge_begin(src, flag),
      ee = app.graph.edge_end(src, flag);
    std::advance(ii, node.current);
    for (; ii != ee; ++ii, ++current) {
      GNode dst = app.graph.getEdgeDst(ii);
      int64_t cap = app.graph.getEdgeData(ii);
      if (cap == 0)// || current < node.current) 
        continue;

      Node& dnode = app.graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      if (node.height - 1 != dnode.height) 
        continue;

      // Push flow
      int64_t amount = std::min(node.excess, cap);
      reduceCapacity(ii, src, dst, amount);

      // Only add once
      if (dst != app.sink && dst != app.source && dnode.excess == 0) 
        ctx.push(dst);
      
      assert (node.excess >= amount);
      node.excess -= amount;
      dnode.excess += amount;
      
      if (node.excess == 0) {
        finished = true;
        node.current = current;
        break;
      }
    }

    if (finished)
      break;

    relabel(src);
    relabeled = true;

    if (node.height == (int) app.graph.size())
      break;

    //prevHeight = node.height;
  }

  return relabeled;
}

struct Counter {
  galois::GAccumulator<int> accum;
  galois::Substrate::PerThreadStorage<int> local;
};

template<DetAlgo version>
struct Process {
  struct LocalState {
    LocalState(Process<version>& self, galois::PerIterAllocTy& alloc) { }
  };

  struct DeterministicId {
    uintptr_t operator()(const GNode& item) const {
      return app.graph.getData(item, galois::MethodFlag::UNPROTECTED).id;
    }
  };

  struct ParallelBreak {
    Process<version>& self;
    bool operator()() {
      if (app.global_relabel_interval > 0 && self.counter.accum.reduce() >= app.global_relabel_interval) {
        app.should_global_relabel = true;
        return true;
      }
      return false;
    }
  };

  ParallelBreak getParallelBreak() {
    return ParallelBreak { *this };
  }

  Counter& counter;

  typedef std::tuple<
    galois::needs_parallel_break<>,
    galois::needs_per_iter_alloc<>,
    galois::has_deterministic_local_state<LocalState>,
    galois::has_deterministic_id<DeterministicId>
  > function_traits;

  Process(Counter& c): counter(c) { }

  void operator()(GNode& src, galois::UserContext<GNode>& ctx) {
    if (version != nondet) {
      if (ctx.isFirstPass()) {
        acquire(src);
      }
      if (version == detDisjoint && ctx.isFirstPass()) {
          return;
      } else {
        app.graph.getData(src, galois::MethodFlag::WRITE);
        ctx.cautiousPoint();
      }
    }

    int increment = 1;
    if (discharge(src, ctx)) {
      increment += BETA;
    }

    counter.accum += increment;
  }
};

template<>
struct Process<nondet> {
  typedef std::tuple<
    galois::needs_parallel_break<>
  > function_traits;

  Counter& counter;
  int limit;
  Process(Counter& c): counter(c) { 
    limit = app.global_relabel_interval / numThreads;
  }

  void operator()(GNode& src, galois::UserContext<GNode>& ctx) {
    int increment = 1;
    acquire(src);
    if (discharge(src, ctx)) {
      increment += BETA;
    }

    int v = *counter.local.getLocal() += increment;
    if (app.global_relabel_interval > 0 && v >= limit) {
      app.should_global_relabel = true;
      ctx.breakLoop();
      return;
    }
  }
};

template<typename EdgeTy>
void writePfpGraph(const std::string& inputFile, const std::string& outputFile) {
  typedef galois::graphs::FileGraph ReaderGraph;
  typedef ReaderGraph::GraphNode ReaderGNode;

  ReaderGraph reader;
  reader.fromFile(inputFile);

  typedef galois::graphs::FileGraphWriter Writer;
  typedef galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Writer p;
  EdgeData edgeData;

  // Count edges
  size_t numEdges = 0;
  for (ReaderGraph::iterator ii = reader.begin(), ei = reader.end(); ii != ei; ++ii) {
    ReaderGNode rsrc = *ii;
    for (auto jj : reader.edges(rsrc)) {
      ReaderGNode rdst = reader.getEdgeDst(jj);
      if (rsrc == rdst) continue;
      if (!reader.hasNeighbor(rdst, rsrc)) 
        ++numEdges;
      ++numEdges;
    }
  }

  p.setNumNodes(reader.size());
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(sizeof(edge_value_type));

  p.phase1();
  for (ReaderGraph::iterator ii = reader.begin(), ei = reader.end(); ii != ei; ++ii) {
    ReaderGNode rsrc = *ii;
    for (auto jj : reader.edges(rsrc)) {
      ReaderGNode rdst = reader.getEdgeDst(jj);
      if (rsrc == rdst) continue;
      if (!reader.hasNeighbor(rdst, rsrc)) 
        p.incrementDegree(rdst);
      p.incrementDegree(rsrc);
    }
  }

  EdgeTy one = 1;
  static_assert(sizeof(one) == sizeof(uint32_t), "Unexpected edge data size");
  one = galois::convert_le32toh(one);

  p.phase2();
  edgeData.create(numEdges);
  for (ReaderGraph::iterator ii = reader.begin(), ei = reader.end(); ii != ei; ++ii) {
    ReaderGNode rsrc = *ii;
    for (auto jj : reader.edges(rsrc)) {
      ReaderGNode rdst = reader.getEdgeDst(jj);
      if (rsrc == rdst) continue;
      if (!reader.hasNeighbor(rdst, rsrc)) 
        edgeData.set(p.addNeighbor(rdst, rsrc), 0);
      EdgeTy cap = useUnitCapacity ? one : reader.getEdgeData<EdgeTy>(jj);
      edgeData.set(p.addNeighbor(rsrc, rdst), cap);
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), rawEdgeData);

  using Wnode = Writer::GraphNode;

  struct IdLess {
    bool operator()(const galois::graphs::EdgeSortValue<Wnode,edge_value_type>& e1, const galois::graphs::EdgeSortValue<Wnode,edge_value_type>& e2) const {
      return e1.dst < e2.dst;
    }
  };

  for (Writer::iterator i = p.begin (), end_i = p.end (); i != end_i; ++i) {
    p.sortEdges<edge_value_type> (*i, IdLess ());
  }

  p.toFile(outputFile);
}

void initializeGraph(std::string inputFile, uint32_t sourceId, uint32_t sinkId, Config *newApp) {
  if (useSymmetricDirectly) {
    galois::graphs::readGraph(newApp->graph, inputFile);
    for(auto ss : newApp->graph)
      for (auto ii : newApp->graph.edges(ss))
        newApp->graph.getEdgeData(ii) = 1;
  } else {
    if (inputFile.find(".gr.pfp") != inputFile.size() - strlen(".gr.pfp")) {
      std::string pfpName = inputFile + ".pfp";
      std::ifstream pfpFile(pfpName.c_str());
      if (!pfpFile.good()) {
        std::cout << "Writing new input file: " << pfpName << "\n";
        writePfpGraph<Graph::edge_data_type>(inputFile, pfpName);
      }
      inputFile = pfpName;
    }
    galois::graphs::readGraph(newApp->graph, inputFile);

    // Assume that input edge data has already been converted instead
#if 0//def HAVE_BIG_ENDIAN
    // Convert edge data to host ordering
    for (auto ss : newApp->graph) {
      for (auto ii : newApp->graph.edges(ss)) {
        Graph::edge_data_type& cap = newApp->graph.getEdgeData(ii);
        static_assert(sizeof(cap) == sizeof(uint32_t), "Unexpected edge data size");
        cap = galois::convert_le32toh(cap);
      }
    }
#endif
  }
  
  Graph& g = newApp->graph;

  if (sourceId == sinkId || sourceId >= g.size() || sinkId >= g.size()) {
    std::cerr << "invalid source or sink id\n";
    abort();
  }
  
  uint32_t id = 0;
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii, ++id) {
    if (id == sourceId) {
      newApp->source = *ii;
      g.getData(newApp->source).height = g.size();
    } else if (id == sinkId) {
      newApp->sink = *ii;
    }
    g.getData(*ii).id = id;
  }
}

template<typename C>
void initializePreflow(C& initial) {
  for (auto ii : app.graph.edges(app.source)) {
    GNode dst = app.graph.getEdgeDst(ii);
    int64_t cap = app.graph.getEdgeData(ii);
    reduceCapacity(ii, app.source, dst, cap);
    Node& node = app.graph.getData(dst);
    node.excess += cap;
    if (cap > 0)
      initial.push_back(dst);
  }
}

void checkSorting (void) {
  for (auto n : app.graph) {
    galois::optional<GNode> prevDst;
    for (auto e : app.graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode dst = app.graph.getEdgeDst (e);
      if (prevDst) {
        Node& prevNode = app.graph.getData (*prevDst, galois::MethodFlag::UNPROTECTED);
        Node& currNode = app.graph.getData (dst, galois::MethodFlag::UNPROTECTED);
        GALOIS_ASSERT (prevNode.id < currNode.id, "Adjacency list unsorted");
      }
      prevDst = dst;
    }
  }
}


void run() {
  typedef galois::worklists::Deterministic<> DWL;
  typedef galois::worklists::dChunkedFIFO<16> Chunk;
  typedef galois::worklists::OrderedByIntegerMetric<Indexer,Chunk> OBIM;

  galois::InsertBag<GNode> initial;
  initializePreflow(initial);

  while (initial.begin() != initial.end()) {
    galois::StatTimer T_discharge("DischargeTime");
    T_discharge.start();
    Counter counter;
    switch (detAlgo) {
      case nondet:
        if (useHLOrder) {
          galois::for_each_local(initial, Process<nondet>(counter), galois::loopname("Discharge"), galois::wl<OBIM>());
        } else {
          galois::for_each_local(initial, Process<nondet>(counter), galois::loopname("Discharge"));
        }
        break;
      case detBase:
        {
          Process<detBase> fn(counter);
          galois::for_each_local(initial,
              fn, 
              galois::loopname("Discharge"),
              galois::wl<DWL>(),
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1400
              galois::has_deterministic_parallel_break<Process<detBase>::ParallelBreak>(fn.getParallelBreak())
#else
              galois::make_trait_with_args<galois::has_deterministic_parallel_break>(fn.getParallelBreak())
#endif
              );
        }
        break;
      case detDisjoint:
        {
          Process<detDisjoint> fn(counter);
          galois::for_each_local(initial,
              fn, 
              galois::loopname("Discharge"),
              galois::wl<DWL>(),
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1400
              galois::has_deterministic_parallel_break<Process<detDisjoint>::ParallelBreak>(fn.getParallelBreak())
#else
              galois::make_trait_with_args<galois::has_deterministic_parallel_break>(fn.getParallelBreak())
#endif
              );
        }
        break;
      default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
    }
    T_discharge.stop();

    if (app.should_global_relabel) {
      galois::StatTimer T_global_relabel("GlobalRelabelTime");
      T_global_relabel.start();
      initial.clear();
      globalRelabel(initial);
      app.should_global_relabel = false;
      std::cout 
        << " Flow after global relabel: "
        << app.graph.getData(app.sink).excess << "\n";
      T_global_relabel.stop();
    } else {
      break;
    }
  }
}


int main(int argc, char** argv) {
  galois::StatManager M;
  bool serial = false;
  LonestarStart(argc, argv, name, desc, url);

  initializeGraph(filename, sourceId, sinkId, &app);

  // TODO: remove later
  checkSorting ();

  if (relabelInt == 0) {
    app.global_relabel_interval = app.graph.size() * ALPHA + app.graph.sizeEdges() / 3;
  } else {
    app.global_relabel_interval = relabelInt;
  }
  std::cout << "number of nodes: " << app.graph.size() << "\n";
  std::cout << "global relabel interval: " << app.global_relabel_interval << "\n";
  std::cout << "serial execution: " << (serial ? "yes" : "no") << "\n";

  galois::StatTimer T;
  T.start();
  run();
  T.stop();

  std::cout << "Flow is " << app.graph.getData(app.sink).excess << "\n";
  
  if (!skipVerify) {
    Config orig;
    initializeGraph(filename, sourceId, sinkId, &orig);
    verify(orig);
    std::cout << "(Partially) Verified\n";
  }

  return 0;
}

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
#include "Galois/Statistic.h"
#include "Galois/Bag.h"
#include "Galois/Graph/LCGraph.h"
#include "llvm/Support/CommandLine.h"

#ifdef GALOIS_USE_EXP
#include "Galois/PriorityScheduling.h"
#endif

#include "Lonestar/BoilerPlate.h"

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
  size_t excess;
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

typedef Galois::Graph::LC_Linear_Graph<Node, int32_t>::with_numa_alloc<true>::type Graph;
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
    return -app.graph.getData(n, Galois::MethodFlag::NONE).height;
  }
};

struct GLess :std::binary_function<GNode, GNode, bool> {
  bool operator()(const GNode& lhs, const GNode& rhs) const {
    int lv = -app.graph.getData(lhs, Galois::MethodFlag::NONE).height;
    int rv = -app.graph.getData(rhs, Galois::MethodFlag::NONE).height;
    return lv < rv;
  }
};
struct GGreater :std::binary_function<GNode, GNode, bool> {
  bool operator()(const GNode& lhs, const GNode& rhs) const {
    int lv = -app.graph.getData(lhs, Galois::MethodFlag::NONE).height;
    int rv = -app.graph.getData(rhs, Galois::MethodFlag::NONE).height;
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
    for (Graph::edge_iterator ii = app.graph.edge_begin(src),
        ee = app.graph.edge_end(src); ii != ee; ++ii) {
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
    for (Graph::edge_iterator jj = app.graph.edge_begin(src),
        ej = app.graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = app.graph.getEdgeDst(jj);
      int cap = app.graph.getEdgeData(jj);
      int dh = app.graph.getData(dst).height;
      if (cap > 0 && sh > dh + 1) {
        std::cerr << "height violated at " << app.graph.getData(src) << "\n";
        abort();
      }
    }
  }
}

Graph::edge_iterator findEdge(Graph& g, GNode src, GNode dst) {
  Graph::edge_iterator ii = g.edge_begin(src, Galois::MethodFlag::NONE), ei = g.edge_end(src, Galois::MethodFlag::NONE);
  for (; ii != ei; ++ii) {
    if (g.getEdgeDst(ii) == dst)
      break;
  }
  return ii;
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

    size_t sum = 0;
    for (Graph::edge_iterator jj = app.graph.edge_begin(src),
        ej = app.graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = app.graph.getEdgeDst(jj);
      uint32_t dstId = app.graph.getData(dst).id;
      int ocap = orig.graph.getEdgeData(findEdge(orig.graph, map[srcId], map[dstId]));
      int delta = 0;
      if (ocap > 0) 
        delta -= ocap - app.graph.getEdgeData(jj);
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

void reduceCapacity(const Graph::edge_iterator& ii, const GNode& src, const GNode& dst, int amount) {
  int& cap1 = app.graph.getEdgeData(ii);
  int& cap2 = app.graph.getEdgeData(findEdge(app.graph, dst, src));
  cap1 -= amount;
  cap2 += amount;
}

template<DetAlgo version,bool useCAS=true>
struct UpdateHeights {
  //typedef int tt_does_not_need_aborts;
  typedef int tt_needs_per_iter_alloc; // For LocalState

  struct LocalState {
    LocalState(UpdateHeights<version,useCAS>& self, Galois::PerIterAllocTy& alloc) { }
  };
  typedef LocalState GaloisDeterministicLocalState;
  static_assert(Galois::has_deterministic_local_state<UpdateHeights>::value, "Oops");

  //struct IdFn {
  //  unsigned long operator()(const GNode& item) const {
  //    return app.graph.getData(item, Galois::MethodFlag::NONE).id;
  //  }
  //};

  /**
   * Do reverse BFS on residual graph.
   */
  void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) {
    if (version != nondet) {
      bool used = false;
      if (version == detDisjoint) {
        ctx.getLocalState(used);
      }

      if (!used) {
        for (Graph::edge_iterator
            ii = app.graph.edge_begin(src, Galois::MethodFlag::CHECK_CONFLICT),
            ee = app.graph.edge_end(src, Galois::MethodFlag::CHECK_CONFLICT);
            ii != ee; ++ii) {
          GNode dst = app.graph.getEdgeDst(ii);
          int rdata = app.graph.getEdgeData(findEdge(app.graph, dst, src));
          if (rdata > 0) {
            app.graph.getData(dst, Galois::MethodFlag::CHECK_CONFLICT);
          }
        }
      }

      if (version == detDisjoint) {
        if (!used)
          return;
      } else {
        app.graph.getData(src, Galois::MethodFlag::WRITE);
      }
    }

    for (Graph::edge_iterator
        ii = app.graph.edge_begin(src, useCAS ? Galois::MethodFlag::NONE : Galois::MethodFlag::CHECK_CONFLICT),
        ee = app.graph.edge_end(src, useCAS ? Galois::MethodFlag::NONE : Galois::MethodFlag::CHECK_CONFLICT);
        ii != ee; ++ii) {
      GNode dst = app.graph.getEdgeDst(ii);
      int rdata = app.graph.getEdgeData(findEdge(app.graph, dst, src));
      if (rdata > 0) {
        Node& node = app.graph.getData(dst, Galois::MethodFlag::NONE);
        int newHeight = app.graph.getData(src, Galois::MethodFlag::NONE).height + 1;
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
  void operator()(const GNode& src) {
    Node& node = app.graph.getData(src, Galois::MethodFlag::NONE);
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

  void operator()(const GNode& src) {
    Node& node = app.graph.getData(src, Galois::MethodFlag::NONE);
    if (src == app.sink || src == app.source || node.height >= (int) app.graph.size())
      return;
    if (node.excess > 0) 
      wl.push_back(src);
  }
};

template<typename IncomingWL>
void globalRelabel(IncomingWL& incoming) {
  Galois::StatTimer T1("ResetHeightsTime");
  T1.start();
  Galois::do_all_local(app.graph, ResetHeights(), "ResetHeights");
  T1.stop();

  Galois::StatTimer T("UpdateHeightsTime");
  T.start();

  switch (detAlgo) {
    case nondet:
#ifdef GALOIS_USE_EXP
      Galois::for_each<Galois::WorkList::BulkSynchronousInline<> >(app.sink, UpdateHeights<nondet>(), "UpdateHeights");
#else
      Galois::for_each(app.sink, UpdateHeights<nondet>(), "UpdateHeights");
#endif
      break;
    case detBase:
      Galois::for_each_det(app.sink, UpdateHeights<detBase>(), "UpdateHeights");
      break;
    case detDisjoint:
      Galois::for_each_det(app.sink, UpdateHeights<detDisjoint>(), "UpdateHeights");
      break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
  T.stop();

  Galois::StatTimer T2("FindWorkTime");
  T2.start();
  Galois::do_all_local(app.graph, FindWork<IncomingWL>(incoming), "FindWork");
  T2.stop();
}

void acquire(const GNode& src) {
  // LC Graphs have a different idea of locking
  for (Graph::edge_iterator 
      ii = app.graph.edge_begin(src, Galois::MethodFlag::CHECK_CONFLICT),
      ee = app.graph.edge_end(src, Galois::MethodFlag::CHECK_CONFLICT);
      ii != ee; ++ii) {
    GNode dst = app.graph.getEdgeDst(ii);
    app.graph.getData(dst, Galois::MethodFlag::CHECK_CONFLICT);
  }
}

void relabel(const GNode& src) {
  int minHeight = std::numeric_limits<int>::max();
  int minEdge;

  int current = 0;
  for (Graph::edge_iterator 
      ii = app.graph.edge_begin(src, Galois::MethodFlag::NONE),
      ee = app.graph.edge_end(src, Galois::MethodFlag::NONE);
      ii != ee; ++ii, ++current) {
    GNode dst = app.graph.getEdgeDst(ii);
    int cap = app.graph.getEdgeData(ii);
    if (cap > 0) {
      const Node& dnode = app.graph.getData(dst, Galois::MethodFlag::NONE);
      if (dnode.height < minHeight) {
        minHeight = dnode.height;
        minEdge = current;
      }
    }
  }

  assert(minHeight != std::numeric_limits<int>::max());
  ++minHeight;

  Node& node = app.graph.getData(src, Galois::MethodFlag::NONE);
  if (minHeight < (int) app.graph.size()) {
    node.height = minHeight;
    node.current = minEdge;
  } else {
    node.height = app.graph.size();
  }
}

bool discharge(const GNode& src, Galois::UserContext<GNode>& ctx) {
  //Node& node = app.graph.getData(src, Galois::MethodFlag::CHECK_CONFLICT);
  Node& node = app.graph.getData(src, Galois::MethodFlag::NONE);
  //int prevHeight = node.height;
  bool relabeled = false;

  if (node.excess == 0 || node.height >= (int) app.graph.size()) {
    return false;
  }

  while (true) {
    //Galois::MethodFlag flag = relabeled ? Galois::MethodFlag::NONE : Galois::MethodFlag::CHECK_CONFLICT;
    Galois::MethodFlag flag = Galois::MethodFlag::NONE;
    bool finished = false;
    int current = node.current;
    Graph::edge_iterator
      ii = app.graph.edge_begin(src, flag),
      ee = app.graph.edge_end(src, flag);
    std::advance(ii, node.current);
    for (; ii != ee; ++ii, ++current) {
      GNode dst = app.graph.getEdgeDst(ii);
      int cap = app.graph.getEdgeData(ii);
      if (cap == 0)// || current < node.current) 
        continue;

      Node& dnode = app.graph.getData(dst, Galois::MethodFlag::NONE);
      if (node.height - 1 != dnode.height) 
        continue;

      // Push flow
      int amount = std::min(static_cast<int>(node.excess), cap);
      reduceCapacity(ii, src, dst, amount);

      // Only add once
      if (dst != app.sink && dst != app.source && dnode.excess == 0) 
        ctx.push(dst);
      
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
  Galois::GAccumulator<int> accum;
  Galois::Runtime::PerThreadStorage<int> local;
};

template<DetAlgo version>
struct Process {
  typedef int tt_needs_parallel_break;
  typedef int tt_needs_per_iter_alloc; // For LocalState

  struct LocalState {
    LocalState(Process<version>& self, Galois::PerIterAllocTy& alloc) { }
  };
  typedef LocalState GaloisDeterministicLocalState;
  static_assert(Galois::has_deterministic_local_state<Process>::value, "Oops");

  uintptr_t galoisDeterministicId(const GNode& item) const {
    return app.graph.getData(item, Galois::MethodFlag::NONE).id;
  }
  static_assert(Galois::has_deterministic_id<Process>::value, "Oops");

  bool galoisDeterministicParallelBreak() {
    if (app.global_relabel_interval > 0 && counter.accum.reduce() >= app.global_relabel_interval) {
      app.should_global_relabel = true;
      return true;
    }
    return false;
  }
  static_assert(Galois::has_deterministic_parallel_break<Process>::value, "Oops");

  Counter& counter;

  Process(Counter& c): counter(c) { }

  void operator()(GNode& src, Galois::UserContext<GNode>& ctx) {
    if (version != nondet) {
      bool used = false;
      if (version == detDisjoint) {
        ctx.getLocalState(used);
      }
      if (!used) {
        acquire(src);
      }
      if (version == detDisjoint) {
        if (!used)
          return;
      } else {
        app.graph.getData(src, Galois::MethodFlag::WRITE);
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
  typedef int tt_needs_parallel_break;

  Counter& counter;
  int limit;
  Process(Counter& c): counter(c) { 
    limit = app.global_relabel_interval / numThreads;
  }

  void operator()(GNode& src, Galois::UserContext<GNode>& ctx) {
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
  typedef Galois::Graph::FileGraph ReaderGraph;
  typedef ReaderGraph::GraphNode ReaderGNode;

  ReaderGraph reader;
  reader.structureFromFile(inputFile);

  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Writer p;
  EdgeData edgeData;

  // Count edges
  size_t numEdges = 0;
  for (ReaderGraph::iterator ii = reader.begin(), ei = reader.end(); ii != ei; ++ii) {
    ReaderGNode rsrc = *ii;
    for (ReaderGraph::edge_iterator jj = reader.edge_begin(rsrc),
        ej = reader.edge_end(rsrc); jj != ej; ++jj) {
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
    for (ReaderGraph::edge_iterator jj = reader.edge_begin(rsrc),
        ej = reader.edge_end(rsrc); jj != ej; ++jj) {
      ReaderGNode rdst = reader.getEdgeDst(jj);
      if (rsrc == rdst) continue;
      if (!reader.hasNeighbor(rdst, rsrc)) 
        p.incrementDegree(rdst);
      p.incrementDegree(rsrc);
    }
  }

  p.phase2();
  edgeData.create(numEdges);
  for (ReaderGraph::iterator ii = reader.begin(), ei = reader.end(); ii != ei; ++ii) {
    ReaderGNode rsrc = *ii;
    for (ReaderGraph::edge_iterator jj = reader.edge_begin(rsrc),
        ej = reader.edge_end(rsrc); jj != ej; ++jj) {
      ReaderGNode rdst = reader.getEdgeDst(jj);
      if (rsrc == rdst) continue;
      if (!reader.hasNeighbor(rdst, rsrc)) 
        edgeData.set(p.addNeighbor(rdst, rsrc), 0);
      EdgeTy cap = useUnitCapacity ? 1 : reader.getEdgeData<EdgeTy>(jj);
      edgeData.set(p.addNeighbor(rsrc, rdst), cap);
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

  p.structureToFile(outputFile);
}

void initializeGraph(std::string inputFile,
    uint32_t sourceId, uint32_t sinkId, Config *newApp) {
  if (useSymmetricDirectly) {
    Galois::Graph::readGraph(newApp->graph, inputFile);
    for (Graph::iterator ss = newApp->graph.begin(), es = newApp->graph.end(); ss != es; ++ss) {
      for (Graph::edge_iterator ii = newApp->graph.edge_begin(*ss), ei = newApp->graph.edge_end(*ss); ii != ei; ++ii)
        newApp->graph.getEdgeData(ii) = 1;
    }
  } else {
    if (inputFile.find(".gr.pfp") != inputFile.size() - strlen(".gr.pfp")) {
      std::string pfpName = inputFile + ".pfp";
      std::ifstream pfpFile(pfpName.c_str());
      if (!pfpFile.good()) {
        std::cout << "Writing new input file: " << pfpName << "\n";
        writePfpGraph<int32_t>(inputFile, pfpName);
      }
      inputFile = pfpName;
    }
    Galois::Graph::readGraph(newApp->graph, inputFile);
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
  for (Graph::edge_iterator ii = app.graph.edge_begin(app.source),
      ee = app.graph.edge_end(app.source); ii != ee; ++ii) {
    GNode dst = app.graph.getEdgeDst(ii);
    int cap = app.graph.getEdgeData(ii);
    reduceCapacity(ii, app.source, dst, cap);
    Node& node = app.graph.getData(dst);
    node.excess += cap;
    if (cap > 0)
      initial.push_back(dst);
  }
}

void run() {
  typedef Galois::WorkList::dChunkedFIFO<16> Chunk;
  typedef Galois::WorkList::OrderedByIntegerMetric<Indexer,Chunk> OBIM;

  Galois::InsertBag<GNode> initial;
  initializePreflow(initial);

  while (initial.begin() != initial.end()) {
    Galois::StatTimer T_discharge("DischargeTime");
    T_discharge.start();
    Counter counter;
    switch (detAlgo) {
      case nondet:
        if (useHLOrder) {
          Galois::for_each_local<OBIM>(initial, Process<nondet>(counter), "Discharge");
        } else {
          Galois::for_each_local(initial, Process<nondet>(counter), "Discharge");
        }
        break;
      case detBase:
        Galois::for_each_det(initial.begin(), initial.end(), Process<detBase>(counter), "Discharge");
        break;
      case detDisjoint:
        Galois::for_each_det(initial.begin(), initial.end(), Process<detDisjoint>(counter), "Discharge");
        break;
      default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
    }
    T_discharge.stop();

    if (app.should_global_relabel) {
      Galois::StatTimer T_global_relabel("GlobalRelabelTime");
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
  Galois::StatManager M;
  bool serial = false;
  LonestarStart(argc, argv, name, desc, url);

  initializeGraph(filename, sourceId, sinkId, &app);
  if (relabelInt == 0) {
    app.global_relabel_interval = app.graph.size() * ALPHA + app.graph.sizeEdges() / 3;
  } else {
    app.global_relabel_interval = relabelInt;
  }
  std::cout << "number of nodes: " << app.graph.size() << "\n";
  std::cout << "global relabel interval: " << app.global_relabel_interval << "\n";
  std::cout << "serial execution: " << (serial ? "yes" : "no") << "\n";

  Galois::StatTimer T;
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

/** Preflow-push application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
//#include <algorithm>
#include <iostream>

#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/FileGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

namespace {

const char* name = "Preflow Push";
const char* desc = "Finds the maximum flow in a network using the preflow push technique\n";
const char* url = "preflow_push";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> sourceId(cll::Positional, cll::desc("sourceID"), cll::Required);
static cll::opt<int> sinkId(cll::Positional, cll::desc("sinkID"), cll::Required);
static cll::opt<int> relabelInt("relabel", cll::desc("relabel interval"), cll::init(0));

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
  size_t excess;
  int height;
  // During verification we reuse this field to store node indices
  union {
    int current;
    int id;
  };

  Node() : excess(0), height(1), current(0) { }
};

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "(excess: " << n.excess
     << ", height: " << n.height
     << ", current: " << n.current << ")";
  return os;
}

typedef Galois::Graph::FirstGraph<Node, int, true> Graph;
typedef Graph::GraphNode GNode;

struct Config {
  Graph graph;
  GNode sink;
  GNode source;
  int num_nodes;
  int num_edges;
  int global_relabel_interval;
  bool should_global_relabel;
  Config() : should_global_relabel(false) {}
};

Config app;

void checkAugmentingPath() {
  // Use id field as visited flag
  for (Graph::active_iterator ii = app.graph.active_begin(), 
      ee = app.graph.active_end(); ii != ee; ++ii) {
    GNode src = *ii;
    src.getData().id = 0;
  }

  std::deque<GNode> queue;

  app.source.getData().id = 1;
  queue.push_back(app.source);

  while (!queue.empty()) {
    GNode& src = queue.front();
    queue.pop_front();
    for (Graph::neighbor_iterator ii = app.graph.neighbor_begin(src),
        ee = app.graph.neighbor_end(src); ii != ee; ++ii) {
      GNode dst = *ii;
      if (dst.getData().id == 0
          && app.graph.getEdgeData(src, ii) > 0) {
        dst.getData().id = 1;
        queue.push_back(dst);
      }
    }
  }

  if (app.sink.getData().id != 0) {
    assert(false && "Augmenting path exisits");
    abort();
  }
}

void checkHeights() {
  for (Graph::active_iterator i = app.graph.active_begin(),
      iend = app.graph.active_end(); i != iend; ++i) {
    GNode src = *i;
    for (Graph::neighbor_iterator j = app.graph.neighbor_begin(src),
        jend = app.graph.neighbor_end(src); j != jend; ++j) {
      GNode dst = *j;
      int sh = src.getData().height;
      int dh = dst.getData().height;
      int cap = app.graph.getEdgeData(src, j);
      if (cap > 0 && sh > dh + 1) {
        std::cerr << "height violated at " << src.getData() << "\n";
        abort();
      }
    }
  }
}

void checkConservation(Config& orig) {
  std::vector<GNode> map;
  map.resize(app.num_nodes);

  // Setup ids assuming same iteration order in both graphs
  int id = 0;
  for (Graph::active_iterator i = app.graph.active_begin(),
      iend = app.graph.active_end(); i != iend; ++i, ++id) {
    i->getData().id = id;
  }
  id = 0;
  for (Graph::active_iterator i = orig.graph.active_begin(),
      iend = orig.graph.active_end(); i != iend; ++i, ++id) {
    i->getData().id = id;
    map[id] = *i;
  }

  // Now do some checking
  for (Graph::active_iterator i = app.graph.active_begin(),
      iend = app.graph.active_end(); i != iend; ++i) {
    GNode src = *i;
    Node node = src.getData();
    int srcId = node.id;

    if (src == app.source || src == app.sink)
      continue;

    if (node.excess != 0 && node.height != app.num_nodes) {
      std::cerr << "Non-zero excess at " << node << "\n";
      abort();
    }

    size_t sum = 0;
    for (Graph::neighbor_iterator j = app.graph.neighbor_begin(src),
        jend = app.graph.neighbor_end(src); j != jend; ++j) {
      GNode dst = *j;
      int dstId = dst.getData().id;
      int ocap = orig.graph.getEdgeData(map[srcId], map[dstId]);
      int delta = 0;
      if (ocap > 0) 
        delta -= ocap - app.graph.getEdgeData(src, j);
      else
        delta += app.graph.getEdgeData(src, j);
      sum += delta;
    }

    if (node.excess != sum) {
      std::cerr << "Not pseudoflow " << node.excess << " != " << sum 
        << " at node" << node.id << "\n";
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

void reduceCapacity(const GNode& src, const GNode& dst, int amount) {
  int& cap1 = app.graph.getEdgeData(src, dst, Galois::NONE);
  int& cap2 = app.graph.getEdgeData(dst, src, Galois::NONE);
  cap1 -= amount;
  cap2 += amount;
}

template<Galois::MethodFlag flag, bool useCAS = false>
struct UpdateHeights {
  typedef int tt_does_not_need_stats;
  /**
   * Do reverse BFS on residual graph.
   */
  template<typename Context>
  void operator()(const GNode& src, Context& ctx) {
    for (Graph::neighbor_iterator
        ii = app.graph.neighbor_begin(src, useCAS ? Galois::NONE : flag),
        ee = app.graph.neighbor_end(src, useCAS ? Galois::NONE : flag);
        ii != ee; ++ii) {
      GNode dst = *ii;
      if (app.graph.getEdgeData(dst, src, Galois::NONE) > 0) {
        Node& node = dst.getData(Galois::NONE);
        int newHeight = src.getData(Galois::NONE).height + 1;
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
  typedef int tt_does_not_need_context;
  typedef int tt_does_not_need_stats;

  template<typename Context>
  void operator()(const GNode& src, Context&) {
    Node& node = src.getData(Galois::NONE);
    node.height = app.num_nodes;
    node.current = 0;
    if (src == app.sink)
      node.height = 0;
  }
};

template<typename WLTy>
struct FindWork {
  typedef int tt_does_not_need_context;
  typedef int tt_does_not_need_stats;
  WLTy& wl;
  FindWork(WLTy& w) : wl(w) {}

  template<typename Context>
  void operator()(const GNode& src, Context&) {
    Node& node = src.getData(Galois::NONE);
    if (src == app.sink || src == app.source || node.height >= app.num_nodes)
      return;
    if (node.excess > 0) 
      wl.push(src);
  }
};

template<Galois::MethodFlag flag, typename IncomingWL>
void globalRelabel(IncomingWL& incoming) {
  Galois::StatTimer T1("ASDF");
  T1.start();
  typedef GaloisRuntime::WorkList::dChunkedLIFO<1024> SimpleScheduler;
  Galois::for_each<SimpleScheduler>(app.graph.active_begin(),
      app.graph.active_end(),
      ResetHeights());
  T1.stop();

  Galois::StatTimer T("BfsTime");
  T.start();
  typedef GaloisRuntime::WorkList::dChunkedFIFO<16> WL;
  std::vector<GNode> single;
  single.push_back(app.sink);
  Galois::for_each<WL>(single.begin(), single.end(), UpdateHeights<flag>());
  T.stop();

  Galois::StatTimer T2("FF");
  T2.start();
  typedef Galois::InsertBag<GNode> NewWorkTy;
  NewWorkTy new_work;
  Galois::for_each<SimpleScheduler>(app.graph.active_begin(),
      app.graph.active_end(),
      FindWork<NewWorkTy>(new_work));
  for (NewWorkTy::iterator it = new_work.begin(), end = new_work.end(); 
      it != end; ++it) {
    incoming.push_back(*it);
  }
  T2.stop();
}

struct Process {
  typedef int tt_needs_parallel_break;
  int counter;

  Process() : counter(0) { }

  template<typename Context>
  void operator()(GNode& src, Context& ctx) {
    int increment = 1;
    if (discharge<Context>(src, ctx)) {
      increment += BETA;
    }

    counter += increment;
    if (app.global_relabel_interval && counter >= app.global_relabel_interval) {
      app.should_global_relabel = true;
      ctx.breakLoop();
      return;
    }
  }

  template<typename Context>
  bool discharge(const GNode& src, Context& ctx) {
    Node& node = src.getData(Galois::CHECK_CONFLICT);
    int prevHeight = node.height;
    bool relabeled = false;

    if (node.excess == 0 || node.height >= app.num_nodes) {
      return false;
    }

    while (true) {
      Galois::MethodFlag f =
        relabeled ? Galois::NONE : Galois::CHECK_CONFLICT;
      bool finished = false;
      int current = 0;

      for (Graph::neighbor_iterator ii = app.graph.neighbor_begin(src, f),
          ee = app.graph.neighbor_end(src, f);
          ii != ee; ++ii, ++current) {
        GNode dst = *ii;
        int cap = app.graph.getEdgeData(src, ii, Galois::NONE);
        if (cap == 0 || current < node.current) 
          continue;

        Node& dnode = dst.getData(Galois::NONE);
        if (node.height - 1 != dnode.height) 
          continue;

        // Push flow
        int amount = std::min(static_cast<int>(node.excess), cap);
        reduceCapacity(src, dst, amount);

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

      if (node.height == app.num_nodes)
        break;

      prevHeight = node.height;
    }

    return relabeled;
  }

  void relabel(const GNode& src) {
    int minHeight = std::numeric_limits<int>::max();
    int minEdge;

    int current = 0;
    for (Graph::neighbor_iterator 
        ii = app.graph.neighbor_begin(src, Galois::NONE),
        ee = app.graph.neighbor_end(src, Galois::NONE);
        ii != ee; ++ii, ++current) {
      GNode dst = *ii;
      int cap = app.graph.getEdgeData(src, ii, Galois::NONE);
      if (cap > 0) {
        Node& dnode = dst.getData(Galois::NONE);
        if (dnode.height < minHeight) {
          minHeight = dnode.height;
          minEdge = current;
        }
      }
    }

    assert(minHeight != std::numeric_limits<int>::max());
    ++minHeight;

    Node& node = src.getData(Galois::NONE);
    if (minHeight < app.num_nodes) {
      node.height = minHeight;
      node.current = minEdge;
    } else {
      node.height = app.num_nodes;
    }
  }
};

void initializeGraph(const char* inputFile,
    int sourceId, int sinkId, Config *newApp) {
  typedef Galois::Graph::LC_FileGraph<int, int> ReaderGraph;
  typedef ReaderGraph::GraphNode ReaderGNode;

  ReaderGraph reader;
  reader.structureFromFile(inputFile);
  reader.emptyNodeData();

  // Assign ids to ReaderGNodes
  newApp->num_nodes = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(),
      ee = reader.active_end(); ii != ee; ++ii, ++newApp->num_nodes) {
    ReaderGNode src = *ii;
    reader.getData(src) = newApp->num_nodes;
  }

  // Create dense map between ids and GNodes
  std::vector<GNode> nodes;
  nodes.resize(newApp->num_nodes);
  for (int i = 0; i < newApp->num_nodes; ++i) {
    Node node;

    if (i == sourceId) {
      node.height = newApp->num_nodes;
    }

    GNode src = newApp->graph.createNode(node);
    newApp->graph.addNode(src);
    if (i == sourceId) {
      newApp->source = src;
    } else if (i == sinkId) {
      newApp->sink = src;
    }
    nodes[i] = src;
  }

  // Create edges
  newApp->num_edges = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(),
      ee = reader.active_end(); ii != ee; ++ii) {
    ReaderGNode rsrc = *ii;
    int rsrcId = reader.getData(rsrc);
    for (ReaderGraph::neighbor_iterator jj = reader.neighbor_begin(rsrc),
        ff = reader.neighbor_end(rsrc); jj != ff; ++jj) {
      ReaderGNode rdst = *jj;
      int rdstId = reader.getData(rdst);
      int cap = reader.getEdgeData(rsrc, rdst);
      newApp->graph.addEdge(nodes[rsrcId], nodes[rdstId], cap);
      ++newApp->num_edges;
      // Add reverse edge if not already there
      if (!reader.has_neighbor(rdst, rsrc)) {
        newApp->graph.addEdge(nodes[rdstId], nodes[rsrcId], 0);
        ++newApp->num_edges;
      }
    }
  }
}

template<typename C>
void initializePreflow(C& initial) {
  for (Graph::neighbor_iterator ii = app.graph.neighbor_begin(app.source),
      ee = app.graph.neighbor_end(app.source); ii != ee; ++ii) {
    GNode dst = *ii;
    int cap = app.graph.getEdgeData(app.source, ii);
    reduceCapacity(app.source, dst, cap);
    Node& node = dst.getData();
    node.excess += cap;
    if (cap > 0)
      initial.push_back(dst);
  }
}

struct Indexer :std::binary_function<GNode, int, int> {
  int operator()(GNode n) const {
    return (-app.graph.getData(n, Galois::NONE).height) >> 2;
  }
};

void run() {
  typedef GaloisRuntime::WorkList::dChunkedFIFO<16> Chunk;
  typedef GaloisRuntime::WorkList::OrderedByIntegerMetric<Indexer,Chunk> OBIM;

  std::vector<GNode> initial;
  initializePreflow(initial);

  while (!initial.empty()) {
    Galois::StatTimer T_discharge("DischargeTime");
    T_discharge.start();
    Galois::for_each(initial.begin(), initial.end(), Process());
    T_discharge.stop();

    if (app.should_global_relabel) {
      Galois::StatTimer T_global_relabel("GlobalRelabelTime");
      T_global_relabel.start();
      initial.clear();
      globalRelabel<Galois::CHECK_CONFLICT>(initial);
      app.should_global_relabel = false;
      std::cout 
        << " Flow after global relabel: "
        << app.sink.getData().excess << "\n";
      T_global_relabel.stop();
    } else {
      break;
    }
  }
}

} // end namespace

int main(int argc, char** argv) {
  bool serial = false;
  LonestarStart(argc, argv, std::cout, name, desc, url);

  if (sourceId < 0 || sinkId < 0 || sourceId == sinkId) {
    std::cerr << "invalid source or sink id\n";
    abort();
  }
  initializeGraph(filename.c_str(), sourceId, sinkId, &app);
  if (sourceId >= app.num_nodes || sinkId >= app.num_nodes) {
    std::cerr << "invalid source or sink id\n";
    abort();
  }
  
  if (relabelInt) {
    app.global_relabel_interval = relabelInt;
  } else {
    app.global_relabel_interval = app.num_nodes * ALPHA + app.num_edges;
    // TODO fix interval by dividing by numThreads ?
    app.global_relabel_interval /= numThreads;
  }

  std::cout << "number of nodes: " << app.num_nodes << "\n";
  std::cout << "global relabel interval: " << app.global_relabel_interval << "\n";
  std::cout << "serial execution: " << (serial ? "yes" : "no") << "\n";

  Galois::StatTimer T;
  T.start();
  run();
  T.stop();

  std::cout << "Flow is " << app.sink.getData().excess << "\n";
  
  if (!skipVerify) {
    Config orig;
    initializeGraph(filename.c_str(), sourceId, sinkId, &orig);
    verify(orig);
    std::cout << "(Partially) Verified\n";
  }

  return 0;
}

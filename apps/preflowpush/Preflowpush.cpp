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
#include <algorithm>
#include <limits>
#include <set>
#include "Galois/Launcher.h"
#include "Galois/Galois.h"
#include "Galois/TypeTraits.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/FileGraph.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Preflow Push";
static const char* description =
  "Finds the maximum flow in a network using the preflow push technique\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/preflowpush.html";
static const char* help =
  "<input file> <source id> <sink id> [global relabel interval]";

/**
 * Alpha parameter the original Goldberg algorithm to control when global
 * relabeling occurs. For comparison purposes, we keep them the same as
 * before, but it is possible to achieve much better performance by adjusting
 * the global relabel frequency.
 */
static const int ALPHA = 6;

/**
 * Beta parameter the original Goldberg algorithm to control when global
 * relabeling occurs. For comparison purposes, we keep them the same as
 * before, but it is possible to achieve much better performance by adjusting
 * the global relabel frequency.
 */
static const int BETA = 12;

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
  int numNodes;
  int numEdges;
  int globalRelabelInterval;
};

Config app;

static void checkAugmentingPath() {
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
          && app.graph.getEdgeData(src, dst) > 0) {
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

static void checkHeights() {
  for (Graph::active_iterator i = app.graph.active_begin(),
      iend = app.graph.active_end(); i != iend; ++i) {
    GNode src = *i;
    for (Graph::neighbor_iterator j = app.graph.neighbor_begin(src),
        jend = app.graph.neighbor_end(src); j != jend; ++j) {
      GNode dst = *j;
      int sh = src.getData().height;
      int dh = dst.getData().height;
      int cap = app.graph.getEdgeData(src, dst);
      if (cap > 0 && sh > dh + 1) {
        std::cerr << "height violated at " << src.getData() << "\n";
        abort();
      }
    }
  }
}

static void checkConservation(Config& orig) {
  std::vector<GNode> map;
  map.resize(app.numNodes);

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

    if (node.excess != 0 && node.height != app.numNodes) {
      std::cerr << "Non-zero excess at " << node << "\n";
      abort();
    }

    int sum = 0;
    for (Graph::neighbor_iterator j = app.graph.neighbor_begin(src),
        jend = app.graph.neighbor_end(src); j != jend; ++j) {
      GNode dst = *j;
      int dstId = dst.getData().id;
      int ocap = orig.graph.getEdgeData(map[srcId], map[dstId]);
      int delta = 0;
      if (ocap > 0) 
        delta -= ocap - app.graph.getEdgeData(src, dst);
      else
        delta += app.graph.getEdgeData(src, dst);
      sum += delta;
    }

    if (node.excess != sum) {
      std::cerr << "Not pseudoflow " << node.excess << " != " << sum 
        << " at node" << node.id << "\n";
      abort();
    }
  }
}

static void verify(Config& orig) {
  // FIXME: doesn't fully check result
  checkHeights();
  checkConservation(orig);
  checkAugmentingPath();
}

static void reduceCapacity(const GNode& src, const GNode& dst, int amount) {
  int& cap1 = app.graph.getEdgeData(src, dst, Galois::Graph::NONE);
  int& cap2 = app.graph.getEdgeData(dst, src, Galois::Graph::NONE);
  cap1 -= amount;
  cap2 += amount;
}

struct CleanGap {
  int height;
  CleanGap(int _height) : height(_height) { }

  template<typename Context>
  void operator()(const GNode& src, Context& ctx) {
    if (src == app.sink || src == app.source)
      return;

    Node& node = src.getData(Galois::Graph::NONE);
    assert(node.height != height);
    if (height < node.height && node.height < app.numNodes)
      node.height = app.numNodes;
  }
};

static void gapAt(int height) {
  using namespace GaloisRuntime::WorkList;
  typedef LocalQueues<ChunkedLIFO<1024>, LIFO<> > WL;
  Galois::for_each<WL>(app.graph.active_begin(), app.graph.active_end(),
      CleanGap(height));
  // TODO
  // gapCounters[height + 1: numNodes] = reset
}

struct GlobalRelabel {
  /**
   * Do reverse BFS on residual graph.
   */
  template<typename Context>
  void operator()(const GNode& src, Context& ctx) {
    for (Graph::neighbor_iterator
        ii = app.graph.neighbor_begin(src, Galois::Graph::CHECK_CONFLICT, ctx),
        ee = app.graph.neighbor_end(src, Galois::Graph::CHECK_CONFLICT, ctx);
        ii != ee; ++ii) {
      GNode dst = *ii;
      if (app.graph.getEdgeData(dst, src, Galois::Graph::NONE) > 0) {
        Node& node = dst.getData(Galois::Graph::NONE);
        int newHeight = src.getData(Galois::Graph::NONE).height + 1;
        if (newHeight < node.height) {
          node.height = newHeight;
          ctx.push(dst);
        }
      }
    }
  }
};

template<typename C>
struct FindWork {
  C& wl;
  FindWork(C& _wl) : wl(_wl) { }

  template<typename Context>
  void operator()(const GNode& src, Context&) {
    Node& node = src.getData(Galois::Graph::NONE);
    if (src == app.sink || src == app.source || node.height > app.numNodes)
      return;
    if (node.excess > 0)
      wl.push_back(src);
    // TODO incrementGap(node.height)
  }
};

template<typename C>
static void globalRelabel(C& newWork) {
  // TODO could parallelize this too
  for (Graph::active_iterator ii = app.graph.active_begin(),
      ee = app.graph.active_end(); ii != ee; ++ii) {
    Node& node = ii->getData(Galois::Graph::NONE);
    node.height = app.numNodes;
    node.current = 0;
    if (*ii == app.sink)
      node.height = 0;
  }

  // TODO
  // reset gapCounters

  typedef GaloisRuntime::WorkList::dChunkedLIFO<8> WL;
  std::vector<GNode> single;
  single.push_back(app.sink);
  Galois::for_each<WL>(single.begin(), single.end(), GlobalRelabel());

  Galois::for_each<WL>(app.graph.active_begin(), app.graph.active_end(),
      FindWork<std::vector<GNode> >(newWork));

  std::cout 
    << " Flow after global relabel: "
    << app.sink.getData().excess << "\n";
}

struct Process {
  typedef int tt_needs_parallel_pause;

  template<typename Context>
  void operator()(const GNode& src, Context& ctx) {
    int increment = 1;
    if (discharge<Context>(src, ctx)) {
      increment += BETA;
    }

    // TODO if value >= globalRelabelInterval : globalRelabel
  }

  template<typename Context>
  void updateGap(int height, int delta, Context& ctx) {
    // TODO
    // atomic {
    //   if (gapCounters[height] += delta == 0) {
    //     gapAt(height);
    //   }

  }

  template<typename Context>
  bool discharge(const GNode& src, Context& ctx) {
    Node& node = src.getData(Galois::Graph::CHECK_CONFLICT);
    int prevHeight = node.height;
    bool relabeled = false;

    if (node.excess == 0 || node.height >= app.numNodes) {
      return false;
    }

    while (true) {
      Galois::Graph::MethodFlag flag =
        relabeled ? Galois::Graph::NONE : Galois::Graph::CHECK_CONFLICT;
      bool finished = false;
      int current = 0;

      for (Graph::neighbor_iterator ii = app.graph.neighbor_begin(src, flag),
          ee = app.graph.neighbor_end(src, flag);
          ii != ee; ++ii, ++current) {
        GNode dst = *ii;
        int cap = app.graph.getEdgeData(src, dst, Galois::Graph::NONE);
        if (cap == 0 || current < node.current) 
          continue;

        Node& dnode = dst.getData(Galois::Graph::NONE);
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

      updateGap<Context>(prevHeight, -1, ctx);

      if (node.height == app.numNodes)
        break;

      updateGap<Context>(node.height, 1, ctx);
      prevHeight = node.height;
    }

    return relabeled;
  }

  void relabel(const GNode& src) {
    int minHeight = std::numeric_limits<int>::max();
    int minEdge;

    int current = 0;
    for (Graph::neighbor_iterator 
        ii = app.graph.neighbor_begin(src, Galois::Graph::NONE),
        ee = app.graph.neighbor_end(src, Galois::Graph::NONE);
        ii != ee; ++ii, ++current) {
      GNode dst = *ii;
      int cap = app.graph.getEdgeData(src, dst, Galois::Graph::NONE);
      if (cap > 0) {
        Node& dnode = dst.getData(Galois::Graph::NONE);
        if (dnode.height < minHeight) {
          minHeight = dnode.height;
          minEdge = current;
        }
      }
    }

    assert(minHeight != std::numeric_limits<int>::max());
    ++minHeight;

    Node& node = src.getData(Galois::Graph::NONE);
    if (minHeight < app.numNodes) {
      node.height = minHeight;
      node.current = minEdge;
    } else {
      node.height = app.numNodes;
    }
  }
};

static void initializeGraph(const char* inputFile, int sourceId, int sinkId, Config *newApp) {
  typedef Galois::Graph::LC_FileGraph<int, int> ReaderGraph;
  typedef ReaderGraph::GraphNode ReaderGNode;

  assert(sourceId != sinkId);

  ReaderGraph reader;
  reader.structureFromFile(inputFile);
  reader.emptyNodeData();

  // Assign ids to ReaderGNodes
  newApp->numNodes = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(),
      ee = reader.active_end(); ii != ee; ++ii, ++newApp->numNodes) {
    ReaderGNode src = *ii;
    reader.getData(src) = newApp->numNodes;
  }

  // Create dense map between ids and GNodes
  std::vector<GNode> nodes;
  nodes.resize(newApp->numNodes);
  for (int i = 0; i < newApp->numNodes; ++i) {
    Node node;

    if (i == sourceId) {
      node.height = newApp->numNodes;
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
  newApp->numEdges = 0;
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
      ++newApp->numEdges;
      // Add reverse edge if not already there
      if (!reader.has_neighbor(rdst, rsrc)) {
        newApp->graph.addEdge(nodes[rdstId], nodes[rsrcId], 0);
        ++newApp->numEdges;
      }
    }
  }
}

static void initializeGaps() {
  for (Graph::active_iterator ii = app.graph.active_begin(),
      ee = app.graph.active_end(); ii != ee; ++ii) {
    GNode src = *ii;
    Node& node = src.getData();
    if (src != app.source && src != app.sink) {
      // TODO increment gap for node.height
    }
  }
}

template<typename C>
static void initializePreflow(C& initial) {
  for (Graph::neighbor_iterator ii = app.graph.neighbor_begin(app.source),
      ee = app.graph.neighbor_end(app.source); ii != ee; ++ii) {
    GNode dst = *ii;
    int cap = app.graph.getEdgeData(app.source, dst);
    reduceCapacity(app.source, dst, cap);
    Node& node = dst.getData();
    node.excess += cap;
    if (cap > 0)
      initial.push_back(dst);
  }
}

struct Indexer : std::binary_function<GNode, int, int> {
  int operator()(const GNode& node) const {
    // TODO Check if conflicts are caught
    return app.numNodes - node.getData(Galois::Graph::NONE).height;
  }
};

int main(int argc, const char** argv) {
  std::cout << "Typetrait: " 
    << Galois::needs_parallel_pause<Process>::value << "\n";

  std::vector<const char*> args = parse_command_line(argc, argv, help);
  if (args.size() < 3) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  const char* inputFile = args[0];
  int sourceId = atoi(args[1]);
  int sinkId = atoi(args[2]);

  assert(sourceId >= 0 && sinkId >= 0);
  initializeGraph(inputFile, sourceId, sinkId, &app);
  assert(sourceId < app.numNodes && sinkId < app.numNodes);
  
  if (args.size() > 3)
    app.globalRelabelInterval = atoi(args[3]);
  else
    app.globalRelabelInterval = app.numNodes * ALPHA + app.numEdges;

  std::cout << "global relabel interval: " << app.globalRelabelInterval << "\n";

  initializeGaps();
  std::vector<GNode> initial;
  initializePreflow(initial);

  using namespace GaloisRuntime::WorkList;
  typedef dChunkedFIFO<16> Chunk;
  typedef OrderedByIntegerMetric<Indexer,Chunk> WL;
  Galois::Launcher::startTiming();
  Galois::for_each<Chunk>(initial.begin(), initial.end(), Process());
  Galois::Launcher::stopTiming();

  std::cout << "Flow is " << app.sink.getData().excess << "\n";
  
  if (!skipVerify) {
    Config orig;
    initializeGraph(inputFile, sourceId, sinkId, &orig);
    verify(orig);
    std::cout << "(Partially) Verified\n";
  }

  return 0;
}


#include <algorithm>
#include <limits>
#include <set>
#include "Galois/Launcher.h"
#include "Galois/Galois.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/FileGraph.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"
#include "Galois/TypeTraits.h"

static const char* name = "Preflow Push";
static const char* description = "Finds the maximum flow in a network using the preflow push technique\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/preflowpush.html";
static const char* help = "<input file> <source id> <sink id>";

/**
 * Parameters from the original Goldberg algorithm to control when global
 * relabeling occurs. For comparison purposes, we keep them the same as
 * before, but it is possible to achieve much better performance by adjusting
 * the global relabel frequency.
 */
static const int ALPHA = 6;
static const int BETA = 12;

struct Node {
  size_t excess;
  unsigned height;
  unsigned current;

  Node() : excess(0), height(1), current(0) { }
};

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "(excess: " << n.excess << ", height: " << n.height << ", current: " << n.current << ")";
  return os;
}

typedef Galois::Graph::FirstGraph<Node, unsigned int, true> Graph;
typedef Graph::GraphNode GNode;

Graph* graph;
GNode sink;
GNode source;
size_t numNodes;
size_t numEdges;
size_t globalRelabelInterval;

static void checkAugmentingPath() {
  std::set<GNode> visited;
  std::deque<GNode> queue;

  visited.insert(source);
  queue.push_back(source);

  while (!queue.empty()) {
    GNode& src = queue.front();
    queue.pop_front();
    for (Graph::neighbor_iterator ii = graph->neighbor_begin(src), ee = graph->neighbor_end(src);
        ii != ee; ++ii) {
      GNode dst = *ii;
      if (visited.find(dst) == visited.end() && graph->getEdgeData(src, dst) > 0) {
        visited.insert(dst);
        queue.push_back(dst);
      }
    }
  }

  if (visited.find(sink) != visited.end()) {
    assert(false && "Augmenting path exisits");
    abort();
  }
}

static void reduceCapacity(GNode& src, GNode& dst, unsigned amount) {
  unsigned& cap1 = graph->getEdgeData(src, dst, Galois::Graph::NONE);
  unsigned& cap2 = graph->getEdgeData(dst, src, Galois::Graph::NONE);
  cap1 -= amount;
  cap2 += amount;
}

struct CleanGap {
  unsigned height;
  CleanGap(unsigned _height) : height(_height) { }

  template<typename Context>
  void operator()(GNode& src, Context& ctx) {
    if (src == sink || src == source)
      return;

    Node& node = src.getData(Galois::Graph::NONE);
    assert(node.height != height);
    if (height < node.height && node.height < numNodes)
      node.height = numNodes;
  }
};

static void gapAt(unsigned height) {
  using namespace GaloisRuntime::WorkList;
  typedef LocalQueues<ChunkedLIFO<1024>, LIFO<> > WL;
  Galois::for_each<WL>(graph->active_begin(), graph->active_end(), CleanGap(height));
  // TODO
  // gapCounters[height + 1: numNodes] = reset
}

struct GlobalRelabel {
  // Reverse bfs on residual graph
  template<typename Context>
  void operator()(GNode& src, Context& ctx) {
    for (Graph::neighbor_iterator ii = graph->neighbor_begin(src, Galois::Graph::CHECK_CONFLICT, ctx),
        ee = graph->neighbor_end(src, Galois::Graph::CHECK_CONFLICT, ctx);
        ii != ee; ++ii) {
      GNode dst = *ii;
      if (graph->getEdgeData(dst, src, Galois::Graph::NONE) > 0) {
        Node& node = dst.getData(Galois::Graph::NONE);
        unsigned newHeight = src.getData(Galois::Graph::NONE).height + 1;
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
  void operator()(GNode& src, Context&) {
    Node& node = src.getData(Galois::Graph::NONE);
    if (src == sink || src == source || node.height > numNodes)
      return;
    if (node.excess > 0)
      wl.push_back(src);
    // TODO incrementGap(node.height)
  }
};

template<typename C>
static void globalRelabel(C& newWork) {
  // TODO could parallelize this too
  for (Graph::active_iterator ii = graph->active_begin(), ee = graph->active_end();
      ii != ee; ++ii) {
    Node& node = ii->getData(Galois::Graph::NONE);
    node.height = numNodes;
    node.current = 0;
    if (*ii == sink)
      node.height = 0;
  }

  // TODO
  // reset gapCounters

  typedef GaloisRuntime::WorkList::dChunkedLIFO<8> WL;
  std::vector<GNode> single;
  single.push_back(sink);
  Galois::for_each<WL>(single.begin(), single.end(), GlobalRelabel());

  Galois::for_each<WL>(graph->active_begin(), graph->active_end(), FindWork<std::vector<GNode> >(newWork));

  std::cout << " Flow after global relabel: " << sink.getData().excess << "\n";
}

struct Process {
  typedef int tt_needs_parallel_pause;

  template<typename Context>
  void operator()(GNode& src, Context& ctx) {
    unsigned increment = 1;
    if (discharge<Context>(src, ctx)) {
      increment += BETA;
    }

    // TODO if value >= globalRelabelInterval : globalRelabel
  }

  template<typename Context>
  void updateGap(unsigned height, int delta, Context& ctx) {
    // TODO
    // atomic {
    //   if (gapCounters[height] += delta == 0) {
    //     gapAt(height);
    //   }

  }

  template<typename Context>
  bool discharge(GNode& src, Context& ctx) {
    Node& node = src.getData(Galois::Graph::CHECK_CONFLICT);
    unsigned prevHeight = node.height;
    bool relabeled = false;

    if (node.excess == 0 || node.height >= numNodes) {
      return false;
    }

    while (true) {
      Galois::Graph::MethodFlag flag = relabeled ? Galois::Graph::NONE : Galois::Graph::CHECK_CONFLICT;
      bool finished = false;
      unsigned current = 0;

      for (Graph::neighbor_iterator ii = graph->neighbor_begin(src, flag),
          ee = graph->neighbor_end(src, flag);
          ii != ee; ++ii, ++current) {
        GNode dst = *ii;
        unsigned cap = graph->getEdgeData(src, dst, Galois::Graph::NONE);
        if (cap > 0 && current >= node.current) {
          unsigned amount = 0;
          Node& dnode = dst.getData(Galois::Graph::NONE);
          if (node.height - 1 == dnode.height) {
            // Push flow
            amount = std::min(static_cast<unsigned>(node.excess), cap);
            reduceCapacity(src, dst, amount);
            // Only add once
            if (dst != sink && dst != source && dnode.excess == 0) {
              ctx.push(dst);
            }
            node.excess -= amount;
            dnode.excess += amount;
            if (node.excess == 0) {
              finished = true;
              node.current = current;
              break;
            }
          }
        }
      }

      if (finished)
        break;

      relabel(src);
      relabeled = true;

      updateGap<Context>(prevHeight, -1, ctx);

      if (node.height == numNodes)
        break;

      updateGap<Context>(node.height, 1, ctx);
      prevHeight = node.height;
    }

    return relabeled;
  }

  void relabel(GNode& src) {
    unsigned minHeight = std::numeric_limits<unsigned>::max();
    unsigned minEdge;

    unsigned current = 0;
    for (Graph::neighbor_iterator ii = graph->neighbor_begin(src, Galois::Graph::NONE),
        ee = graph->neighbor_end(src, Galois::Graph::NONE);
        ii != ee; 
        ++ii, ++current) {
      GNode dst = *ii;
      unsigned cap = graph->getEdgeData(src, dst, Galois::Graph::NONE);
      if (cap > 0) {
        Node& dnode = dst.getData(Galois::Graph::NONE);
        if (dnode.height < minHeight) {
          minHeight = dnode.height;
          minEdge = current;
        }
      }
    }

    assert(minHeight != std::numeric_limits<unsigned>::max());
    ++minHeight;

    Node& node = src.getData(Galois::Graph::NONE);
    if (minHeight < numNodes) {
      node.height = minHeight;
      node.current = minEdge;
    } else {
      node.height = numNodes;
    }
  }
};

static void initializeGraph(const char* inputFile, unsigned sourceId, unsigned sinkId) {
  typedef Galois::Graph::LC_FileGraph<unsigned, unsigned> ReaderGraph;
  typedef ReaderGraph::GraphNode ReaderGNode;

  assert(sourceId != sinkId);

  ReaderGraph reader;
  reader.structureFromFile(inputFile);
  reader.emptyNodeData();

  // Assign ids to ReaderGNodes
  numNodes = 0;
  for (ReaderGraph::active_iterator ii = reader.active_begin(), ee = reader.active_end();
      ii != ee;
      ++ii, ++numNodes) {
    ReaderGNode src = *ii;
    reader.getData(src) = numNodes;
  }

  graph = new Graph();

  // Create dense map between ids and GNodes
  std::vector<GNode> nodes;
  nodes.resize(numNodes);
  for (size_t i = 0; i < numNodes; ++i) {
    Node node;

    if (i == sourceId) {
      node.height = numNodes;
    }

    GNode src = graph->createNode(node);
    graph->addNode(src);
    if (i == sourceId) {
      source = src;
    } else if (i == sinkId) {
      sink = src;
    }
    nodes[i] = src;
  }

  // Create edges
  for (ReaderGraph::active_iterator ii = reader.active_begin(), ee = reader.active_end();
      ii != ee; ++ii) {
    ReaderGNode rsrc = *ii;
    unsigned rsrcId = reader.getData(rsrc);
    for (ReaderGraph::neighbor_iterator jj = reader.neighbor_begin(rsrc), ff = reader.neighbor_end(rsrc);
        jj != ff; ++jj) {
      ReaderGNode rdst = *jj;
      unsigned rdstId = reader.getData(rdst);
      graph->addEdge(nodes[rsrcId], nodes[rdstId], reader.getEdgeData(rsrc, rdst));
      ++numEdges;
      // Add reverse edge if not already there
      if (!reader.has_neighbor(rdst, rsrc)) {
        graph->addEdge(nodes[rdstId], nodes[rsrcId], 0);
        ++numEdges;
      }
    }
  }

  globalRelabelInterval = numNodes * ALPHA + numEdges;
}

static void initializeGaps() {
  for (Graph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
    GNode src = *ii;
    Node& node = src.getData();
    if (src != source && src != sink) {
      // TODO increment gap for node.height
    }
  }
}

template<typename C>
static void initializePreflow(C& initial) {
  for (Graph::neighbor_iterator ii = graph->neighbor_begin(source), ee = graph->neighbor_end(source);
      ii != ee; ++ii) {
    GNode dst = *ii;
    unsigned cap = graph->getEdgeData(source, dst);
    reduceCapacity(source, dst, cap);
    Node& node = dst.getData();
    node.excess += cap;
    if (cap > 0)
      initial.push_back(dst);
  }
}

struct Indexer : std::binary_function<GNode, unsigned, unsigned> {
  unsigned operator()(GNode& node) const {
    // TODO Check if conflicts are caught
    return numNodes - node.getData(Galois::Graph::NONE).height;
  }
};

int main(int argc, const char** argv) {
  std::cout << "Typetrait: " << Galois::needs_parallel_pause<Process>::value << "\n";
  std::vector<const char*> args = parse_command_line(argc, argv, help);
  if (args.size() != 3) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  const char* inputFile = args[0];
  unsigned sourceId = atoi(args[1]);
  unsigned sinkId = atoi(args[2]);

  initializeGraph(inputFile, sourceId, sinkId);
  
  std::cout << "global relabel interval: " << globalRelabelInterval << "\n";

  initializeGaps();
  std::vector<GNode> initial;
  initializePreflow(initial);

  using namespace GaloisRuntime::WorkList;
  typedef dChunkedFIFO<16> Chunk;
  typedef OrderedByIntegerMetric<Indexer,Chunk> WL;
  Galois::Launcher::startTiming();
  Galois::for_each<Chunk>(initial.begin(), initial.end(), Process());
  Galois::Launcher::stopTiming();

  std::cout << "Flow is " << sink.getData().excess << "\n";
  
  if (!skipVerify) {

  }

  delete graph;
  
  return 0;
}

#if 0
  private static boolean validEdgeCap(int scap, int dcap, int maxscap, int maxdcap) {
    return (maxscap - scap + maxdcap - dcap) == 0 && scap <= maxscap + maxdcap && dcap <= maxscap + maxdcap
        && scap >= 0 && dcap >= 0;
  }

  @SuppressWarnings("unused")
  private void checkFlows() {
    graph.map(new LambdaVoid<GNode<Node>>() {
      @Override
      public void call(GNode<Node> src) {
        final MutableInteger inflow = new MutableInteger();
        final MutableInteger outflow = new MutableInteger();

        src.map(new Lambda2Void<GNode<Node>, GNode<Node>>() {
          @Override
          public void call(GNode<Node> dst, GNode<Node> src) {
            Edge e1 = graph.getEdgeData(src, dst);
            Edge e2 = graph.getEdgeData(dst, src);

            int scap = e1.cap;
            int dcap = e2.cap;
            int maxscap = e1.ocap;
            int maxdcap = e2.ocap;

            if (!validEdgeCap(scap, dcap, maxscap, maxdcap)) {
              throw new IllegalStateException("edge values are inconsistent: " + toStringEdge(src, dst));
            }
            if (maxscap > scap) {
              outflow.add(maxscap - scap);
            } else if (maxdcap > dcap) {
              inflow.add(maxdcap - dcap);
            }
          }
        }, src);

        Node node = src.getData();

        if (node.isSource) {
          if (inflow.get() > 0)
            throw new IllegalStateException("source has inflow");
        } else if (node.isSink) {
          if (outflow.get() > 0)
            throw new IllegalStateException("sink has outflow");
        } else {
          if (node.excess != 0)
            throw new IllegalStateException("node " + toStringNode(src) + " still has excess flow");
          int flow = inflow.get() - outflow.get();
          if (flow != 0)
            throw new IllegalStateException("node " + toStringNode(src) + " does not conserve flow: " + flow);
        }
      }
    });
  }

  private void checkFlowsForCut() {
    // Check psuedoflow
    // XXX: Currently broken
    // for (int i = 0; i < numNodes; i++) {
    // for (int j = nodes[i]; j < nodes[i+1]; j++) {
    // int edgeIdx = edges[j];
    // int ocap = orig.getCapacity(edgeIdx, i);
    // if (ocap <= 0)
    // continue;
    // // Original edge
    // int dst = getMate(edgeIdx, i);
    // int icap = getCapacity(edgeIdx, i);
    // int dcap = getCapacity(edgeIdx, dst);
    // if (icap + dcap != ocap || icap < 0 || dcap < 0) {
    // throw new IllegalStateException("Incorrect flow at " +
    // toStringEdge(edgeIdx) + " " + ocap);
    // }
    // }
    // }

    // Check conservation
    graph.map(new LambdaVoid<GNode<Node>>() {
      @Override
      public void call(GNode<Node> src) {
        Node node = src.getData();

        if (node.isSource || node.isSink) {
          return;
        }

        if (node.excess < 0)
          throw new IllegalStateException("Excess at " + toStringNode(src));

        final MutableInteger sum = new MutableInteger();

        src.map(new Lambda2Void<GNode<Node>, GNode<Node>>() {
          @Override
          public void call(GNode<Node> dst, GNode<Node> src) {
            int ocap = graph.getEdgeData(src, dst).ocap;
            int delta = 0;
            if (ocap > 0) {
              delta -= ocap - graph.getEdgeData(src, dst).cap;
            } else {
              delta += graph.getEdgeData(src, dst).cap;
            }
            sum.add(delta);
          }
        }, src);

        if (node.excess != sum.get())
          throw new IllegalStateException("Not pseudoflow " + node.excess + " != " + sum + " at node " + node.id);
      }
    });
  }

  private void checkHeights() {
    graph.map(new LambdaVoid<GNode<Node>>() {
      @Override
      public void call(GNode<Node> src) {
        src.map(new Lambda2Void<GNode<Node>, GNode<Node>>() {
          @Override
          public void call(GNode<Node> dst, GNode<Node> src) {
            int sh = src.getData().height;
            int dh = dst.getData().height;
            int cap = graph.getEdgeData(src, dst).cap;
            if (cap > 0 && sh > dh + 1) {
              throw new IllegalStateException("height violated at " + src + " with " + toStringEdge(src, dst));
            }
          }
        }, src);
      }
    });
  }

  private void checkMaxFlow() {
    Boolean fullVerify = SystemProperties.getBooleanProperty("verify.full", false);
    double expected = SystemProperties.getDoubleProperty("verify.result", Double.MIN_VALUE);
    if (fullVerify || expected == Double.MIN_VALUE) {
      // checkFlows(orig);
      checkFlowsForCut();
      checkHeights();
      checkAugmentingPathExistence();
    } else {
      double result = sink.getData().excess;
      if (result != expected) {
        throw new IllegalStateException("Inconsistent flows: " + expected + " != " + result);
      }
    }
  }
#endif

/*
 * PageRank.cpp
 *
 */

#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/IO/gr.h"

#include "Galois/Graphs/Serialize.h"
#include "Galois/Graphs/FileGraph.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
using namespace std;

static const char* name = "Page Rank";
static const char* description = "Computes page ranks a la Page and Brin\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/pagerank.html";
static const char* help = "<input file> [max_iterations]";

// damping factor: prob that user will continue to next link
static const double alpha = 0.85;
static const double tolerance = 0.00001;
static unsigned int max_iterations = 10;

struct Node {
  double v0;
  double v1;
  unsigned int id;
  bool hasNoOuts;
  
  Node(int _id = -1, double _v0 = 0) : v0(_v0), v1(0), id(_id), hasNoOuts(false) {}
  std::string toString() {
    std::ostringstream s;
    s << '[' << id << "] v0: " << v0 << " v1: " << v1;
    return s.str();
  }
};

typedef Galois::Graph::FirstGraph<Node, double, true> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

static double getPageRank(Node& data, unsigned int iterations) {
  if ((iterations & 1) == 0)
    return data.v0;
  else
    return data.v1;
}

static void setPageRank(Node& data, unsigned int iterations, double value) {
  if ((iterations & 1) == 0)
    data.v1 = value;
  else
    data.v0 = value;
}

void runBody() {
  unsigned int iterations = 0;
  double max_delta;
  unsigned int numNodes = graph.size();

  do {
    max_delta = std::numeric_limits<double>::min();
    unsigned int small_delta = 0;
    double lost_potential = 0;
    for (Graph::active_iterator src = graph.active_begin(), esrc = graph.active_end();
        src != esrc; ++src) {
      double value = 0;
      for (Graph::neighbor_iterator dst = graph.neighbor_begin(*src, Galois::Graph::NONE), 
           edst = graph.neighbor_end(*src, Galois::Graph::NONE);
           dst != edst; ++dst) {
        double w = graph.getEdgeData(*src, *dst, Galois::Graph::NONE);
        Node& ddata = graph.getData(*dst, Galois::Graph::NONE);
        value += getPageRank(ddata, iterations) * w;
      }
        
      if (alpha > 0)
        value = value * (1 - alpha) + 1/numNodes * alpha;

      Node& sdata = graph.getData(*src, Galois::Graph::NONE);
      if (sdata.hasNoOuts) {
        lost_potential += getPageRank(sdata, iterations);
      }
      
      double diff = abs(value - getPageRank(sdata, iterations));
      if (diff > max_delta)
        max_delta = diff;
      if (diff < tolerance)
        small_delta++;
      setPageRank(sdata, iterations, value);
    }

    // Redistribute lost potential
    if (lost_potential > 0) {
      for (Graph::active_iterator src = graph.active_begin(), esrc = graph.active_end();
          src != esrc; ++src) {
        Node& sdata = graph.getData(*src, Galois::Graph::NONE);
        double value = getPageRank(sdata, iterations);
        setPageRank(sdata, iterations, value + (1 - alpha) * (lost_potential * 1/numNodes));
      }
    }

    std::cout << "iteration: " << iterations
              << " max delta: " << max_delta
              << " small delta: " << small_delta
              << " (" << small_delta / (float) numNodes << ")"
              << "\n";
    iterations++;
  } while (iterations < max_iterations && max_delta > tolerance);
}

struct process {
  template<typename ContextTy>
  void __attribute__((noinline)) operator()(GNode& node, ContextTy& lwl) {
  }
};

void runBodyParallel(const GNode src) {
  using namespace GaloisRuntime::WorkList;

  ChunkedFIFO<GNode, 32> wl;

  unsigned int iterations = 0;
  double max_delta = 0;
  do {
    Galois::for_each(wl, process());
    iterations++;
    std::cout << "iteration: " << iterations << " max delta: " << max_delta << "\n";
  } while (iterations < max_iterations && max_delta > tolerance);
}


bool verify() {
  return true;
}

static void makeGraph(const char* input) {
  typedef Galois::Graph::LC_FileGraph<Node, void> InGraph;
  typedef InGraph::GraphNode InGNode;
  InGraph in_graph;
  Timer phase;

  phase.start();
  in_graph.structureFromFile(input);
  std::cout << "Read " << in_graph.size() << " nodes\n";
  phase.stop();
  GaloisRuntime::reportStat("Read1", phase.get());
  
  // TODO(ddn): Bag map
  phase.start();
  typedef std::pair<InGNode, double> Element;
  typedef std::vector<Element> Elements;
  typedef std::vector<Elements> Map;
  
  Map in_edges(in_graph.size());
  std::vector<bool> has_out_edges(in_graph.size());
  for (InGraph::active_iterator src = in_graph.active_begin(), esrc = in_graph.active_end();
      src != esrc; ++src) {
    int neighbors = std::distance(in_graph.neighbor_begin(*src, Galois::Graph::NONE), 
           in_graph.neighbor_end(*src, Galois::Graph::NONE));
    for (InGraph::neighbor_iterator dst = in_graph.neighbor_begin(*src, Galois::Graph::NONE), 
         edst = in_graph.neighbor_end(*src, Galois::Graph::NONE);
         dst != edst; ++dst) {
      Element e(*src, 1.0/neighbors);
      in_edges[*dst].push_back(e);
    }
    has_out_edges[*src] = neighbors != 0;
  }
  phase.stop();
  GaloisRuntime::reportStat("Read2", phase.get());

  // TODO(ddn): better way of making
  phase.start();
  unsigned int id = 0;
  std::vector<GNode> nodes(in_graph.size());
  for (Map::iterator i = in_edges.begin(), ei = in_edges.end(); i != ei; ++i) {
    Node n(id, has_out_edges[id]);
    GNode node = graph.createNode(n);
    graph.addNode(node);
    nodes[id] = node;
    id++;
  }

  id = 0;
  for (Map::iterator i = in_edges.begin(), ei = in_edges.end(); i != ei; ++i) {
    GNode src = nodes[id];
    for (Elements::iterator j = i->begin(), ej = i->end(); j != ej; ++j) {
      graph.addEdge(src, nodes[j->first], j->second);
    }
    id++;
  }
  phase.stop();
  GaloisRuntime::reportStat("Read3", phase.get());
}

void printTop(int topn) {
  for (Graph::active_iterator src = graph.active_begin(), esrc = graph.active_end();
      src != esrc; ++src) {
    // TODO
  }
}

int main(int argc, const char **argv) {
  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 1) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);
  
  const char* inputfile = args[0];
  if (args.size() > 1)
    max_iterations = atoi(args[1]);
 
  Timer phase;
  phase.start();
  makeGraph(inputfile);
  phase.stop();
  GaloisRuntime::reportStat("TimePhase1", phase.get());

  Galois::Launcher::startTiming();
  runBody();
  Galois::Launcher::stopTiming();
  GaloisRuntime::reportStat("Time", Galois::Launcher::elapsedTime());

  printTop(10);
  if (!skipVerify && !verify()) {
    cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}

// vim:sw=2:sts=2:ts=8

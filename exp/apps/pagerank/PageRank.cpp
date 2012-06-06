/** Page rank application -*- C++ -*-
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
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Galois/Graphs/LCGraph.h"

#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin\n";
static const char* url = NULL;

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> max_iterations("maxiter", cll::desc("Maximum iterations"), cll::init(10));


// damping factor: prob that user will continue to next link
static const double alpha = 0.85;
static const double tolerance = 0.00001;

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
    for (Graph::iterator src = graph.begin(),
        esrc = graph.end(); src != esrc; ++src) {
      double value = 0;
      for (Graph::neighbor_iterator
          dst = graph.neighbor_begin(*src, Galois::NONE), 
          edst = graph.neighbor_end(*src, Galois::NONE);
          dst != edst; ++dst) {
        double w = graph.getEdgeData(*src, *dst, Galois::NONE);
        Node& ddata = graph.getData(*dst, Galois::NONE);
        value += getPageRank(ddata, iterations) * w;
      }
        
      if (alpha > 0)
        value = value * (1 - alpha) + 1/numNodes * alpha;

      Node& sdata = graph.getData(*src, Galois::NONE);
      if (sdata.hasNoOuts) {
        lost_potential += getPageRank(sdata, iterations);
      }
      
      double diff = value - getPageRank(sdata, iterations);
      if (diff < 0)
        diff = -diff;

      if (diff > max_delta)
        max_delta = diff;
      if (diff < tolerance)
        small_delta++;
      setPageRank(sdata, iterations, value);
    }

    // Redistribute lost potential
    if (lost_potential > 0) {
      for (Graph::iterator src = graph.begin(),
          esrc = graph.end();
          src != esrc; ++src) {
        Node& sdata = graph.getData(*src, Galois::NONE);
        double value = getPageRank(sdata, iterations);
        double delta = (1 - alpha) * (lost_potential * 1/numNodes);
        setPageRank(sdata, iterations, value + delta);
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
  void operator()(GNode& node, ContextTy& lwl) {
  }
};

void runBodyParallel(const GNode src) {
  using namespace GaloisRuntime::WorkList;

  unsigned int iterations = 0;
  double max_delta = 0;
  do {
    //??? wl never had anything in it
    Galois::for_each<ChunkedFIFO<32> >((GNode*)0,(GNode*)0, process());
    iterations++;
    std::cout << "iteration: " << iterations 
      << " max delta: " << max_delta << "\n";
  } while (iterations < max_iterations && max_delta > tolerance);
}


bool verify() {
  return true;
}

static void makeGraph(const char* input) {
  typedef Galois::Graph::LC_CSR_Graph<Node, int> InGraph;
  typedef InGraph::GraphNode InGNode;
  InGraph in_graph;
  Galois::StatTimer R1T("Read1");

  R1T.start();
  in_graph.structureFromFile(input);
  std::cout << "Read " << in_graph.size() << " nodes\n";
  R1T.stop();
  
  // TODO(ddn): Bag map
  Galois::StatTimer R2T("Read2");
  R2T.start();
  typedef std::pair<InGNode, double> Element;
  typedef std::vector<Element> Elements;
  typedef std::vector<Elements> Map;
  
  Map in_edges(in_graph.size());
  std::vector<bool> has_out_edges(in_graph.size());
  for (InGraph::iterator src = in_graph.begin(),
      esrc = in_graph.end();
      src != esrc; ++src) {
    int neighbors =
      std::distance(in_graph.edge_begin(*src, Galois::NONE), 
          in_graph.edge_end(*src, Galois::NONE));

    for (InGraph::edge_iterator
        dst = in_graph.edge_begin(*src, Galois::NONE), 
        edst = in_graph.edge_end(*src, Galois::NONE);
        dst != edst; ++dst) {
      Element e(*src, 1.0/neighbors);
      in_edges[in_graph.getEdgeDst(dst)].push_back(e);
    }
    has_out_edges[*src] = neighbors != 0;
  }
  R2T.stop();

  // TODO(ddn): better way of making
  Galois::StatTimer R3T("Read3");
  R3T.start();
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
  R3T.stop();
}

void printTop(int topn) {
  for (Graph::iterator src = graph.begin(),
      esrc = graph.end();
      src != esrc; ++src) {
    // TODO
  }
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
 
  Galois::StatTimer RT("ReadTotal");
  RT.start();
  makeGraph(filename.c_str());
  RT.stop();

  Galois::StatTimer T;
  T.start();
  runBody();
  T.stop();

  printTop(10);
  if (!skipVerify && !verify()) {
    std::cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}

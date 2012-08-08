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
#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph2.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/Serialize.h"
#include "Galois/Galois.h"

#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <sys/mman.h>

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin\n";
static const char* url = NULL;

enum Phase {
  transpose,
  parallel,
  serial
};

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional, cll::desc("[output file]"));
static cll::opt<unsigned int> max_iterations("max-iterations", cll::desc("Maximum iterations"), cll::init(10));
static cll::opt<Phase> phase(cll::desc("Phase:"),
    cll::values(
      clEnumVal(transpose, "Transpose graph"),
      clEnumVal(parallel, "Compute PageRank in parallel"),
      clEnumVal(serial, "Compute PageRank in serial"),
      clEnumValEnd), cll::init(parallel));

// d is the damping factor. Alpha is the prob that user will do a random jump, i.e., 1 - d
static const double alpha = 1.0 - 0.85;
static const double tolerance = 0.00001;

struct Node {
  double v0;
  double v1;
  unsigned int id;
  bool hasNoOuts;
};

typedef Galois::Graph::LC_CSR_Graph<Node, double> Graph;
typedef Graph::GraphNode GNode;

Graph graph;
unsigned int iterations;

static double getPageRank(Node& data) {
  if ((iterations & 1) == 0)
    return data.v0;
  else
    return data.v1;
}

static void setPageRank(Node& data, double value) {
  if ((iterations & 1) == 0)
    data.v1 = value;
  else
    data.v0 = value;
}

void serialPageRank() {
  iterations = 0;
  double max_delta;
  unsigned int numNodes = graph.size();

  while (true) {
    max_delta = std::numeric_limits<double>::min();
    unsigned int small_delta = 0;
    double lost_potential = 0;
    for (Graph::iterator src = graph.begin(),
        esrc = graph.end(); src != esrc; ++src) {
      double value = 0;
      for (Graph::edge_iterator
          ii = graph.edge_begin(*src, Galois::NONE), 
          ei = graph.edge_end(*src, Galois::NONE);
          ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        double w = graph.getEdgeData(ii);

        Node& ddata = graph.getData(dst, Galois::NONE);
        value += getPageRank(ddata) * w;
      }
       
      // assuming uniform prior probability, i.e., 1 / numNodes
      if (alpha > 0)
        value = value * (1 - alpha) + alpha/numNodes;

      Node& sdata = graph.getData(*src, Galois::NONE);
      if (sdata.hasNoOuts) {
        lost_potential += getPageRank(sdata);
      }
      
      double diff = value - getPageRank(sdata);
      if (diff < 0)
        diff = -diff;

      if (diff > max_delta)
        max_delta = diff;
      if (diff < tolerance)
        small_delta++;
      setPageRank(sdata, value);
    }

    // Redistribute lost potential
    if (lost_potential > 0) {
      for (Graph::iterator src = graph.begin(),
          esrc = graph.end();
          src != esrc; ++src) {
        Node& sdata = graph.getData(*src, Galois::NONE);
        double value = getPageRank(sdata);
        // assuming uniform prior probability, i.e., 1 / numNodes
        double delta = (1 - alpha) * (lost_potential / numNodes);
        setPageRank(sdata, value + delta);
      }
    }

    std::cout << "iteration: " << iterations
              << " max delta: " << max_delta
              << " small delta: " << small_delta
              << " (" << small_delta / (float) numNodes << ")"
              << "\n";
    if (iterations < max_iterations && max_delta > tolerance) {
      iterations++;
      continue;
    } else
      break;
  }
}

struct Process {
  template<typename ContextTy>
  void operator()(GNode& node, ContextTy& lwl) {
  }
};

void parallelPageRank() {
  using namespace GaloisRuntime::WorkList;

  double max_delta = 0;
  while (true) {
    //??? wl never had anything in it
    Galois::for_each<ChunkedFIFO<32> >((GNode*)0,(GNode*)0, Process());
    iterations++;

    std::cout << "iteration: " << iterations 
      << " max delta: " << max_delta << "\n";

    if (iterations < max_iterations && max_delta > tolerance) {
      iterations++;
      continue;
    } else
      break;
  }
}


//! Transpose in-edges to out-edges
static void transposeGraph() {
  typedef Galois::Graph::LC_CSR_Graph<size_t, void> InputGraph;
  typedef InputGraph::GraphNode InputNode;
  typedef Galois::Graph::FirstGraph<size_t, double, true> OutputGraph;
  typedef OutputGraph::GraphNode OutputNode;

  InputGraph input;
  OutputGraph output;
  input.structureFromFile(inputFilename);

  std::vector<OutputNode> nodes;
  nodes.resize(input.size());
  size_t node_id = 0;
  for (InputGraph::iterator ii = input.begin(), ei = input.end(); ii != ei; ++ii, ++node_id) {
    input.getData(*ii) = node_id;
    OutputNode node = output.createNode(node_id);
    output.addNode(node);
    nodes[node_id] = node;
  }

  bool* has_out_edges = new bool[input.size()];
  node_id = 0;
  for (InputGraph::iterator ii = input.begin(), ei = input.end(); ii != ei; ++ii, ++node_id) {
    GNode src = *ii;
    size_t sid = input.getData(src);

    size_t num_neighbors = std::distance(input.edge_begin(src), input.edge_end(src));
    has_out_edges[node_id] = num_neighbors != 0;

    double w = 1.0/num_neighbors;
    for (InputGraph::edge_iterator jj = input.edge_begin(src), ej = input.edge_end(src); jj != ej; ++jj) {
      GNode dst = *jj;
      size_t did = input.getData(dst);

      output.getEdgeData(output.addEdge(nodes[did], nodes[sid])) += w;
    }
  }
  
  outputGraph(outputFilename.c_str(), output);
  std::cout << "Wrote " << outputFilename << "\n";

  std::string nodeFilename = outputFilename + ".node";
  int fd = open(nodeFilename.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd == -1) { perror(__FILE__); abort(); }
  if (write(fd, has_out_edges, sizeof(bool) * input.size()) == -1) {
    perror(__FILE__); abort();
  }
  close(fd);
  delete [] has_out_edges;
  std::cout << "Wrote " << nodeFilename << "\n";
}

static void readGraph() {
  graph.structureFromFile(inputFilename);

  std::string nodeFilename = outputFilename + ".node";
  int fd = open(nodeFilename.c_str(), O_RDONLY);
  size_t length = sizeof(bool) * graph.size();
  if (fd == -1) { perror(__FILE__); abort(); }
  void *m = mmap(0, length, PROT_READ, MAP_PRIVATE, fd, 0);
  if (m == MAP_FAILED) { perror(__FILE__); abort(); }
  bool* has_out_edges = reinterpret_cast<bool*>(m);

  size_t node_id = 0;
  double initial = 1.0/graph.size();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++node_id) {
    GNode src = *ii;
    Node& n = graph.getData(src);
    n.v0 = initial;
    n.v1 = 0;
    n.id = node_id;
    n.hasNoOuts = !has_out_edges[node_id];
  }

  munmap(m, length);
  close(fd);
}

//! Make values unique
struct TopPair {
  double value;
  unsigned int id;
  TopPair(double v, unsigned int i): value(v), id(i) { }
  bool operator<(const TopPair& b) const {
    if (value < b.value)
      return true;
    return id < b.id;
  }
};

void printTop(int topn) {
  typedef std::map<TopPair,GNode> Top;
  Top top;

  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    Node& n = graph.getData(src);
    double value = getPageRank(n);
    TopPair key(value, n.id);

    if (top.size() < topn) {
      top[key] = src;
      continue;
    }

    if (top.begin()->first.value < value) {
      top[key] = src;
      top.erase(top.begin());
    }
  }

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (Top::reverse_iterator ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  Galois::StatTimer T;
  Galois::StatTimer RT("ReadTime");
  switch (phase) {
    case transpose:
      RT.start(); transposeGraph(); RT.stop();
      break;
    case parallel:
      RT.start(); readGraph(); RT.stop();
      T.start(); parallelPageRank(); T.stop();
      if (!skipVerify) printTop(10);
      break;
    case serial:
      RT.start(); readGraph(); RT.stop();
      T.start(); serialPageRank(); T.stop();
      if (!skipVerify) printTop(10);
      break;
    default:
      std::cerr << "Unknown option\n";
      abort();
  }

  return 0;
}

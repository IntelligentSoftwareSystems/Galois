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
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/LCGraph.h"

#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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
static cll::opt<unsigned int> max_iterations("max-iterations", cll::desc("Maximum iterations"), cll::init(100));
static cll::opt<Phase> phase(cll::desc("Phase:"),
    cll::values(
      clEnumVal(transpose, "Transpose graph"),
      clEnumVal(parallel, "Compute PageRank in parallel"),
      clEnumVal(serial, "Compute PageRank in serial"),
      clEnumValEnd), cll::init(parallel));

//! d is the damping factor. Alpha is the prob that user will do a random jump, i.e., 1 - d
static const double alpha = 1.0 - 0.85;

//! maximum relative change until we deem convergence
static const double tolerance = 0.001;

struct Node {
  double v0;
  double v1;
  unsigned int id;
  bool hasNoOuts;
};

typedef galois::graphs::LC_CSR_Graph<Node, double> Graph;
typedef Graph::GraphNode GNode;

Graph graph;
unsigned int iterations;

static double getPageRank(Node& data, unsigned int it) {
  if ((it & 1) == 0)
    return data.v0;
  else
    return data.v1;
}

static void setPageRank(Node& data, unsigned int it, double value) {
  if ((it & 1) == 0)
    data.v1 = value;
  else
    data.v0 = value;
}

void serialPageRank() {
  iterations = 0;
  unsigned int numNodes = graph.size();
  double tol = tolerance / numNodes;

  std::cout << "target max delta: " << tol << "\n";

  while (true) {
    double max_delta = std::numeric_limits<double>::min();
    unsigned int small_delta = 0;
    double lost_potential = 0;

    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      GNode src = *ii;
      double value = 0;
      for (Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        double w = graph.getEdgeData(ii);

        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        value += getPageRank(ddata, iterations) * w;
      }
       
      // assuming uniform prior probability, i.e., 1 / numNodes
      if (alpha > 0)
        value = value * (1 - alpha) + alpha/numNodes;

      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      if (sdata.hasNoOuts) {
        lost_potential += getPageRank(sdata, iterations);
      }
      
      double diff = value - getPageRank(sdata, iterations);
      if (diff < 0)
        diff = -diff;

      if (diff > max_delta)
        max_delta = diff;
      if (diff <= tol)
        ++small_delta;
      setPageRank(sdata, iterations, value);
    }

    // Redistribute lost potential
    if (lost_potential > 0) {
      unsigned int next = iterations + 1;
      for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src = *ii;
        Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        double value = getPageRank(sdata, next);
        // assuming uniform prior probability, i.e., 1 / numNodes
        double delta = (1 - alpha) * (lost_potential / numNodes);
        setPageRank(sdata, iterations, value + delta);
      }
    }

    std::cout << "iteration: " << iterations
              << " max delta: " << max_delta
              << " small delta: " << small_delta
              << " (" << small_delta / (float) numNodes << ")"
              << "\n";

    if (iterations < max_iterations && max_delta > tol) {
      ++iterations;
      continue;
    } else
      break;
  }

  if (iterations >= max_iterations) {
    std::cout << "Failed to converge\n";
  }
}

struct Process {
  struct Accum {
    galois::GReduceMax<double> max_delta;
    galois::GAccumulator<unsigned int> small_delta;
    galois::GAccumulator<double> lost_potential;
  };

  Accum& accum;
  double tol;
  double addend;

  Process(Accum& a, double t, unsigned int numNodes): accum(a), tol(t), addend(alpha/numNodes) { }

  void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
    operator()(src);
  }

  void operator()(const GNode& src) const {
    double value = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      double w = graph.getEdgeData(ii);

      Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      value += getPageRank(ddata, iterations) * w;
    }
     
    // assuming uniform prior probability, i.e., 1 / numNodes
    if (alpha > 0)
      value = value * (1 - alpha) + addend;

    Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    if (sdata.hasNoOuts) {
      accum.lost_potential += getPageRank(sdata, iterations);
    }
    
    double diff = value - getPageRank(sdata, iterations);
    if (diff < 0)
      diff = -diff;

    accum.max_delta.update(diff);
    if (diff <= tol)
      accum.small_delta += 1;
    setPageRank(sdata, iterations, value);
  }
};

struct RedistributeLost {
  double delta;
  unsigned int next;

  RedistributeLost(double p): delta((1 - alpha) * p), next(iterations + 1) { }

  void operator()(const GNode& src) const {
    Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    double value = getPageRank(sdata, next);
    // assuming uniform prior probability, i.e., 1 / numNodes
    setPageRank(sdata, iterations, value + delta);
  }
};

void parallelPageRank() {
  iterations = 0;
  unsigned int numNodes = graph.size();
  double tol = tolerance / numNodes;

  std::cout << "target max delta: " << tol << "\n";
  
  while (true) {
    Process::Accum accum;
    galois::do_all(graph.begin(), graph.end(), Process(accum, tol, numNodes));
    double lost_potential = accum.lost_potential.reduce();
    if (lost_potential > 0) {
      galois::do_all(graph.begin(), graph.end(), RedistributeLost(lost_potential / numNodes));
    }

    unsigned int small_delta = accum.small_delta.reduce();
    double max_delta = accum.max_delta.reduce();

    std::cout << "iteration: " << iterations
              << " max delta: " << max_delta
              << " small delta: " << small_delta
              << " (" << small_delta / (float) numNodes << ")"
              << "\n";

    if (iterations < max_iterations && max_delta > tol) {
      ++iterations;
      continue;
    } else
      break;
  }

  if (iterations >= max_iterations) {
    std::cout << "Failed to converge\n";
  }
}


#if 0
//! Transpose in-edges to out-edges
static void transposeGraph() {
  typedef galois::graphs::LC_CSR_Graph<size_t, void> InputGraph;
  typedef InputGraph::GraphNode InputNode;
  typedef galois::graphs::FirstGraph<size_t, double, true> OutputGraph;
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
    InputNode src = *ii;
    size_t sid = input.getData(src);
    assert(sid < input.size());

    size_t num_neighbors = std::distance(input.edge_begin(src), input.edge_end(src));
    has_out_edges[node_id] = num_neighbors != 0;

    double w = 1.0/num_neighbors;
    for (InputGraph::edge_iterator jj = input.edge_begin(src), ej = input.edge_end(src); jj != ej; ++jj) {
      InputNode dst = input.getEdgeDst(*jj);
      size_t did = input.getData(dst);
      assert(did < input.size());

      output.getEdgeData(output.addEdge(nodes[did], nodes[sid])) += w;
    }
  }
  
  outputGraph(outputFilename.c_str(), output);
  std::cout << "Wrote " << outputFilename << "\n";

  std::string nodeFilename = outputFilename + ".node";
  int fd = open(nodeFilename.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd == -1) { perror(nodeFilename.c_str()); abort(); }
  if (write(fd, has_out_edges, sizeof(bool) * input.size()) == -1) {
    perror(outputFilename.c_str()); abort();
  }
  close(fd);
  delete [] has_out_edges;
  std::cout << "Wrote " << nodeFilename << "\n";
}
#else
static void transposeGraph() { abort(); }
#endif

static void readGraph() {
  galois::graphs::readGraph(graph, inputFilename);

  std::string nodeFilename = inputFilename + ".node";
  int fd = open(nodeFilename.c_str(), O_RDONLY);
  size_t length = sizeof(bool) * graph.size();
  if (fd == -1) { perror(nodeFilename.c_str()); abort(); }
  void *m = mmap(0, length, PROT_READ, MAP_PRIVATE, fd, 0);
  if (m == MAP_FAILED) { perror(nodeFilename.c_str()); abort(); }
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
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

void printTop(int topn) {
  typedef std::map<TopPair,GNode> Top;
  Top top;

  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    Node& n = graph.getData(src);
    double value = getPageRank(n, iterations);
    TopPair key(value, n.id);

    if ((int) top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
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
  galois::StatManager statManager;

  galois::StatTimer T;
  galois::StatTimer RT("ReadTime");
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

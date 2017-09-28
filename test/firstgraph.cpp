#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"

#include <iostream>
#include <string>

using SymGraph = galois::graphs::FirstGraph<unsigned int, unsigned int, false>;
using OutGraph = galois::graphs::FirstGraph<unsigned int, unsigned int, true, false>;
using InOutGraph = galois::graphs::FirstGraph<unsigned int, unsigned int, true, true>;

std::string filename;

template<typename Graph>
void initGraph(Graph& g) {
  unsigned int i = 1;
  for (auto n: g) {
    g.getData(n) = i++;
  }
}

template<typename Graph>
unsigned int traverseGraph(Graph& g) {
  unsigned int sum = 0;
  for (auto n: g) {
    auto node = g.getData(n);
    for (auto oe: g.edges(n)) {
      sum += node * g.getEdgeData(oe);
    }
    for (auto ie: g.in_edges(n)) {
      sum -= node * g.getEdgeData(ie);
    }
  }
  return sum;
}

template<typename Graph>
void exp(Graph& g, galois::StatTimer& timer, std::string prompt) {
  timer.start();
  galois::graphs::readGraph(g, filename);
  timer.stop();
  initGraph(g);
  std::cout << prompt << ": sum = " << traverseGraph(g) << std::endl;
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  if (argc < 1) {
    std::cout << "Usage: ./test-firstgraph <input> <num_threads>" << std::endl;
    return 0;
  } 
  filename = argv[1];

  auto numThreads = galois::setActiveThreads(std::stoul(argv[2]));
  std::cout << "Loading " << filename << " with " << numThreads << " threads." << std::endl;

  galois::StatTimer outT("OutGraphTime");
  OutGraph outG;
  exp(outG, outT, "out graph");

  galois::StatTimer symT("SymGraphTime");
  SymGraph symG;
  exp(symG, symT, "symmetric graph");

  galois::StatTimer inoutT("InOutGraphTime");
  InOutGraph inoutG;
  exp(inoutG, inoutT, "in-out graph");

  return 0;
}

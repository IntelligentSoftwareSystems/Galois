#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"

#include <iostream>
#include <string>

using OutGraph = galois::graphs::FirstGraph<unsigned int, unsigned int, true, false>;
using InOutGraph = galois::graphs::FirstGraph<unsigned int, unsigned int, true, true>;
using SymGraph = galois::graphs::FirstGraph<unsigned int, unsigned int, false>;

std::string filename;

template<typename Graph>
void initGraph(Graph& g) {
  unsigned int i = 1;
  for (auto n: g) {
    g.getData(n) = i++;
  }
}

template<typename Graph>
void traverseGraph(Graph& g) {
  unsigned int sum = 0;

  for (auto n: g) {
    for (auto oe: g.edges(n)) {
      sum += g.getData(n) * g.getEdgeData(oe);
    }
  }
  std::cout << "  outgoing sum = " << sum << std::endl;

  for (auto n: g) {
    for (auto ie: g.in_edges(n)) {
      sum -= g.getData(n) * g.getEdgeData(ie);
    }
  }
  std::cout << "  all sum = " << sum << std::endl;
}

template<typename Graph>
void run(Graph& g, galois::StatTimer& timer, std::string prompt) {
  std::cout << prompt << std::endl;
  timer.start();
  galois::graphs::readGraph(g, filename);
  timer.stop();
  initGraph(g);
  traverseGraph(g);
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
  run(outG, outT, "out graph");

  galois::StatTimer inoutT("InOutGraphTime");
  InOutGraph inoutG;
  run(inoutG, inoutT, "in-out graph");

  galois::StatTimer symT("SymGraphTime");
  SymGraph symG;
  run(symG, symT, "symmetric graph");

  return 0;
}

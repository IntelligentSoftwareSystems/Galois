/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/runtime/Profile.h"

#include <iostream>
#include <string>

using OutGraph =
    galois::graphs::MorphGraph<unsigned int, unsigned int, true, false>;
using InOutGraph =
    galois::graphs::MorphGraph<unsigned int, unsigned int, true, true>;
using SymGraph = galois::graphs::MorphGraph<unsigned int, unsigned int, false>;

std::string filename;
std::string statfile;
std::string graphtype;

template <typename Graph>
void initGraph(Graph& g) {
  unsigned int i = 1;
  for (auto n : g) {
    g.getData(n) = i++;
  }
}

template <typename Graph>
void traverseGraph(Graph& g) {
  uint64_t sum = 0;

  for (auto n : g) {
    for (auto oe : g.edges(n)) {
      sum += g.getEdgeData(oe);
    }
  }
  std::cout << "  out sum = " << sum << std::endl;

  for (auto n : g) {
    for (auto ie : g.in_edges(n)) {
      sum -= g.getEdgeData(ie);
    }
  }
  std::cout << "  all sum = " << sum << std::endl;
}

template <typename Graph>
void run(Graph& g, galois::StatTimer& timer, std::string prompt) {
  std::cout << prompt << std::endl;

  galois::graphs::FileGraph f;
  f.fromFileInterleaved<typename Graph::file_edge_data_type>(filename);

  size_t approxGraphSize =
      120 * f.sizeEdges() *
      sizeof(typename Graph::edge_data_type); // 120*|E|*sizeof(E)
  size_t numThreads = galois::getActiveThreads();
  galois::preAlloc(numThreads +
                   approxGraphSize / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  timer.start();
  galois::runtime::profileVtune(
      [&g, &f]() {
        galois::graphs::readGraphDispatch(g, typename Graph::read_tag(), f);
      },
      "Construct MorphGraph");
  timer.stop();

  galois::reportPageAlloc("MeminfoPost");

  initGraph(g);
  traverseGraph(g);
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  if (argc < 4) {
    std::cout << "Usage: ./test-morphgraph <input> <num_threads> "
                 "<out|in-out|symmetric> [stat_file]"
              << std::endl;
    return 0;
  }

  filename  = argv[1];
  graphtype = argv[3];

  auto numThreads = galois::setActiveThreads(std::stoul(argv[2]));
  std::cout << "Loading " << filename << " with " << numThreads << " threads."
            << std::endl;

  if (argc >= 5) {
    galois::runtime::setStatFile(argv[4]);
  }

  if ("out" == graphtype) {
    galois::StatTimer outT("OutGraphTime");
    OutGraph outG;
    run(outG, outT, "out graph");
  } else if ("in-out" == graphtype) {
    galois::StatTimer inoutT("InOutGraphTime");
    InOutGraph inoutG;
    run(inoutG, inoutT, "in-out graph");
  } else if ("symmetric" == graphtype) {
    galois::StatTimer symT("SymGraphTime");
    SymGraph symG;
    run(symG, symT, "symmetric graph");
  }

  galois::runtime::reportParam("Load MorphGraph", "Threads", numThreads);
  galois::runtime::reportParam("Load MorphGraph", "File", filename);
  galois::runtime::reportParam("Load MorphGraph", "Graph Type", graphtype);
  return 0;
}

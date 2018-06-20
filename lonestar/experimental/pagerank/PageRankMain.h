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

#ifndef PAGE_RANK_MAIN_H
#define PAGE_RANK_MAIN_H

#include "Lonestar/BoilerPlate.h"

#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/DoAllWrap.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/FileGraph.h"
#include "galois/runtime/Profile.h"

#include <iostream>
#include <fstream>
#include <set>

static const char* const name = "Page Rank";
static const char* const desc = "Computes Page Rank over a directed graph";
static const char* const url  = "pagerank";

namespace hidden {
// use the command line options below instead of these constants

static const double INIT_PR_VAL = 1.0;

static const double RANDOM_JUMP = 0.15;

static const double TERM_THRESH = 1.0e-3;

} // namespace hidden

static const unsigned DEFAULT_CHUNK_SIZE = 4;

typedef galois::GAccumulator<size_t> ParCounter;

namespace cll = llvm::cl;

static cll::opt<double> initVal("initVal",
                                cll::desc("Initial PageRank Value per node"),
                                cll::init(hidden::INIT_PR_VAL));
static cll::opt<double> tolerance("tolerance",
                                  cll::desc("Termination Threshold tolerance"),
                                  cll::init(hidden::TERM_THRESH));
static cll::opt<double> randJmp("randJmp", cll::desc("Random Jump Probability"),
                                cll::init(hidden::RANDOM_JUMP));
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

template <typename G>
class PageRankMain {

public:
  virtual std::string getVersion() const = 0;

  virtual void initGraph(G& graph) = 0;

  virtual size_t runPageRank(G& graph) = 0;

  virtual void verify(G& graph) {
    // TODO
  }

  void run(int argc, char* argv[]) {
    LonestarStart(argc, argv, name, desc, url);
    galois::StatManager sm;

    G graph;

    galois::StatTimer t_input("time to initialize input:");

    t_input.start();
    initGraph(graph);
    t_input.stop();

    std::cout << "PageRank version: " << getVersion()
              << ", tolerance: " << tolerance << ", initVal: " << initVal
              << "Random Jump: " << randJmp << std::endl;
    std::cout << "input file: " << static_cast<std::string>(inputFile)
              << ", nodes: " << graph.size() << ", edges: " << graph.sizeEdges()
              << std::endl;

    galois::reportPageAlloc("MeminfoPre");
    galois::StatTimer t_run;

    t_run.start();
    const size_t activities = runPageRank(graph);
    t_run.stop();

    galois::reportPageAlloc("MeminfoPost");

    std::cout << "PageRank version: " << getVersion()
              << ", activities executed: " << activities << std::endl;

    galois::StatTimer t_verify("Time to verify:");
    t_verify.start();
    verify(graph);
    t_verify.stop();
  }
};

#endif // PAGE_RANK_MAIN_H

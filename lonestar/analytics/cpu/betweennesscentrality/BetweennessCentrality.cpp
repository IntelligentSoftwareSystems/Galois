/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2020, The University of Texas at Austin. All rights reserved.
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

#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/Utils.h"

////////////////////////////////////////////////////////////////////////////////

constexpr static const char* const REGION_NAME = "BC";

enum Algo { Level = 0, Async, Outer, AutoAlgo };

const char* const ALGO_NAMES[] = {"Level", "Async", "Outer", "Auto"};

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

////////////////////////////////////////////////////////////////////////////////

namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

static cll::opt<std::string> sourcesToUse("sourcesToUse",
                                          cll::desc("Whitespace separated list "
                                                    "of sources in a file to "
                                                    "use in BC"),
                                          cll::init(""));

static cll::opt<unsigned int>
    numOfSources("numOfSources",
                 cll::desc("Number of sources to compute BC on (default all)"),
                 cll::init(0));

static llvm::cl::opt<unsigned int>
    iterLimit("numOfOutSources",
              llvm::cl::desc("Number of sources WITH EDGES "
                             " to compute BC on (default is all); does "
                             "not work with Level BC"),
              llvm::cl::init(0));

static cll::opt<bool>
    singleSourceBC("singleSource",
                   cll::desc("Level: Use for single source BC (default off)"),
                   cll::init(false));

static cll::opt<uint64_t>
    startSource("startNode",
                cll::desc("Level/Outer: Starting source node used for "
                          "betweeness-centrality (default 0); works with "
                          "singleSource flag only"),
                cll::init(0));

static cll::opt<bool>
    output("output", cll::desc("Output BC (Level/Async) (default: false)"),
           cll::init(false));

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm (default value AutoAlgo):"),
    cll::values(clEnumVal(Level, "Level"), clEnumVal(Async, "Async"),
                clEnumVal(Outer, "Outer"),
                clEnumVal(AutoAlgo,
                          "Auto: choose among the algorithms automatically")),
    cll::init(AutoAlgo));

////////////////////////////////////////////////////////////////////////////////

static const char* name = "Betweenness Centrality";
static const char* desc = "Computes betwenness centrality in an unweighted "
                          "graph";

////////////////////////////////////////////////////////////////////////////////

// include implementations for other BCs; here so that it has access to command
// line arguments above at global scope
// @todo not the best coding practice; passing cl in via argument might be
// better

#include "LevelStructs.h"
#include "AsyncStructs.h"
#include "OuterStructs.h"

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, nullptr, &inputFile);

  galois::StatTimer autoAlgoTimer("AutoAlgo_0");
  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  if (algo == AutoAlgo) {
    galois::graphs::FileGraph degreeGraph;
    degreeGraph.fromFile(inputFile);
    degreeGraph.initNodeDegrees();
    autoAlgoTimer.start();
    if (isApproximateDegreeDistributionPowerLaw(degreeGraph)) {
      algo = Async;
    } else {
      algo = Level;
    }
    autoAlgoTimer.stop();
    galois::gInfo("Choosing ", ALGO_NAMES[algo], " algorithm");
  }

  switch (algo) {
  case Level:
    // see LevelStructs.h
    galois::gInfo("Running level BC");
    doLevelBC();
    break;
  case Async:
    // see AsyncStructs.h
    galois::gInfo("Running async BC");
    doAsyncBC();
    break;
  case Outer:
    // see OuterStructs.h
    galois::gInfo("Running outer BC");
    doOuterBC();
    break;
  default:
    GALOIS_DIE("Unknown BC algorithm type");
  }

  totalTime.stop();
  return 0;
}

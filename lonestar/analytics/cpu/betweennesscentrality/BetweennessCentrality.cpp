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

static cll::opt<std::string>
    startNodesFile("startNodesFile",
                   cll::desc("File containing whitespace separated list of "
                             "source nodes; if set, -startNodes is ignored"));
static cll::opt<std::string> startNodesString(
    "startNodes",
    cll::desc("String containing whitespace separated list of source nodes "
              "(default value \"0\"); ignore if -startNodesFile is used"),
    cll::init("0"));
static cll::opt<uint32_t> numStartNodes(
    "numStartNodes",
    cll::desc("Number of source nodes for computing betweenness centrality "
              "(default value 1); if -startNodes or -startNodesFile contain "
              "more sources, then betweenness centrality is recomputed for "
              "each set of -numStartNodes"),
    cll::init(1));

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

  std::vector<uint32_t> startNodes;
  if (!startNodesFile.getValue().empty()) {
    std::ifstream file(startNodesFile);
    if (!file.good()) {
      std::cerr << "Failed to open file: " << startNodesFile << "\n";
      abort();
    }
    startNodes.insert(startNodes.end(), std::istream_iterator<uint64_t>{file},
                      std::istream_iterator<uint64_t>{});
  } else {
    std::istringstream str(startNodesString);
    startNodes.insert(startNodes.end(), std::istream_iterator<uint64_t>{str},
                      std::istream_iterator<uint64_t>{});
  }

  if (numStartNodes > startNodes.size()) {
    galois::gWarn(numStartNodes, " source nodes are not specified; using ",
                  startNodes.size(), " source nodes instead");
    numStartNodes = startNodes.size();
  }

  if ((startNodes.size() % numStartNodes) != 0) {
    galois::gWarn(
        "Ignoring the last ", (startNodes.size() % numStartNodes),
        " source nodes because -numStartNodes does not divide the number of "
        "source nodes specified using -startNodes or -startNodesFile");
    size_t truncate = (startNodes.size() / numStartNodes) * numStartNodes;
    startNodes.resize(truncate);
  }

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

  for (size_t i = 0, end = startNodes.size(); i < end; i += numStartNodes) {
    std::vector<uint32_t> startNodesBatch(
        startNodes.begin() + i, startNodes.begin() + i + numStartNodes);
    switch (algo) {
    case Level:
      galois::gInfo("Running level BC");
      doLevelBC(startNodesBatch);
      break;
    case Async:
      // see AsyncStructs.h
      galois::gInfo("Running async BC");
      doAsyncBC(startNodesBatch);
      break;
    case Outer:
      // see OuterStructs.h
      galois::gInfo("Running outer BC");
      doOuterBC(startNodesBatch);
      break;
    default:
      GALOIS_DIE("Unknown BC algorithm type");
    }
  }

  totalTime.stop();
  return 0;
}

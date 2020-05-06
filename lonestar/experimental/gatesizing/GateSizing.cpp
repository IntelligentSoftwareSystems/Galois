/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "CircuitGraph.h"
#include "CellLib.h"
#include "Verilog.h"
#include "Sdc.h"
#include "StaticTimingAnalysis.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>

const char* name = "Gate Sizing";
const char* desc = 0;
const char* url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputCircuit(cll::Positional, cll::desc("<input .v>"), cll::Required);
static cll::opt<std::string> lib("lib", cll::desc("path to the cell library"),
                                 cll::Required);
static cll::opt<std::string>
    outputCircuit("o", cll::desc("output file for gate-sized .v"),
                  cll::Required);
static cll::opt<std::string> sdcFile("sdc", cll::desc("path to the sdc file"));

void doGateSizing() {}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  // do not call clear() unless you are constructing new instances

  galois::StatTimer T("TotalTime");
  T.start();

  CellLib cellLib;
  cellLib.read(lib);
  cellLib.printDebug();
  std::cout << "parsed cell library" << std::endl;

  VerilogModule vModule;
  vModule.read(inputCircuit, &cellLib);
  vModule.printDebug();
  std::cout << "parsed verilog module" << std::endl;

  CircuitGraph graph;
  graph.construct(vModule);
  graph.initialize();
  graph.print();
  std::cout << "constructed circuit graph" << std::endl;

  //  SDC sdc(&cellLib, &vModule, &graph);
  //  sdc.setConstraints(sdcFile);
  //  graph.print();
  //  std::cout << "set constraints from sdc file to circuit graph" <<
  //  std::endl;

  //  doStaticTimingAnalysis(graph);
  //  graph.print();
  //  std::cout << "finished static timinig analysis" << std::endl;

  //  doGateSizing();
  //  vModule.printDebug();
  //  graph.print();
  //  std::cout << "finished gate sizing" << std::endl;

  T.stop();
  vModule.write(outputCircuit);
  std::cout << "wrote modified verilog module" << std::endl;

  auto gStat = graph.getStatistics();
  std::cout << gStat.first << " nodes, " << gStat.second << " edges"
            << std::endl;
  return 0;
}

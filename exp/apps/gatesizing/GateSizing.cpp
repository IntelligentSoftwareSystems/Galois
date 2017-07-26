/** Connected components -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Size the gates from a cell library for a given circuit to fit the timing 
 * constraint and optimize for area/power.
 *
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "Galois/ParallelSTL.h"
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
const char* url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputCircuit(cll::Positional, cll::desc("<input .v>"), cll::Required);
static cll::opt<std::string> lib("lib", cll::desc("path to the cell library"), cll::Required);
static cll::opt<std::string> outputCircuit("o", cll::desc("output file for gate-sized .v"), cll::Required);
static cll::opt<std::string> sdcFile("sdc", cll::desc("path to the sdc file"));

// do not call clear() unless you are constructing new instances
static CellLib cellLib;
static VerilogModule vModule;
static SDC sdc;

void doGateSizing() {

}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer T("TotalTime");
  T.start();

  cellLib.read(lib);
//  cellLib.printCellLibDebug();
  std::cout << "parsed cell library" << std::endl;

  vModule.read(inputCircuit, &cellLib);
//  vModule.printVerilogModuleDebug();
  std::cout << "parsed verilog module" << std::endl;

  sdc.read(sdcFile, &cellLib);
//  sdc.printSdcDebug();
  std::cout << "parsed sdc file" << std::endl;

  constructCircuitGraph(graph, vModule);
  initializeCircuitGraph(graph, sdc);
  printCircuitGraph(graph);
  std::cout << "constructed circuit graph" << std::endl;

  doStaticTimingAnalysis(graph);
  printCircuitGraph(graph);
  std::cout << "finished static timinig analysis" << std::endl;

  doGateSizing();
//  vModule.printVerilogModuleDebug();
//  printCircuitGraph(graph);
  std::cout << "finished gate sizing" << std::endl;

  T.stop();
  vModule.writeVerilogModule(outputCircuit);
  std::cout << "wrote modified verilog module" << std::endl;

  auto gStat = getCircuitGraphStatistics(graph);
  std::cout << gStat.first << " nodes, " << gStat.second << " edges" << std::endl;
  return 0;
}


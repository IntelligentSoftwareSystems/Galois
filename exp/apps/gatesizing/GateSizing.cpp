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

#include "Verilog.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <unordered_map>

const char* name = "Gate Sizing";
const char* desc = 0;
const char* url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputCircuit(cll::Positional, cll::desc("<input .v>"), cll::Required);
static cll::opt<std::string> lib("lib", cll::desc("path to the cell library"), cll::Required);
static cll::opt<std::string> outputCircuit("out", cll::desc("path to the gate-sized .v"), cll::Required);
static cll::opt<std::string> sdc("sdc", cll::desc("path to the sdc file"));

static double maxDelay = 0.0;
static std::string moduleName;

struct Pin {
  std::string portName;
};

struct Node {
  std::string gateName;
  std::string gateType;
  Pin outPin;
  Pin wireName;
};

struct Edge {
  Pin inPin;
};

//typedef Galois::Graph::FirstGraph<Node, Edge, true, true> Graph;
typedef Galois::Graph::FirstGraph<Node, Edge, true, true> Graph;
typedef Graph::GraphNode GNode;

Graph graph;
std::unordered_map<std::string, GNode> nodeMap;
std::unordered_set<GNode> primaryInputs, primaryOutputs;

void readCellLib() {
  ;
}

void readSDC() {
  if (!sdc.empty()) {
    std::ifstream ifs(sdc);
    if (ifs.is_open()) {
      std::string s1, s2;
      ifs >> s1 >> s2 >> maxDelay;
      std::cout << "maxDelay = " << maxDelay << std::endl;
    } else {
      std::cout << "Cannot open " << sdc << ". Set maxDelay = 0.0" << std::endl;
    }
  }
}

void constructCircuitGraph(VerilogModule& v) {

}

void doGateSizing() {

}

void printGraph() {

}

void writeCircuit() {

}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer T("TotalTime");
  T.start();

  readCellLib();
  VerilogModule vModule(inputCircuit);
  vModule.printVerilogModule();
  readSDC();

  constructCircuitGraph(vModule);
  doGateSizing();
  printGraph();
  writeCircuit();

  T.stop();

  return 0;
}


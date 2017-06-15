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

#include "CellLib.h"
#include "Verilog.h"
#include "Sdc.h"

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
static cll::opt<std::string> outputCircuit("out", cll::desc("path to the gate-sized .v"), cll::Required);
static cll::opt<std::string> sdcFile("sdc", cll::desc("path to the sdc file"));

// do not call clear() unless you are constructing new instances
static CellLib cellLib;
static VerilogModule vModule;
static SDC sdc;

struct Node {
  VerilogPin *pin;
  float slew, capacitance;
  float arrivalTime, requiredTime, slack;
  float internalPower, netPower;
};

struct Edge {
  VerilogWire *wire;
  bool isRise;
};

//typedef Galois::Graph::FirstGraph<Node, Edge, true, true> Graph;
typedef Galois::Graph::FirstGraph<Node, Edge, true, true> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

std::unordered_map<VerilogPin *, GNode> nodeMap;
GNode dummySrc, dummySink;

void constructCircuitGraph() {
  auto unprotected = Galois::MethodFlag::UNPROTECTED;

  dummySrc = graph.createNode();
  graph.addNode(dummySrc, unprotected);
  graph.getData(dummySrc, unprotected).pin = nullptr;

  dummySink = graph.createNode();
  graph.addNode(dummySink, unprotected);
  graph.getData(dummySink, unprotected).pin = nullptr;

  // create nodes for all input pins 
  // and connect dummySrc to them
  for (auto item: vModule.inputs) {
    auto pin = item.second;
    auto n = graph.createNode();
    nodeMap.insert({pin, n});

    graph.addNode(n, unprotected);
    graph.getData(n, unprotected).pin = pin;

    auto e = graph.addMultiEdge(dummySrc, n, unprotected);
    graph.getEdgeData(e).wire = nullptr;
  }

  // create nodes for all output pins 
  // and connect them to dummySink
  for (auto item: vModule.outputs) {
    auto pin = item.second;
    auto n = graph.createNode();
    nodeMap.insert({pin, n});

    graph.addNode(n, unprotected);
    graph.getData(n, unprotected).pin = pin;

    auto e = graph.addMultiEdge(n, dummySink, unprotected);
    graph.getEdgeData(e).wire = nullptr;
  }

  // create pins for all gates 
  // and connect all their inputs to all their outputs
  for (auto item: vModule.gates) {
    auto gate = item.second;

    for (auto pin: gate->outPins) {
      auto n = graph.createNode();
      nodeMap.insert({pin, n});

      graph.addNode(n, unprotected);
      graph.getData(n, unprotected).pin = pin;
    }

    for (auto pin: gate->inPins) {
      auto n = graph.createNode();
      nodeMap.insert({pin, n});

      graph.addNode(n, unprotected);
      graph.getData(n, unprotected).pin = pin;

      auto inPinNode = nodeMap.at(pin);
      for (auto outPin: gate->outPins) {
        auto outPinNode = nodeMap.at(outPin);
        auto e = graph.addMultiEdge(inPinNode, outPinNode, unprotected);
        graph.getEdgeData(e).wire = nullptr;
      }
    }
  } // end for all gates

  // connect pins according to verilog wires
  for (auto item: vModule.wires) {
    auto wire = item.second;
    auto rootNode = nodeMap.at(wire->root);
    for (auto leaf: wire->leaves) {
      auto leafNode = nodeMap.at(leaf);
      auto e = graph.addMultiEdge(rootNode, leafNode, unprotected);
      graph.getEdgeData(e).wire = wire;
    }
  }
}

void doGateSizing() {

}

void printGraph() {
  auto unprotected = Galois::MethodFlag::UNPROTECTED;

  for (auto n: graph) {
    std::cout << "node: ";
    auto pin = graph.getData(n, unprotected).pin;
    if (pin) {
      if (pin->gate) {
        std::cout << pin->gate->name << ".";
      }
      std::cout << pin->name;
    }
    else {
      if (!std::distance(graph.edge_begin(n, unprotected), graph.edge_end(n, unprotected))) {
        std::cout << "dummySink";
      }
      else if (!std::distance(graph.in_edge_begin(n, unprotected), graph.in_edge_end(n, unprotected))) {
        std::cout << "dummySrc";
      }
    }
    std::cout << std::endl;

    for (auto oe: graph.edges(n)) {
      auto pin = graph.getData(graph.getEdgeDst(oe), unprotected).pin;
      std::cout << "  outgoing edge to ";
      if (pin) {
        if (pin->gate) {
          std::cout << pin->gate->name << ".";
        }
        std::cout << pin->name;
      }
      else {
        std::cout << "dummySink";
      }

      auto wire = graph.getEdgeData(oe).wire;
      if (wire) {
        std::cout << " (wire " << wire->name << ")";
      }

      std::cout << std::endl;
   }

   for (auto ie: graph.in_edges(n)) {
      auto pin = graph.getData(graph.getEdgeDst(ie), unprotected).pin;
      std::cout << "  incoming edge from ";
      if (pin) {
        if (pin->gate) {
          std::cout << pin->gate->name << ".";
        }
        std::cout << pin->name;
      }
      else {
        std::cout << "dummySrc";
      }

      auto wire = graph.getEdgeData(ie).wire;
      if (wire) {
        std::cout << " (wire " << wire->name << ")";
      }

      std::cout << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer T("TotalTime");
  T.start();

  cellLib.read(lib);
  cellLib.printCellLibDebug();
  std::cout << "cell library parsed\n" << std::endl;

  vModule.read(inputCircuit, &cellLib);
  vModule.printVerilogModuleDebug();
  std::cout << "verilog module parsed\n" << std::endl;

  sdc.read(sdcFile, &cellLib);
  sdc.printSdcDebug();
  std::cout << "sdc module parsed\n" << std::endl;

  constructCircuitGraph();
  printGraph();
  std::cout << "graph constructed\n" << std::endl;

  doGateSizing();
//  printGraph();

  T.stop();
  vModule.writeVerilogModule(outputCircuit);
  std::cout << "verilog module written\n" << std::endl;

  return 0;
}


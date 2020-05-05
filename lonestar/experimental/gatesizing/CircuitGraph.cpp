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

#include "CircuitGraph.h"

#include <iostream>

static auto unprotected = galois::MethodFlag::UNPROTECTED;

void CircuitGraph::construct(VerilogModule& vModule) {
  dummySrc = g.createNode();
  g.addNode(dummySrc, unprotected);
  g.getData(dummySrc, unprotected).pin = nullptr;

  dummySink = g.createNode();
  g.addNode(dummySink, unprotected);
  g.getData(dummySink, unprotected).pin = nullptr;

  // create nodes for all input pins
  // and connect dummySrc to them
  for (auto item : vModule.inputs) {
    auto pin = item.second;
    auto n   = g.createNode();
    nodeMap.insert({pin, n});

    g.addNode(n, unprotected);
    g.getData(n, unprotected).pin = pin;

    auto e                = g.addMultiEdge(dummySrc, n, unprotected);
    g.getEdgeData(e).wire = nullptr;
  }

  // create nodes for all output pins
  // and connect them to dummySink
  for (auto item : vModule.outputs) {
    auto pin = item.second;
    auto n   = g.createNode();
    nodeMap.insert({pin, n});

    g.addNode(n, unprotected);
    g.getData(n, unprotected).pin = pin;

    auto e                = g.addMultiEdge(n, dummySink, unprotected);
    g.getEdgeData(e).wire = nullptr;
  }

  // create pins for all gates
  // and connect all their inputs to all their outputs
  for (auto item : vModule.gates) {
    auto gate = item.second;

    for (auto pin : gate->outPins) {
      auto n = g.createNode();
      nodeMap.insert({pin, n});

      g.addNode(n, unprotected);
      g.getData(n, unprotected).pin = pin;
    }

    for (auto pin : gate->inPins) {
      auto n = g.createNode();
      nodeMap.insert({pin, n});

      g.addNode(n, unprotected);
      g.getData(n, unprotected).pin = pin;

      auto inPinNode = nodeMap.at(pin);
      for (auto outPin : gate->outPins) {
        auto outPinNode = nodeMap.at(outPin);
        auto e          = g.addMultiEdge(inPinNode, outPinNode, unprotected);
        g.getEdgeData(e).wire = nullptr;
      }
    }
  } // end for all gates

  // connect pins according to verilog wires
  for (auto item : vModule.wires) {
    auto wire     = item.second;
    auto rootNode = nodeMap.at(wire->root);
    for (auto leaf : wire->leaves) {
      auto leafNode         = nodeMap.at(leaf);
      auto e                = g.addMultiEdge(rootNode, leafNode, unprotected);
      g.getEdgeData(e).wire = wire;
    }
  }
} // end CircuitGraph::construct()

void CircuitGraph::initialize() {
  for (auto n : g) {
    auto& data              = g.getData(n, unprotected);
    data.precondition       = 0;
    data.isDummy            = false;
    data.isPrimary          = false;
    data.isOutput           = false;
    data.totalNetC          = 0.0;
    data.totalPinC          = 0.0;
    data.rise.slew          = 0.0;
    data.rise.arrivalTime   = -std::numeric_limits<float>::infinity();
    data.rise.requiredTime  = std::numeric_limits<float>::infinity();
    data.rise.slack         = std::numeric_limits<float>::infinity();
    data.rise.internalPower = 0.0;
    data.rise.netPower      = 0.0;
    data.fall.slew          = 0.0;
    data.fall.arrivalTime   = -std::numeric_limits<float>::infinity();
    data.fall.requiredTime  = std::numeric_limits<float>::infinity();
    data.fall.slack         = std::numeric_limits<float>::infinity();
    data.fall.internalPower = 0.0;
    data.fall.netPower      = 0.0;

    for (auto e : g.edges(n)) {
      auto& eData     = g.getEdgeData(e);
      eData.riseDelay = 0.0;
      eData.fallDelay = 0.0;
    }
  }

  g.getData(dummySrc, unprotected).isDummy = true;
  for (auto oe : g.edges(dummySrc)) {
    auto pi        = g.getEdgeDst(oe);
    auto& data     = g.getData(pi, unprotected);
    data.isPrimary = true;
    if (data.pin->name != "1'b0") {
      data.rise.arrivalTime = 0.0;
    }
    if (data.pin->name != "1'b1") {
      data.fall.arrivalTime = 0.0;
    }
  }

  g.getData(dummySink, unprotected).isDummy  = true;
  g.getData(dummySink, unprotected).isOutput = true;
  for (auto ie : g.in_edges(dummySink)) {
    auto po        = g.getEdgeDst(ie);
    auto& data     = g.getData(po, unprotected);
    data.isPrimary = true;
    data.isOutput  = true;
  }

  for (auto n : g) {
    auto& data = g.getData(n, unprotected);
    auto pin   = data.pin;
    if (pin) {
      auto gate = pin->gate;
      if (gate) {
        if (gate->outPins.count(pin)) {
          data.isOutput = true;

          // wires are not changing, so initialize here
          auto wire      = pin->wire;
          data.totalNetC = wire->wireLoad->wireCapacitance(wire->leaves.size());
        }
      }
    }
  }
}

static void printCircuitGraphPinName(CircuitGraph& cg, GNode n, VerilogPin* pin,
                                     std::string prompt) {
  std::cout << prompt;
  if (pin) {
    if (pin->gate) {
      std::cout << pin->gate->name << ".";
    }
    std::cout << pin->name;
  } else {
    std::cout << ((n == cg.dummySink) ? "dummySink" : "dummySrc");
  }
  std::cout << std::endl;
}

template <typename T>
static void printCircuitGraphEdge(CircuitGraph& cg, T e, std::string prompt) {
  auto& g  = cg.g;
  auto dst = g.getEdgeDst(e);
  printCircuitGraphPinName(cg, dst, g.getData(dst, unprotected).pin, prompt);

  auto& eData = g.getEdgeData(e);
  auto wire   = eData.wire;
  if (wire) {
    std::cout << "    wire: " << wire->name << std::endl;
  }
  std::cout << "    riseDelay = " << eData.riseDelay << std::endl;
  std::cout << "    fallDelay = " << eData.fallDelay << std::endl;
}

void CircuitGraph::print() {
  for (auto n : g) {
    auto& data = g.getData(n, unprotected);
    printCircuitGraphPinName(*this, n, data.pin, "node: ");

    std::cout << "  type = ";
    std::cout << ((data.isDummy) ? "dummy"
                                 : (data.isPrimary) ? "primary" : "gate");
    std::cout << ((data.isOutput) ? " output" : " input") << std::endl;

    if (data.isOutput && !data.isDummy) {
      std::cout << "  totalNetC = " << data.totalNetC << std::endl;
      std::cout << "  totalPinC = " << data.totalPinC << std::endl;
      std::cout << "  rise.internalPower = " << data.rise.internalPower
                << std::endl;
      std::cout << "  rise.netPower = " << data.rise.netPower << std::endl;
      std::cout << "  fall.internalPower = " << data.fall.internalPower
                << std::endl;
      std::cout << "  fall.netPower = " << data.fall.netPower << std::endl;
    }
    if (!data.isDummy) {
      std::cout << "  rise.slew = " << data.rise.slew << std::endl;
      std::cout << "  rise.arrivalTime = " << data.rise.arrivalTime
                << std::endl;
      std::cout << "  rise.requiredTime = " << data.rise.requiredTime
                << std::endl;
      std::cout << "  rise.slack = " << data.rise.slack << std::endl;
      std::cout << "  fall.slew = " << data.fall.slew << std::endl;
      std::cout << "  fall.arrivalTime = " << data.fall.arrivalTime
                << std::endl;
      std::cout << "  fall.requiredTime = " << data.fall.requiredTime
                << std::endl;
      std::cout << "  fall.slack = " << data.fall.slack << std::endl;
    }

    for (auto oe : g.edges(n)) {
      printCircuitGraphEdge(*this, oe, "  outgoing edge to ");
    } // end for oe

    for (auto ie : g.in_edges(n)) {
      printCircuitGraphEdge(*this, ie, "  incoming edge from ");
    } // end for ie
  }
} // end CircuitGraph::print()

std::pair<size_t, size_t> CircuitGraph::getStatistics() {
  size_t numNodes = std::distance(g.begin(), g.end());
  size_t numEdges = 0;
  for (auto n : g) {
    numEdges += std::distance(g.edge_begin(n), g.edge_end(n));
  }
  return std::make_pair(numNodes, numEdges);
}

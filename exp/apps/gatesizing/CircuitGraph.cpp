#include "CircuitGraph.h"

#include <iostream>

std::unordered_map<VerilogPin *, GNode> nodeMap;
GNode dummySrc, dummySink;

void constructCircuitGraph(Graph& g, VerilogModule& vModule) {
  auto unprotected = Galois::MethodFlag::UNPROTECTED;

  dummySrc = g.createNode();
  g.addNode(dummySrc, unprotected);
  g.getData(dummySrc, unprotected).pin = nullptr;

  dummySink = g.createNode();
  g.addNode(dummySink, unprotected);
  g.getData(dummySink, unprotected).pin = nullptr;

  // create nodes for all input pins 
  // and connect dummySrc to them
  for (auto item: vModule.inputs) {
    auto pin = item.second;
    auto n = g.createNode();
    nodeMap.insert({pin, n});

    g.addNode(n, unprotected);
    g.getData(n, unprotected).pin = pin;

    auto e = g.addMultiEdge(dummySrc, n, unprotected);
    g.getEdgeData(e).wire = nullptr;
  }

  // create nodes for all output pins 
  // and connect them to dummySink
  for (auto item: vModule.outputs) {
    auto pin = item.second;
    auto n = g.createNode();
    nodeMap.insert({pin, n});

    g.addNode(n, unprotected);
    g.getData(n, unprotected).pin = pin;

    auto e = g.addMultiEdge(n, dummySink, unprotected);
    g.getEdgeData(e).wire = nullptr;
  }

  // create pins for all gates 
  // and connect all their inputs to all their outputs
  for (auto item: vModule.gates) {
    auto gate = item.second;

    for (auto pin: gate->outPins) {
      auto n = g.createNode();
      nodeMap.insert({pin, n});

      g.addNode(n, unprotected);
      g.getData(n, unprotected).pin = pin;
    }

    for (auto pin: gate->inPins) {
      auto n = g.createNode();
      nodeMap.insert({pin, n});

      g.addNode(n, unprotected);
      g.getData(n, unprotected).pin = pin;

      auto inPinNode = nodeMap.at(pin);
      for (auto outPin: gate->outPins) {
        auto outPinNode = nodeMap.at(outPin);
        auto e = g.addMultiEdge(inPinNode, outPinNode, unprotected);
        g.getEdgeData(e).wire = nullptr;
      }
    }
  } // end for all gates

  // connect pins according to verilog wires
  for (auto item: vModule.wires) {
    auto wire = item.second;
    auto rootNode = nodeMap.at(wire->root);
    for (auto leaf: wire->leaves) {
      auto leafNode = nodeMap.at(leaf);
      auto e = g.addMultiEdge(rootNode, leafNode, unprotected);
      g.getEdgeData(e).wire = wire;
    }
  }
} // end constructCircuitGraph()

void printCircuitGraph(Graph& g) {
  auto unprotected = Galois::MethodFlag::UNPROTECTED;

  for (auto n: g) {
    std::cout << "node: ";
    auto pin = g.getData(n, unprotected).pin;
    if (pin) {
      if (pin->gate) {
        std::cout << pin->gate->name << ".";
      }
      std::cout << pin->name;
    }
    else {
      if (!std::distance(g.edge_begin(n, unprotected), g.edge_end(n, unprotected))) {
        std::cout << "dummySink";
      }
      else if (!std::distance(g.in_edge_begin(n, unprotected), g.in_edge_end(n, unprotected))) {
        std::cout << "dummySrc";
      }
    }
    std::cout << std::endl;

    for (auto oe: g.edges(n)) {
      auto pin = g.getData(g.getEdgeDst(oe), unprotected).pin;
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

      auto wire = g.getEdgeData(oe).wire;
      if (wire) {
        std::cout << " (wire " << wire->name << ")";
      }

      std::cout << std::endl;
   } // end for oe

   for (auto ie: g.in_edges(n)) {
      auto pin = g.getData(g.getEdgeDst(ie), unprotected).pin;
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

      auto wire = g.getEdgeData(ie).wire;
      if (wire) {
        std::cout << " (wire " << wire->name << ")";
      }

      std::cout << std::endl;
    }
  } // end for ie
} // end printCircuitGraph()


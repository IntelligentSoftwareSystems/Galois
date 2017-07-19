#include "CircuitGraph.h"

#include <iostream>

Graph graph;
std::unordered_map<VerilogPin *, GNode> nodeMap;
GNode dummySrc, dummySink;

static auto unprotected = Galois::MethodFlag::UNPROTECTED;

void constructCircuitGraph(Graph& g, VerilogModule& vModule) {
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

void initializeCircuitGraph(Graph& g, SDC& sdc) {
  for (auto n: g) {
    auto& data = g.getData(n, unprotected);
    data.slew = 0.0;
    data.totalNetC = 0.0;
    data.arrivalTime = 0.0;
    data.requiredTime = sdc.targetDelay;
    data.slack = sdc.targetDelay;
    data.internalPower = 0.0;
    data.netPower = 0.0;

    for (auto e: g.edges(n)) {
      auto& eData = g.getEdgeData(e);
      eData.isRise = false;
      eData.delay = 0.0;
      eData.internalPower = 0.0;
      eData.netPower = 0.0;
    }
  }

  for (auto oe: g.edges(dummySrc)) {
    auto pi = g.getEdgeDst(oe);
    g.getData(pi, unprotected).slew = sdc.primaryInputSlew;
  }

  for (auto ie: g.in_edges(dummySink)) {
    auto po = g.getEdgeDst(ie);
    g.getData(po, unprotected).totalNetC = sdc.primaryOutputTotalNetC;
  }
}

void printCircuitGraph(Graph& g) {
  for (auto n: g) {
    std::cout << "node: ";
    auto& data = g.getData(n, unprotected);
    auto pin = data.pin;
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

    std::cout << "  slew = " << data.slew << std::endl;
    std::cout << "  totalNetC = " << data.totalNetC << std::endl;
    std::cout << "  arrivalTime = " << data.arrivalTime << std::endl;
    std::cout << "  requiredTime = " << data.requiredTime << std::endl;
    std::cout << "  slack = " << data.slack << std::endl;
    std::cout << "  internalPower = " << data.internalPower << std::endl;
    std::cout << "  netPower = " << data.netPower << std::endl;

    for (auto oe: g.edges(n)) {
      auto toPin = g.getData(g.getEdgeDst(oe), unprotected).pin;
      std::cout << "  outgoing edge to ";
      if (toPin) {
        if (toPin->gate) {
          std::cout << toPin->gate->name << ".";
        }
        std::cout << toPin->name;
      }
      else {
        std::cout << "dummySink";
      }
      std::cout << std::endl;

      auto eData = g.getEdgeData(oe);
      auto wire = eData.wire;
      if (wire) {
        std::cout << "    wire: " << wire->name << std::endl;
      }
      else {
        std::cout << "    isRise = " << ((eData.isRise) ? "true" : "false") << std::endl;
        std::cout << "    delay = " << eData.delay << std::endl;
        std::cout << "    internalPower = " << eData.internalPower << std::endl;
        std::cout << "    netPower = " << eData.netPower << std::endl;
      }
   } // end for oe

   for (auto ie: g.in_edges(n)) {
      auto fromPin = g.getData(g.getEdgeDst(ie), unprotected).pin;
      std::cout << "  incoming edge from ";
      if (fromPin) {
        if (fromPin->gate) {
          std::cout << fromPin->gate->name << ".";
        }
        std::cout << fromPin->name;
      }
      else {
        std::cout << "dummySrc";
      }
      std::cout << std::endl;

      auto eData = g.getEdgeData(ie);
      auto wire = eData.wire;
      if (wire) {
        std::cout << "    wire: " << wire->name << std::endl;
      }
      else {
        std::cout << "    isRise = " << ((eData.isRise) ? "true" : "false") << std::endl;
        std::cout << "    delay = " << eData.delay << std::endl;
        std::cout << "    internalPower = " << eData.internalPower << std::endl;
        std::cout << "    netPower = " << eData.netPower << std::endl;
      }
    }
  } // end for ie
} // end printCircuitGraph()


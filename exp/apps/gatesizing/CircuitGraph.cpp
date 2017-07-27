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

  nodeMap.clear();
} // end constructCircuitGraph()

void initializeCircuitGraph(Graph& g, SDC& sdc) {
  for (auto n: g) {
    auto& data = g.getData(n, unprotected);
    data.slew = 0.0;
    data.totalNetC = 0.0;
    data.totalPinC = 0.0;
    data.arrivalTime = 0.0;
    data.requiredTime = std::numeric_limits<float>::infinity();
    data.slack = std::numeric_limits<float>::infinity();
    data.internalPower = 0.0;
    data.netPower = 0.0;
    data.isRise = false;
    data.isPrimaryInput = false;
    data.isPrimaryOutput = false;
    data.isGateInput = false;
    data.isGateOutput = false;

    for (auto e: g.edges(n)) {
      g.getEdgeData(e).delay = 0.0;
    }
  }

  for (auto oe: g.edges(dummySrc)) {
    auto pi = g.getEdgeDst(oe);
    auto& data = g.getData(pi, unprotected);
    data.isPrimaryInput = true;
    data.slew = sdc.primaryInputSlew;
  }

  for (auto ie: g.in_edges(dummySink)) {
    auto po = g.getEdgeDst(ie);
    auto& data = g.getData(po, unprotected);
    data.isPrimaryOutput = true;
    data.requiredTime = sdc.targetDelay;
    data.totalPinC = sdc.primaryOutputTotalPinC;
    data.totalNetC = sdc.primaryOutputTotalNetC;
  }

  for (auto n: g) {
    auto& data = g.getData(n, unprotected);
    auto pin = data.pin;
    if (pin) {
      auto gate = pin->gate;
      if (gate) {
        if (gate->inPins.count(pin)) {
          data.isGateInput = true;
        }
        else if (gate->outPins.count(pin)) {
          data.isGateOutput = true;

          // wires are not changing, so initialize here
          auto wire = g.getEdgeData(g.edge_begin(n)).wire;
          data.totalNetC = wire->wireLoad->wireCapacitance(wire->leaves.size());
        }
      }
    }
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
      std::cout << ((n == dummySink) ? "dummySink" : "dummySrc");
    }
    std::cout << std::endl;

    if (data.isPrimaryInput || data.isGateInput) {
      std::cout << "  type = " << ((data.isPrimaryInput) ? "primary" : "gate") << " input" << std::endl;
      std::cout << "  slew = " << data.slew << std::endl;
    }
    if (data.isGateOutput || data.isPrimaryOutput) {
      std::cout << "  type = " << ((data.isPrimaryOutput) ? "primary" : "gate") << " output" << std::endl;
      std::cout << "  totalNetC = " << data.totalNetC << std::endl;
      std::cout << "  totalPinC = " << data.totalPinC << std::endl;
      std::cout << "  internalPower = " << data.internalPower << std::endl;
      std::cout << "  netPower = " << data.netPower << std::endl;
    }
    if (n != dummySrc && n != dummySink) {
      std::cout << "  arrivalTime = " << data.arrivalTime << std::endl;
      std::cout << "  requiredTime = " << data.requiredTime << std::endl;
      std::cout << "  slack = " << data.slack << std::endl;
      std::cout << "  isRise = " << ((data.isRise) ? "true" : "false") << std::endl;
    }

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
        std::cout << "    delay = " << eData.delay << std::endl;
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
        std::cout << "    delay = " << eData.delay << std::endl;
      }
    }
  } // end for ie
} // end printCircuitGraph()


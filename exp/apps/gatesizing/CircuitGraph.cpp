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
    data.precondition = 0;
    data.isDummy = false;
    data.isPrimary = false;
    data.isOutput = false;
    data.totalNetC = 0.0;
    data.totalPinC = 0.0;
    data.rise.slew = 0.0;
    data.rise.arrivalTime = -std::numeric_limits<float>::infinity();
    data.rise.requiredTime = std::numeric_limits<float>::infinity();
    data.rise.slack = std::numeric_limits<float>::infinity();
    data.rise.internalPower = 0.0;
    data.rise.netPower = 0.0;
    data.fall.slew = 0.0;
    data.fall.arrivalTime = -std::numeric_limits<float>::infinity();
    data.fall.requiredTime = std::numeric_limits<float>::infinity();
    data.fall.slack = std::numeric_limits<float>::infinity();
    data.fall.internalPower = 0.0;
    data.fall.netPower = 0.0;

    for (auto e: g.edges(n)) {
      auto& eData = g.getEdgeData(e);
      eData.riseDelay = 0.0;
      eData.fallDelay = 0.0;
    }
  }

  g.getData(dummySrc, unprotected).isDummy = true;
  for (auto oe: g.edges(dummySrc)) {
    auto pi = g.getEdgeDst(oe);
    auto& data = g.getData(pi, unprotected);
    data.isPrimary = true;
    data.rise.slew = sdc.primaryInputRiseSlew;
    data.rise.arrivalTime = 0.0;
    data.fall.slew = sdc.primaryInputFallSlew;
    data.fall.arrivalTime = 0.0;
  }

  g.getData(dummySink, unprotected).isDummy = true;
  g.getData(dummySink, unprotected).isOutput = true;
  for (auto ie: g.in_edges(dummySink)) {
    auto po = g.getEdgeDst(ie);
    auto& data = g.getData(po, unprotected);
    data.isPrimary = true;
    data.isOutput = true;
    data.totalPinC = sdc.primaryOutputTotalPinC;
    data.totalNetC = sdc.primaryOutputTotalNetC;
    data.rise.requiredTime = sdc.targetDelay;
    data.fall.requiredTime = sdc.targetDelay;
  }

  for (auto n: g) {
    auto& data = g.getData(n, unprotected);
    auto pin = data.pin;
    if (pin) {
      auto gate = pin->gate;
      if (gate) {
        if (gate->outPins.count(pin)) {
          data.isOutput = true;

          // wires are not changing, so initialize here
          auto wire = g.getEdgeData(g.edge_begin(n)).wire;
          data.totalNetC = wire->wireLoad->wireCapacitance(wire->leaves.size());
        }
      }
    }
  }
}

static void printCircuitGraphPinName(GNode n, VerilogPin *pin, std::string prompt) {
  std::cout << prompt;
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
}

template<typename T>
static void printCircuitGraphEdge(Graph& g, T e, std::string prompt) {
  auto dst = g.getEdgeDst(e);
  printCircuitGraphPinName(dst, g.getData(dst, unprotected).pin, prompt);

  auto& eData = g.getEdgeData(e);
  auto wire = eData.wire;
  if (wire) {
    std::cout << "    wire: " << wire->name << std::endl;
  }
  else {
    std::cout << "    riseDelay = " << eData.riseDelay << std::endl;
    std::cout << "    fallDelay = " << eData.fallDelay << std::endl;
  }
}

void printCircuitGraph(Graph& g) {
  for (auto n: g) {
    auto& data = g.getData(n, unprotected);
    printCircuitGraphPinName(n, data.pin, "node: ");

    std::cout << "  type = ";
    std::cout << ((data.isDummy) ? "dummy" : 
                  (data.isPrimary) ? "primary" : "gate");
    std::cout << ((data.isOutput) ? " output" : " input") << std::endl;

    if (data.isOutput && !data.isDummy) {
      std::cout << "  totalNetC = " << data.totalNetC << std::endl;
      std::cout << "  totalPinC = " << data.totalPinC << std::endl;
      std::cout << "  rise.internalPower = " << data.rise.internalPower << std::endl;
      std::cout << "  rise.netPower = " << data.rise.netPower << std::endl;
      std::cout << "  fall.internalPower = " << data.fall.internalPower << std::endl;
      std::cout << "  fall.netPower = " << data.fall.netPower << std::endl;
    }
    if (!data.isDummy) {
      std::cout << "  rise.slew = " << data.rise.slew << std::endl;
      std::cout << "  rise.arrivalTime = " << data.rise.arrivalTime << std::endl;
      std::cout << "  rise.requiredTime = " << data.rise.requiredTime << std::endl;
      std::cout << "  rise.slack = " << data.rise.slack << std::endl;
      std::cout << "  fall.slew = " << data.fall.slew << std::endl;
      std::cout << "  fall.arrivalTime = " << data.fall.arrivalTime << std::endl;
      std::cout << "  fall.requiredTime = " << data.fall.requiredTime << std::endl;
      std::cout << "  fall.slack = " << data.fall.slack << std::endl;
    }

    for (auto oe: g.edges(n)) {
      printCircuitGraphEdge(g, oe, "  outgoing edge to ");
    } // end for oe

    for (auto ie: g.in_edges(n)) {
      printCircuitGraphEdge(g, ie, "  incoming edge from ");
    } // end for ie
  }
} // end printCircuitGraph()

std::pair<size_t, size_t> getCircuitGraphStatistics(Graph& g) {
  size_t numNodes = std::distance(g.begin(), g.end());
  size_t numEdges = 0;
  for (auto n: g) {
    numEdges += std::distance(g.edge_begin(n), g.edge_end(n));
  }
  return std::make_pair(numNodes, numEdges);
}

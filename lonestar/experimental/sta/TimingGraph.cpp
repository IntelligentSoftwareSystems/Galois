#include "TimingGraph.h"

#include <unordered_set>
#include <string>
#include <iostream>
#include <deque>

static auto unprotected = galois::MethodFlag::UNPROTECTED;
static std::string name0 = "1\'b0";
static std::string name1 = "1\'b1";

void TimingGraph::addPin(VerilogPin* pin) {
  for (size_t j = 0; j < 2; j++) {
    auto n = g.createNode();
    g.addNode(n);
    nodeMap[pin][j] = n;

    auto& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
    data.pin = pin;
    data.isPrimary = true;
    data.isRise = (bool)j;

    data.t.insert(data.t.begin(), libs.size(), NodeTiming());
    for (size_t k = 0; k < libs.size(); k++) {
      data.t[k].pin = nullptr;
    }
  }
}

void TimingGraph::addGate(VerilogGate* gate) {
  std::unordered_set<GNode> outNodes;
  std::unordered_set<GNode> inNodes;

  // allocate fall/rise pins in gate
  for (auto& gp: gate->pins) {
    auto p = gp.second;
    for (size_t j = 0; j < 2; j++) {
      auto n = g.createNode();
      g.addNode(n, unprotected);
      nodeMap[p][j] = n;

      auto& data = g.getData(n, unprotected);
      data.pin = p;
      data.isRise = (bool)j;

      data.t.insert(data.t.begin(), libs.size(), NodeTiming());
      for (size_t k = 0; k < libs.size(); k++) {
        data.t[k].pin = libs[k]->findCell(p->gate->cellType)->findCellPin(p->name);
      }

      if (data.t[0].pin->isOutput) {
        data.isOutput = true;
        outNodes.insert(n);
      }
      if (data.t[0].pin->isInput) {
        inNodes.insert(n);
      }
    }
  }

  // add timing arcs among gate pins
  for (auto on: outNodes) {
    auto& oData = g.getData(on, unprotected);
    auto op = oData.t[0].pin;
    for (auto in: inNodes) {
      auto& iData = g.getData(in, unprotected);
      auto ip = iData.t[0].pin;
      auto isNegUnate = (oData.isRise != iData.isRise);
      if (op->isUnateAtEdge(ip, isNegUnate, oData.isRise)) {
        auto e = g.addMultiEdge(in, on, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.t.insert(eData.t.begin(), libs.size(), EdgeTiming());
      }
    }
  }
}

void TimingGraph::addWire(VerilogWire* w) {
  // dangling wires due to assign statements
  if (0 == w->outDeg()) {
    return;
  }

  // scan for the source of w
  GNode src[2];
  for (auto p: w->pins) {
    for (size_t j = 0; j < 2; j++) {
      auto n = nodeMap[p][j];
      auto& data = g.getData(n, unprotected);
      // w goes from 1'b0 or 1'b1
      if (p->name == name0 || p->name == name1) {
        src[j] = n;
      }
      // w goes from primary input
      else if (data.isPrimary && !data.isOutput) {
        src[j] = n;
      }
      // w goes from gate output 
      else if (!data.isPrimary && data.isOutput) {
        src[j] = n;
      }
    }
  }

  // connect w.src to other pins
  for (size_t j = 0; j < 2; j++) {
    auto& srcData = g.getData(src[j], unprotected);
    // 1'b0 rise is never connected
    if (srcData.pin->name == name0 && 1 == j) {
      continue;
    }
    // 1'b1 fall is never connected
    if (srcData.pin->name == name1 && 0 == j) {
      continue;
    }
    for (auto p: w->pins) {
      auto n = nodeMap[p][j];
      if (n != src[j]) {
        auto e = g.addMultiEdge(src[j], n, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = p->wire;
        eData.t.insert(eData.t.begin(), libs.size(), EdgeTiming());
        for (size_t k = 0; k < libs.size(); k++) {
          eData.t[k].wireLoad = libs[k]->defaultWireLoad;
        }
      }
    }
  }
}

void TimingGraph::addDummySrc() {
  dummySrc = g.createNode();
  g.addNode(dummySrc, unprotected);
  auto& data = g.getData(dummySrc, unprotected);
  data.isDummy = true;
  data.pin = nullptr;

  for (auto i: m.inPins) {
    for (size_t j = 0; j < 2; j++) {
      auto e = g.addMultiEdge(dummySrc, nodeMap[i][j], unprotected);
      auto& eData = g.getEdgeData(e);
      eData.wire = nullptr;
    }
  }
}

void TimingGraph::addDummySink() {
  dummySink = g.createNode();
  g.addNode(dummySink, unprotected);
  auto& data = g.getData(dummySink, unprotected);
  data.isDummy = true;
  data.isOutput = true;
  data.pin = nullptr;

  for (auto i: m.outPins) {
    for (size_t j = 0; j < 2; j++) {
      auto on = nodeMap[i][j];
      g.getData(on, unprotected).isOutput = true;
      auto e = g.addMultiEdge(on, dummySink, unprotected);
      auto& eData = g.getEdgeData(e);
      eData.wire = nullptr;
    }
  }
}

void TimingGraph::construct() {
  for (auto& i: m.pins) {
    addPin(i.second);
  }

  addDummySrc();
  addDummySink(); // also marks primary output nodes

  for (auto& i: m.gates) {
    addGate(i.second);
  }

  // add wires among pins
  for (auto& i: m.wires) {
    addWire(i.second);
  }
}

void TimingGraph::initialize() {

}

std::string TimingGraph::getNodeName(GNode n) {
  auto& data = g.getData(n, unprotected);

  std::string nName;
  if (data.isDummy) {
    if (data.isOutput) {
      nName = "Dummy output";
    }
    else {
      nName = "Dummy input";
    }
  }
  else if (data.isPrimary) {
    if (data.isOutput) {
      nName = "Primary output ";
    }
    else {
      nName = "Primary input ";
    }
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "rise" : "fall";
  }
  else {
    if (data.isOutput) {
      nName = "Gate output ";
    }
    else {
      nName = "Gate input ";
    }
    nName += data.pin->gate->name;
    nName += ".";
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "rise" : "fall";
  }

  return nName;
}

void TimingGraph::print(std::ostream& os) {
  os << "Timing graph for module " << m.name << std::endl;

  for (auto n: g) {
    os << "  " << getNodeName(n) << std::endl;

    for (auto e: g.edges(n)) {
      auto w = g.getEdgeData(e).wire;
      if (w) {
        os << "    Wire " << w->name;
      }
      else {
        os << "    Timing arc";
      }
      os << " to " << getNodeName(g.getEdgeDst(e)) << std::endl;
    }

    for (auto ie: g.in_edges(n)) {
      auto w = g.getEdgeData(ie).wire;
      if (w) {
        os << "    Wire " << w->name;
      }
      else {
        os << "    Timing arc";
      }
      os << " from " << getNodeName(g.getEdgeDst(ie)) << std::endl;
    } 
  } // end for n
}

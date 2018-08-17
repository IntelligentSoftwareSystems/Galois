#include "TimingGraph.h"
#include "galois/Bag.h"

#include <unordered_set>
#include <string>
#include <iostream>

static auto unprotected = galois::MethodFlag::UNPROTECTED;
static std::string name0 = "1\'b0";
static std::string name1 = "1\'b1";
static float infinity = std::numeric_limits<float>::infinity();

void TimingGraph::addPin(VerilogPin* pin) {
  for (size_t j = 0; j < 2; j++) {
    auto n = g.createNode();
    g.addNode(n);
    nodeMap[pin][j] = n;

    auto& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
    data.pin = pin;
    data.isRise = (bool)j;
    if (pin->name == name0 || pin->name == name1) {
      data.isPowerNode = true;
    }
    else {
      data.isPrimary = true;
      if (m.isOutPin(pin)) {
        data.isOutput = true;
      }
    }

    data.t.insert(data.t.begin(), libs.size(), NodeTiming());
    for (size_t k = 0; k < libs.size(); k++) {
      data.t[k].pin = nullptr;
      data.t[k].arrival = infinity;
      data.t[k].required = infinity;
      data.t[k].slack = infinity;
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
        data.t[k].arrival = infinity;
        data.t[k].required = infinity;
        data.t[k].slack = infinity;
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
        for (size_t k = 0; k < libs.size(); k++) {
          eData.t[k].wireLoad = nullptr;
        }
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
      if (data.isPowerNode) {
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

void TimingGraph::addDummyNodes() {
  galois::InsertBag<GNode> noPredNodes;
  galois::InsertBag<GNode> noSuccNodes;
  galois::do_all(
      galois::iterate(g),
      [&] (GNode n) {
        auto numPred = std::distance(g.in_edge_begin(n, unprotected), g.in_edge_end(n, unprotected));
        auto numSucc = std::distance(g.edge_begin(n, unprotected), g.edge_end(n, unprotected));
        if (0 == numPred) { noPredNodes.push_back(n); }
        if (0 == numSucc) { noSuccNodes.push_back(n); }
      }
      , galois::loopname("ScanForNeighborsOfDummyNodes")
      , galois::steal()
  );

  dummySrc = g.createNode();
  g.addNode(dummySrc, unprotected);
  auto& dSrcData = g.getData(dummySrc, unprotected);
  dSrcData.isDummy = true;
  dSrcData.pin = nullptr;
  for (auto n: noPredNodes) {
    auto e = g.addMultiEdge(dummySrc, n, unprotected);
    auto& eData = g.getEdgeData(e);
    eData.wire = nullptr;
  }

  dummySink = g.createNode();
  g.addNode(dummySink, unprotected);
  auto& dSinkData = g.getData(dummySink, unprotected);
  dSinkData.isDummy = true;
  dSinkData.isOutput = true;
  dSinkData.pin = nullptr;
  for (auto n: noSuccNodes) {
    auto e = g.addMultiEdge(n, dummySink, unprotected);
    auto& eData = g.getEdgeData(e);
    eData.wire = nullptr;
  }
}

void TimingGraph::construct() {
  for (auto& i: m.pins) {
    addPin(i.second);
  }

  for (auto& i: m.gates) {
    addGate(i.second);
  }

  // add wires among pins
  for (auto& i: m.wires) {
    addWire(i.second);
  }

  addDummyNodes();
}

void TimingGraph::computeTopoL() {
  galois::for_each(
      galois::iterate({dummySrc}),
      [&] (GNode n, auto& ctx) {
        // lock outgoing neighbors for cautiousness
        g.edges(n);

        auto& myTopoL = g.getData(n).topoL;
        for (auto ie: g.in_edges(n, unprotected)) {
          auto prev = g.getEdgeDst(ie);
          auto prevTopoL = g.getData(prev).topoL;
          if (myTopoL <= prevTopoL) { myTopoL = prevTopoL + 1; }
        }

        for (auto e: g.edges(n)) {
          auto succ = g.getEdgeDst(e);
          auto& succTopoL = g.getData(succ).topoL;
          succTopoL -= 1;
          if (0 == succTopoL) { ctx.push(succ); }
        }
      }
      , galois::loopname("TimingGraphTopoLevel")
  );

#if 0
  std::for_each(
      g.begin(), g.end(),
      [&] (GNode n) {
        auto myTopoL = g.getData(n, unprotected).topoL;
        for (auto e: g.edges(n, unprotected)) {
          auto succ = g.getEdgeDst(e);
          auto succTopoL = g.getData(succ, unprotected).topoL;
          if (myTopoL >= succTopoL) {
            std::cout << "Topo error: (" << getNodeName(n) << ").topoL = " << myTopoL;
            std::cout << ", (" << getNodeName(succ) << ").topoL = " << succTopoL << std::endl;
          }
        }
      }
  );
#endif
}

void TimingGraph::computeRevTopoL() {
  galois::for_each(
      galois::iterate({dummySink}),
      [&] (GNode n, auto& ctx) {
        // lock incoming neighbors for cautiousness
        g.in_edges(n);

        auto& myRevTopoL = g.getData(n).revTopoL;
        for (auto e: g.edges(n, unprotected)) {
          auto succ = g.getEdgeDst(e);
          auto succRevTopoL = g.getData(succ).revTopoL;
          if (myRevTopoL <= succRevTopoL) { myRevTopoL = succRevTopoL + 1; }
        }

        for (auto ie: g.in_edges(n)) {
          auto prev = g.getEdgeDst(ie);
          auto& prevRevTopoL = g.getData(prev).revTopoL;
          prevRevTopoL--;
          if (0 == prevRevTopoL) { ctx.push(prev); }
        }
      }
      , galois::loopname("TimingGraphRevTopoLevel")
  );

#if 0
  std::for_each(
      g.begin(), g.end(),
      [&] (GNode n) {
        auto myRevTopoL = g.getData(n, unprotected).revTopoL;
        for (auto ie: g.in_edges(n, unprotected)) {
          auto prev = g.getEdgeDst(ie);
          auto prevRevTopoL = g.getData(prev, unprotected).revTopoL;
          if (myRevTopoL >= prevRevTopoL) {
            std::cout << "revTopo error: (" << getNodeName(n) << ").revTopoL = " << myRevTopoL;
            std::cout << ", (" << getNodeName(prev) << ").revTopoL = " << prevRevTopoL << std::endl;
          }
        }
      }
  );
#endif
}

void TimingGraph::initialize() {
  galois::do_all(
      galois::iterate(g),
      [&] (GNode n) {
        auto& data = g.getData(n, unprotected);

        // for computing levels
        data.topoL = std::distance(g.in_edge_begin(n, unprotected), g.in_edge_end(n, unprotected));
        data.revTopoL = std::distance(g.edge_begin(n, unprotected), g.edge_end(n, unprotected));

        // default arrival and required time for each timing corner
        if (data.isDummy) {
          return;
        }
        else if ((data.isPrimary && !data.isOutput) || data.isPowerNode) {
          for (size_t k = 0; k < libs.size(); k++) {
            data.t[k].arrival = 0.0;
            if (TIMING_MODE_MIN_DELAY == modes[k]) {
              data.t[k].required *= -1.0;
            }
          }
        }
        else {
          for (size_t k = 0; k < libs.size(); k++) {
            if (TIMING_MODE_MAX_DELAY == modes[k]) {
              data.t[k].arrival *= -1.0;
            }
            else if (TIMING_MODE_MIN_DELAY == modes[k]) {
              data.t[k].required *= -1.0;
            }
          }
        }
      }
      , galois::loopname("TimingGraphInitialize")
      , galois::steal()
  );

//  print();
  computeTopoL();
  computeRevTopoL();
}

void TimingGraph::setConstraints() {
  // use sdc here
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
  else if (data.isPowerNode) {
    nName = "Power ";
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "rise" : "fall";
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
    auto& data = g.getData(n, unprotected);
    os << "    topoL = " << data.topoL << ", revTopoL = " << data.revTopoL << std::endl;

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

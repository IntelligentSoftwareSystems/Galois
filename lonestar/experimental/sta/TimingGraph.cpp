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
      // rising gnd and falling vdd are dummy nodes to simplify computation
      if ((pin->name == name0 && 1 == j) || (pin->name == name1 && 0 == j)) {
        data.isDummy = true;
      }
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
        for (size_t k = 0; k < libs.size(); k++) {
          eData.t[k].wireLoad = nullptr;
          eData.t[k].delay = 0.0;
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
    for (auto p: w->pins) {
      auto n = nodeMap[p][j];
      if (n != src[j]) {
        auto e = g.addMultiEdge(src[j], n, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = p->wire;
        eData.t.insert(eData.t.begin(), libs.size(), EdgeTiming());
        for (size_t k = 0; k < libs.size(); k++) {
          eData.t[k].wireLoad = libs[k]->defaultWireLoad;
          eData.t[k].delay = 0.0;
        }
      }
    }
  }
}

void TimingGraph::addDummyNodes() {
  dummySrc = g.createNode();
  g.addNode(dummySrc, unprotected);
  auto& dSrcData = g.getData(dummySrc, unprotected);
  dSrcData.isDummy = true;
  dSrcData.pin = nullptr;
  dSrcData.t.insert(dSrcData.t.begin(), libs.size(), NodeTiming());
  for (size_t k = 0; k < libs.size(); k++) {
    dSrcData.t[k].pin = nullptr;
  }

  dummySink = g.createNode();
  g.addNode(dummySink, unprotected);
  auto& dSinkData = g.getData(dummySink, unprotected);
  dSinkData.isDummy = true;
  dSinkData.isOutput = true;
  dSinkData.pin = nullptr;
  dSinkData.t.insert(dSinkData.t.begin(), libs.size(), NodeTiming());
  for (size_t k = 0; k < libs.size(); k++) {
    dSinkData.t[k].pin = nullptr;
  }

  for (auto i: m.pins) {
    auto p = i.second;
    for (size_t j = 0; j < 2; j++) {
      auto n = nodeMap[p][j];
      auto numPred = std::distance(g.in_edge_begin(n, unprotected), g.in_edge_end(n, unprotected));
      if (!numPred) {
        auto e = g.addMultiEdge(dummySrc, n, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = nullptr;
        eData.t.insert(eData.t.begin(), libs.size(), EdgeTiming());
        for (size_t k = 0; k < libs.size(); k++) {
          eData.t[k].wireLoad = nullptr;
          eData.t[k].delay = 0.0;
        }
      }
      auto numSucc = std::distance(g.edge_begin(n, unprotected), g.edge_end(n, unprotected));
      if (!numSucc) {
        auto e = g.addMultiEdge(n, dummySink, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = nullptr;
        eData.t.insert(eData.t.begin(), libs.size(), EdgeTiming());
        for (size_t k = 0; k < libs.size(); k++) {
          eData.t[k].wireLoad = nullptr;
          eData.t[k].delay = 0.0;
        }
      }
    }
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

void TimingGraph::initFlag(bool value) {
  galois::do_all(
      galois::iterate(g),
      [&] (GNode n) {
        g.getData(n).flag.store(value);
      }
      , galois::loopname("TimingGraphInitFlag")
      , galois::steal()
  );
}

void TimingGraph::computeTopoL() {
  // data.flag indicates "to be done" if true in this function
  initFlag(true);

  galois::for_each(
      galois::iterate({dummySrc}),
      [&] (GNode n, auto& ctx) {
        auto& data = g.getData(n);
        if (!data.flag.load()) {
          return; // this node is already done
        }

        size_t myTopoL = 0;
        for (auto ie: g.in_edges(n)) {
          auto pred = g.getEdgeDst(ie);
          auto& predData = g.getData(pred);
          if (predData.flag.load()) {
            return; // this predecessor is to be done
          }
          else if (myTopoL <= predData.topoL) {
            myTopoL = predData.topoL + 1;
          }
        }

        data.topoL = myTopoL;
        data.flag.store(false); // done computing topoL
        for (auto e: g.edges(n)) {
          ctx.push(g.getEdgeDst(e));
        }
      }
      , galois::loopname("TimingGraphComputeTopoL")
      , galois::no_conflicts()
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
  // data.flag indicates "to be done" if true in this function
  initFlag(true);

  galois::for_each(
      galois::iterate({dummySink}),
      [&] (GNode n, auto& ctx) {
        auto& data = g.getData(n);
        if (!data.flag.load()) {
          return; // this node is already done
        }

        size_t myRevTopoL = 0;
        for (auto e: g.edges(n)) {
          auto succ = g.getEdgeDst(e);
          auto& succData = g.getData(succ);
          if (succData.flag.load()) {
            return; // this successor is to be done
          }
          else if (myRevTopoL <= succData.revTopoL) {
            myRevTopoL = succData.revTopoL + 1;
          }
        }

        data.revTopoL = myRevTopoL;
        data.flag.store(false); // done computing revTopoL
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
      , galois::loopname("TimingGraphComputeRevTopoL")
      , galois::no_conflicts()
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

        for (size_t k = 0; k < libs.size(); k++) {
          if (TIMING_MODE_MAX_DELAY == modes[k]) {
            data.t[k].slew = 0.0;
            data.t[k].arrival = (data.isTimingSource()) ? 0.0 : -infinity;
            data.t[k].required = infinity;
          }
          else {
            data.t[k].slew = (data.isTimingSource()) ? 0.0 : libs[k]->defaultMaxSlew;
            data.t[k].arrival = (data.isTimingSource()) ? 0.0 : infinity;
            data.t[k].required = -infinity;
          }

          data.t[k].slack = infinity;
          data.t[k].driveC = (data.isGateInput()) ? data.t[k].pin->c[data.isRise] : 0.0;
        }
      }
      , galois::loopname("TimingGraphInitialize")
      , galois::steal()
  );

//  print();
  computeTopoL();
  computeRevTopoL();
//  std::cout << "Levelization done.\n" << std::endl;
//  print();
}

void TimingGraph::setConstraints() {
  // use sdc here
}

void TimingGraph::computeDriveC(GNode n) {
  auto& data = g.getData(n);

  for (auto e: g.edges(n)) {
    auto& eData = g.getEdgeData(e);
    auto wOutDeg = eData.wire->outDeg();

    auto succ = g.getEdgeDst(e);
    auto& succData = g.getData(succ);

    for (size_t k = 0; k < libs.size(); k++) {
      if (e == g.edge_begin(n)) {
        data.t[k].driveC = eData.t[k].wireLoad->wireC(wOutDeg);
      }
      data.t[k].driveC += succData.t[k].driveC;
    }
  }
}

void TimingGraph::computeArrivalByWire(GNode n, Graph::in_edge_iterator ie) {
  auto& data = g.getData(n);

  auto pred = g.getEdgeDst(ie);
  auto& predData = g.getData(pred);

  auto& ieData = g.getEdgeData(ie);
  auto wOutDeg = ieData.wire->outDeg();

  for (size_t k = 0; k < libs.size(); k++) {
    auto delay = ieData.t[k].wireLoad->wireDelay(data.t[k].driveC, wOutDeg);
    ieData.t[k].delay = delay;
    data.t[k].arrival = predData.t[k].arrival + delay;
    data.t[k].slew = predData.t[k].slew;
  }
}

void TimingGraph::computeExtremeSlew(GNode n, Graph::in_edge_iterator ie, size_t k) {
  auto& data = g.getData(n);
  auto outPin = data.t[k].pin;

  auto pred = g.getEdgeDst(ie);
  auto& predData = g.getData(pred);
  auto inPin = predData.t[k].pin;

  bool isNeg = (data.isRise != predData.isRise);
  std::vector<float> param = {predData.t[k].slew, data.t[k].driveC};

  if (TIMING_MODE_MAX_DELAY == modes[k]) {
    auto slew = outPin->extractMax(param, TABLE_SLEW, inPin, isNeg, data.isRise).first;
    if (data.t[k].slew < slew) {
      data.t[k].slew = slew;
    }
  }
  else {
    auto slew = outPin->extractMin(param, TABLE_SLEW, inPin, isNeg, data.isRise).first;
    if (data.t[k].slew > slew) {
      data.t[k].slew = slew;
    }
  }
}

void TimingGraph::computeArrivalByTimingArc(GNode n, Graph::in_edge_iterator ie, size_t k) {
  auto& data = g.getData(n);
  auto outPin = data.t[k].pin;

  auto pred = g.getEdgeDst(ie);
  auto& predData = g.getData(pred);
  auto inPin = predData.t[k].pin;

  bool isNeg = (data.isRise != predData.isRise);
  std::vector<float> param = {predData.t[k].slew, data.t[k].driveC};

  auto& ieData = g.getEdgeData(ie);

  if (TIMING_MODE_MAX_DELAY == modes[k]) {
    auto delayResult = outPin->extractMax(param, TABLE_DELAY, inPin, isNeg, data.isRise);
    auto delay = delayResult.first;
    auto& when = delayResult.second;
    ieData.t[k].delay = delay;
    if (data.t[k].arrival < predData.t[k].arrival + delay) {
      data.t[k].arrival = predData.t[k].arrival + delay;
      if (isExactSlew) {
        data.t[k].slew = outPin->extract(param, TABLE_SLEW, inPin, isNeg, data.isRise, when);
      }
    }
  }
  else {
    auto delayResult = outPin->extractMin(param, TABLE_DELAY, inPin, isNeg, data.isRise);
    auto delay = delayResult.first;
    auto& when = delayResult.second;
    ieData.t[k].delay = delay;
    if (data.t[k].arrival > predData.t[k].arrival + delay) {
      data.t[k].arrival = predData.t[k].arrival + delay;
      if (isExactSlew) {
        data.t[k].slew = outPin->extract(param, TABLE_SLEW, inPin, isNeg, data.isRise, when);
      }
    }
  }
}

void TimingGraph::computeArrivalTime() {
  auto topoLIndexer = [&] (GNode n) {
    return g.getData(n, unprotected).topoL;
  };

  using LIFO = galois::worklists::PerThreadChunkLIFO<>;
  using OBIM
      = galois::worklists::OrderedByIntegerMetric<decltype(topoLIndexer), LIFO>
        ::template with_barrier<true>::type;

  galois::for_each(
    galois::iterate({dummySrc}),
    [&] (GNode n, auto& ctx) {
      auto& data = g.getData(n);

      if (data.isGateInput() || data.isPseudoPrimaryOutput()) {
        // should have one incoming neighbor only
        for (auto ie: g.in_edges(n)) {
          computeArrivalByWire(n, ie);
        }
      }
      else if (data.isPseudoPrimaryInput()) {
        computeDriveC(n);
        // compute slew if driving cell exists
      }
      else if (data.isGateOutput()) {
        computeDriveC(n);

        // compute arrival time & slew
        for (auto ie: g.in_edges(n)) {
          for (size_t k = 0; k < libs.size(); k++) {
            computeArrivalByTimingArc(n, ie, k);
            if (!isExactSlew) {
              computeExtremeSlew(n, ie, k);
            }
          }
        }
      }

      data.flag.store(false);

      // schedule outgoing neighbors
      for (auto e: g.edges(n)) {
        auto succ = g.getEdgeDst(e);
        auto& succData = g.getData(succ);
        if (!succData.isDummy) {
          auto& succInQueue = succData.flag;
          bool succQueued = false;
          if (succInQueue.compare_exchange_weak(succQueued, true)) {
            ctx.push(succ);
          }
        }
      }
    }
    , galois::loopname("TimingGraphComputeArrivalTime")
    , galois::no_conflicts()
    , galois::wl<OBIM>(topoLIndexer)
  );

  print();
}

std::string TimingGraph::getNodeName(GNode n) {
  auto& data = g.getData(n, unprotected);

  std::string nName;
  if (data.isPowerNode) {
    nName = "Power ";
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "rise" : "fall";
  }
  else if (data.isDummy) {
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
    auto& data = g.getData(n, unprotected);
    os << "    topoL = " << data.topoL << ", revTopoL = " << data.revTopoL << std::endl;
    for (size_t k = 0; k < libs.size(); k++) {
      os << "    corner " << k;
      os << ": arrival = " << data.t[k].arrival;
      os << ", slew = " << data.t[k].slew;
      os << ", driveC = " << data.t[k].driveC;
      os << std::endl;
    }

    for (auto e: g.edges(n)) {
      auto w = g.getEdgeData(e).wire;
      if (w) {
        os << "    Wire " << w->name;
      }
      else {
        os << "    Timing arc";
      }
      os << " to " << getNodeName(g.getEdgeDst(e)) << std::endl;

      for (size_t k = 0; k < libs.size(); k++) {
        os << "    corner " << k;
        os << ": delay = " << g.getEdgeData(e).t[k].delay;
        os << std::endl;
      }
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

      for (size_t k = 0; k < libs.size(); k++) {
        os << "    corner " << k;
        os << ": delay = " << g.getEdgeData(ie).t[k].delay;
        os << std::endl;
      }
    } 
  } // end for n
}

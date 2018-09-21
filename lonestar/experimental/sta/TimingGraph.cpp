#include "TimingGraph.h"
#include "galois/Bag.h"
#include "galois/Reduction.h"

#include <unordered_set>
#include <string>
#include <iostream>
#include <map>

static auto unprotected = galois::MethodFlag::UNPROTECTED;
static std::string name0 = "1\'b0";
static std::string name1 = "1\'b1";
static MyFloat infinity = std::numeric_limits<MyFloat>::infinity();

void TimingGraph::addPin(VerilogPin* pin) {
  for (size_t j = 0; j < 2; j++) {
    auto n = g.createNode();
    g.addNode(n);
    nodeMap[pin][j] = n;

    auto& data = g.getData(n, unprotected);
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
      auto isNeg = (oData.isRise != iData.isRise);
      if (op->isEdgeDefined(ip, isNeg, oData.isRise)) {
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
      [&] (GNode n) { g.getData(n).flag.store(value); }
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
  std::map<size_t, size_t> numInEachTopoL;
  std::for_each(
      g.begin(), g.end(),
      [&] (GNode n) {
        auto myTopoL = g.getData(n, unprotected).topoL;
        numInEachTopoL[myTopoL] += 1;
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

  for (auto& i: numInEachTopoL) {
    std::cout << "topoL " << i.first << ": " << i.second << " nodes" << std::endl;
  }
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

        // remove driving cells. to be re-established by sdc
        if (data.isDrivingCell) {
          g.removeNode(n);
          return;
        }

        // for timing computation
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
          data.t[k].pinC = (data.isGateInput()) ? data.t[k].pin->c[data.isRise] : 0.0;
          data.t[k].wireC = 0.0;

          // get rid of any driving cell info
          if (data.isPrimaryInput()) {
            data.t[k].pin = nullptr;
          }
        }
      }
      , galois::loopname("TimingGraphInitialize")
      , galois::steal()
  );

  computeTopoL();
  computeRevTopoL();
//  std::cout << "Levelization done.\n" << std::endl;
//  print();
}

void TimingGraph::setConstraints(SDC& sdc) {
  // set max required time
  for (auto& p: m.outPins) {
    for (size_t j = 0; j < 2; j++) {
      auto n = nodeMap[p][j];
      auto& data = g.getData(n, unprotected);
      for (size_t k = 0; k < libs.size(); k++) {
        if (TIMING_MODE_MAX_DELAY == modes[k]) {
          data.t[k].required = sdc.maxDelayPI2PO;
        }
      }
    }
  }

  // set output loads
  for (auto& i: sdc.pinLoads) {
    auto p = i.first;
    auto load = i.second;
    for (size_t j = 0; j < 2; j++) {
      auto n = nodeMap[p][j];
      auto& data = g.getData(n, unprotected);
      for (size_t k = 0; k < libs.size(); k++) {
        data.t[k].pinC = load;
      }
    }
  }

  // set driving cells
  for (auto& i: sdc.mapPin2DrivingCells) {
    auto p = i.first;
    auto dCell = i.second;
    GNode dN[2];

    // set primary input as the output pin of the driving cell
    for (size_t j = 0; j < 2; j++) {
      auto n = nodeMap[p][j];
      auto& data = g.getData(n, unprotected);
      for (size_t k = 0; k < libs.size(); k++) {
        data.t[k].pin = dCell->toCellPin;
        // treat this pin as internal pins
        if (TIMING_MODE_MAX_DELAY == modes[k]) {
          data.t[k].arrival = -infinity;
        }
        else {
          data.t[k].slew = libs[k]->defaultMaxSlew;
          data.t[k].arrival = infinity;
        }
      }
    }

    // allocate the input pin of the driving cell
    for (size_t j = 0; j < 2; j++) {
      auto n = g.createNode();
      g.addNode(n);
      dN[j] = n;

      auto& data = g.getData(n, unprotected);
      data.pin = nullptr;
      data.isRise = (bool)j;
      data.isDrivingCell = true;
      data.t.insert(data.t.begin(), libs.size(), NodeTiming());
      for (size_t k = 0; k < libs.size(); k++) {
        data.t[k].pin = dCell->fromCellPin;
        data.t[k].slew = dCell->slew[j];
        data.t[k].arrival = 0.0;
        data.t[k].slack = infinity;
        data.t[k].pinC = 0.0;
        data.t[k].wireC = 0.0;

        if (TIMING_MODE_MAX_DELAY == modes[k]) {
          data.t[k].required = infinity;
        }
        else {
          data.t[k].required = -infinity;
        }
      }
    }

    // connect the primiray input to the driving pin
    for (size_t j = 0; j < 2; j++) {
      auto on = nodeMap[p][j];
      auto& oData = g.getData(on, unprotected);

      for (size_t d = 0; d < 2; d++) {
        auto in = dN[d];
        auto& iData = g.getData(in, unprotected);
        iData.topoL = 0;
        iData.revTopoL = oData.revTopoL + 1;
        bool isNeg = (oData.isRise != iData.isRise);

        if (dCell->toCellPin->isEdgeDefined(dCell->fromCellPin, isNeg, oData.isRise)) {
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

//  print();
}

void TimingGraph::computeDriveC(GNode n) {
  auto& data = g.getData(n);

  for (auto e: g.edges(n)) {
    auto& eData = g.getEdgeData(e);

    auto succ = g.getEdgeDst(e);
    auto& succData = g.getData(succ);

    for (size_t k = 0; k < libs.size(); k++) {
      if (e == g.edge_begin(n)) {
        data.t[k].wireC = eData.t[k].wireLoad->wireC(eData.wire);
      }
      data.t[k].pinC += succData.t[k].pinC;
    }
  }
}

void TimingGraph::computeArrivalByWire(GNode n, Graph::in_edge_iterator ie) {
  auto& data = g.getData(n);

  auto pred = g.getEdgeDst(ie);
  auto& predData = g.getData(pred);

  auto& ieData = g.getEdgeData(ie);

  for (size_t k = 0; k < libs.size(); k++) {
    MyFloat loadC = 0.0;
    if (TREE_TYPE_BALANCED == libs[k]->wireTreeType) {
      loadC = data.t[k].pinC;
    }
    else if (TREE_TYPE_WORST_CASE == libs[k]->wireTreeType) {
      loadC = predData.t[k].pinC;
    }
    auto delay = ieData.t[k].wireLoad->wireDelay(loadC, ieData.wire, data.pin);
    ieData.t[k].delay = delay;
    data.t[k].arrival = predData.t[k].arrival + delay;
    data.t[k].slew = predData.t[k].slew;
  }
}

void TimingGraph::computeArrivalByTimingArc(GNode n, Graph::in_edge_iterator ie, size_t k) {
  auto& data = g.getData(n);
  auto outPin = data.t[k].pin;
  if (!outPin) {
    return; // not driven by a timing arc
  }

  auto pred = g.getEdgeDst(ie);
  auto& predData = g.getData(pred);
  auto inPin = predData.t[k].pin;
  if (predData.isRealDummy()) {
    return; // invalid timing arc
  }

  bool isNeg = (data.isRise != predData.isRise);

  Parameter param = {
    {VARIABLE_INPUT_NET_TRANSITION,         predData.t[k].slew},
    {VARIABLE_TOTAL_OUTPUT_NET_CAPACITANCE, data.t[k].pinC + data.t[k].wireC}
  };

  Parameter paramNoC = {
    {VARIABLE_INPUT_NET_TRANSITION,         predData.t[k].slew},
    {VARIABLE_TOTAL_OUTPUT_NET_CAPACITANCE, 0.0}
  };

  auto& ieData = g.getEdgeData(ie);

  if (TIMING_MODE_MAX_DELAY == modes[k]) {
    auto delayResult = outPin->extractMax(param, TABLE_DELAY, inPin, isNeg, data.isRise);
    auto delay = delayResult.first;
    if (predData.isDrivingCell) {
      // offset for the primary inputs =
      //     delay for the driving cell with the load
      //   - delay for the driving cell without the load (Genus)
      delay -= outPin->extractMax(paramNoC, TABLE_DELAY, inPin, isNeg, data.isRise).first;
    }
    auto& when = delayResult.second;
    ieData.t[k].delay = delay;
    if (data.t[k].arrival < predData.t[k].arrival + delay) {
      data.t[k].arrival = predData.t[k].arrival + delay;
      if (isExactSlew) {
        data.t[k].slew = outPin->extract(param, TABLE_SLEW, inPin, isNeg, data.isRise, when);
      }
    }
    if (!isExactSlew) {
      auto slew = outPin->extractMax(param, TABLE_SLEW, inPin, isNeg, data.isRise).first;
      if (data.t[k].slew < slew) {
        data.t[k].slew = slew;
      }
    }
  }
  else {
    auto delayResult = outPin->extractMin(param, TABLE_DELAY, inPin, isNeg, data.isRise);
    auto delay = delayResult.first;
    if (predData.isDrivingCell) {
      // offset for the primary inputs =
      //     delay for the driving cell with the load
      //   - delay for the driving cell without the load (Genus)
      delay -= outPin->extractMin(paramNoC, TABLE_DELAY, inPin, isNeg, data.isRise).first;
    }
    auto& when = delayResult.second;
    ieData.t[k].delay = delay;
    if (data.t[k].arrival > predData.t[k].arrival + delay) {
      data.t[k].arrival = predData.t[k].arrival + delay;
      if (isExactSlew) {
        data.t[k].slew = outPin->extract(param, TABLE_SLEW, inPin, isNeg, data.isRise, when);
      }
    }
    if (!isExactSlew) {
      auto slew = outPin->extractMin(param, TABLE_SLEW, inPin, isNeg, data.isRise).first;
      if (data.t[k].slew > slew) {
        data.t[k].slew = slew;
      }
    }
  }
}

void TimingGraph::computeArrivalTime() {
  auto topoLIndexer = [&] (GNode n) {
    return g.getData(n, unprotected).topoL;
  };

  using FIFO = galois::worklists::PerThreadChunkFIFO<>;
  using LIFO = galois::worklists::PerThreadChunkLIFO<>;
  using OBIM
      = galois::worklists::OrderedByIntegerMetric<decltype(topoLIndexer), LIFO>
        ::template with_barrier<true>::type
//        ::template with_monotonic<true>::type
        ;

  size_t numLevels = g.getData(dummySink, unprotected).topoL + 1;
  std::vector<galois::GAccumulator<size_t>*> enqueued, executed;
  for (size_t i = 0; i < numLevels; i++) {
    enqueued.push_back(new galois::GAccumulator<size_t>);
    executed.push_back(new galois::GAccumulator<size_t>);
  }
  *(enqueued[g.getData(dummySrc, unprotected).topoL]) += 1;

  galois::for_each(
    galois::iterate({dummySrc}),
    [&] (GNode n, auto& ctx) {
      auto& data = g.getData(n);
      *(executed[data.topoL]) += 1;

      if (data.isGateInput() || data.isPseudoPrimaryOutput()) {
        // should have one incoming neighbor only
        for (auto ie: g.in_edges(n)) {
          computeArrivalByWire(n, ie);
        }
      }
      else if (data.isGateOutput() || data.isPseudoPrimaryInput()) {
        computeDriveC(n);
        for (auto ie: g.in_edges(n)) {
          for (size_t k = 0; k < libs.size(); k++) {
            computeArrivalByTimingArc(n, ie, k);
          }
        }
      }

      data.flag.store(false);

      // schedule outgoing neighbors
      for (auto e: g.edges(n)) {
        auto succ = g.getEdgeDst(e);
        auto& succData = g.getData(succ);
        if (/*(1 == succData.topoL - data.topoL) &&*/ !succData.isDummy) {
          auto& succInQueue = succData.flag;
          if (!succInQueue) {
            bool succQueued = false;
            if (succInQueue.compare_exchange_strong(succQueued, true)) {
              *(enqueued[succData.topoL]) += 1;
              ctx.push(succ);
            }
          }
        }
      }
    }
    , galois::loopname("TimingGraphComputeArrivalTime")
    , galois::no_conflicts()
    , galois::wl<OBIM>(topoLIndexer)
  );

  size_t totalEnqueued = 0;
  size_t totalExecuted = 0;
  for (size_t i = 0; i < numLevels; i++) {
    auto q = enqueued[i]->reduce();
    auto x = executed[i]->reduce();
    totalEnqueued += q;
    totalExecuted += x;
    if (q != x) {
      std::cout << "Level " << i << ": " << q << " enqueued, " << x << " executed." << std::endl;
    }
    delete enqueued[i];
    delete executed[i];
  }
  std::cout << totalEnqueued << " enqueued, " << totalExecuted << " executed." << std::endl;

  std::cout << "ComputeArrivalTime done." << std::endl;
  print();
}

std::string TimingGraph::getNodeName(GNode n) {
  auto& data = g.getData(n, unprotected);

  std::string nName;
  if (data.isDrivingCell) {
    nName = "Driving cell pin, ";
    nName += (data.isRise) ? "rise" : "fall";
  }
  else if (data.isPowerNode) {
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
    nName += "/";
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "rise" : "fall";
  }

  return nName;
}

void TimingGraph::print(std::ostream& os) {
  os << "Timing graph for module " << m.name << std::endl;

  g.sortAllEdgesByDst();

  for (auto n: g) {
    auto& data = g.getData(n, unprotected);
    if (data.isDrivingCell) {
      continue;
    }

    os << "  " << getNodeName(n) << std::endl;
    os << "    topoL = " << data.topoL << ", revTopoL = " << data.revTopoL << std::endl;
    os << "    outDeg = " << std::distance(g.edge_begin(n, unprotected), g.edge_end(n, unprotected));
    os << ", inDeg = " << std::distance(g.in_edge_begin(n, unprotected), g.in_edge_end(n, unprotected)) << std::endl;
    for (size_t k = 0; k < libs.size(); k++) {
      os << "    corner " << k;
      os << ": arrival = " << data.t[k].arrival;
      os << ", slew = " << data.t[k].slew;
      os << ", pinC = " << data.t[k].pinC;
      os << ", wireC = " << data.t[k].wireC;
      os << std::endl;
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
  } // end for n
}

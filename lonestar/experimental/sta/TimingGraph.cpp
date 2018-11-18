#include "TimingEngine.h"
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
    if (pin->name == name0) {
      data.nType = (0 == j) ? POWER_GND : DUMMY_POWER;
    }
    else if (pin->name == name1) {
      data.nType = (1 == j) ? DUMMY_POWER : POWER_VDD;
    }
    else if (m.isOutPin(pin)) {
      data.nType = PRIMARY_OUTPUT;
    }
    else {
      data.nType = PRIMARY_INPUT;
    }

    data.t.insert(data.t.begin(), engine->numCorners, NodeTiming());
    for (size_t k = 0; k < engine->numCorners; k++) {
      data.t[k].pin = nullptr;
    }
  }
}

void TimingGraph::addGate(VerilogGate* gate) {
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

      data.t.insert(data.t.begin(), engine->numCorners, NodeTiming());
      for (size_t k = 0; k < engine->numCorners; k++) {
        data.t[k].pin = engine->libs[k]->findCell(p->gate->cellType)->findCellPin(p->name);
      }

      auto dir = data.t[0].pin->pinDir();
      switch (dir) {
      case INPUT:
        data.nType = GATE_INPUT;
        break;
      case OUTPUT:
        data.nType = GATE_OUTPUT;
        break;
      case INOUT:
        data.nType = GATE_INOUT;
        break;
      case INTERNAL:
        data.nType = GATE_INTERNAL;
        break;
      }
    }
  }

  auto addTimingArc =
      [&] (GNode i, GNode o, bool isConstraint) {
        auto e = g.addMultiEdge(i, o, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = nullptr;
        eData.isConstraint = isConstraint;
        eData.t.insert(eData.t.begin(), engine->numCorners, EdgeTiming());
        for (size_t k = 0; k < engine->numCorners; k++) {
          eData.t[k].wireLoad = nullptr;
          eData.t[k].delay = 0.0;
        }
      };

  // add timing arcs among gate pins
  for (auto& ov: gate->pins) {
    auto ovp = ov.second;
    for (auto& iv: gate->pins) {
      auto ivp = iv.second;
      for (size_t i = 0; i < 2; i++) {
        auto on = nodeMap[ovp][i];
        auto op = g.getData(on, unprotected).t[0].pin;
        for (size_t j = 0; j < 2; j++) {
          auto in = nodeMap[ivp][j];
          auto ip = g.getData(in, unprotected).t[0].pin;
          if (op->isEdgeDefined(ip, j, i)) {
            addTimingArc(in, on, false);
          }
          else if (op->isEdgeDefined(ip, j, i, MIN_CONSTRAINT)) {
            addTimingArc(in, on, true);
          }
          else if (op->isEdgeDefined(ip, j, i, MAX_CONSTRAINT)) {
            addTimingArc(in, on, true);
          }
        } // end for j
      } // end for i
    } // end for iv
  } // end for ov
}

void TimingGraph::addWire(VerilogPin* p) {
  // p is the source of a wire
  for (size_t j = 0; j < 2; j++) {
    auto src = nodeMap[p][j];
    for (auto to: p->wire->pins) {
      if (to != p) {
        auto dst = nodeMap[to][j];
        auto e = g.addMultiEdge(src, dst, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = p->wire;
        eData.isConstraint = false;
        eData.t.insert(eData.t.begin(), engine->numCorners, EdgeTiming());
        for (size_t k = 0; k < engine->numCorners; k++) {
          eData.t[k].wireLoad = engine->libs[k]->defaultWireLoad;
          eData.t[k].delay = 0.0;
        }
      }
    }
  }
}

void TimingGraph::construct() {
  // add nodes for module ports
  for (auto& i: m.pins) {
    addPin(i.second);
  }

  // add nodes for gate pins
  for (auto& i: m.gates) {
    addGate(i.second);
  }

  // add edges for wires from power nodes or primary inputs
  for (auto& i: m.pins) {
    auto p = i.second;
    switch (g.getData(nodeMap[p][0], unprotected).nType) {
    case PRIMARY_INPUT:
    case DUMMY_POWER:
    case POWER_GND:
    case POWER_VDD:
      addWire(p);
      break;
    default:
      break;
    }
  }

  // add edges for wires from gate outputs
  for (auto& i: m.gates) {
    for (auto j: i.second->pins) {
      auto p = j.second;
      switch (g.getData(nodeMap[p][0], unprotected).nType) {
      case GATE_OUTPUT:
      case GATE_INOUT:
        addWire(p);
        break;
      default:
        break;
      }
    }
  }

  // get end points
  galois::do_all(
      galois::iterate(g),
      [&] (GNode n) {
        auto& data = g.getData(n);
        data.topoL = 1;
        data.revTopoL = 1;
        auto numEdges = std::distance(g.edge_begin(n), g.edge_end(n));
        if (!numEdges) { bFront.push_back(n); }
        auto numInEdges = std::distance(g.in_edge_begin(n), g.in_edge_end(n));
        if (!numInEdges) { fFront.push_back(n); }
      }
      , galois::loopname("ConstructTimingGraphFrontiers")
      , galois::steal()
  );
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

  std::cout << "Forward frontier:" << std::endl;
  for (auto& i: fFront) {
    std::cout << "  " << getNodeName(i) << std::endl;
  }

  galois::for_each(
      galois::iterate(fFront),
      [&] (GNode n, auto& ctx) {
        auto& data = g.getData(n);
        if (!data.flag.load()) {
          return; // this node is already done
        }

        size_t myTopoL = 1;
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

#if 1
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

  std::cout << "Backward frontier:" << std::endl;
  for (auto& i: bFront) {
    std::cout << "  " << getNodeName(i) << std::endl;
  }

  galois::for_each(
      galois::iterate(bFront),
      [&] (GNode n, auto& ctx) {
        auto& data = g.getData(n);
        if (!data.flag.load()) {
          return; // this node is already done
        }

        size_t myRevTopoL = 1;
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

#if 1
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

        // for timing computation
        bool isTimingSource =
            (PRIMARY_INPUT == data.nType) ||
            (POWER_VDD == data.nType) ||
            (POWER_GND == data.nType);
        for (size_t k = 0; k < engine->numCorners; k++) {
          if (MAX_DELAY_MODE == engine->modes[k]) {
            data.t[k].slew = 0.0;
            data.t[k].arrival = (isTimingSource) ? 0.0 : -infinity;
            data.t[k].required = infinity;
          }
          else {
            data.t[k].slew = (isTimingSource) ? 0.0 : infinity;
            data.t[k].arrival = (isTimingSource) ? 0.0 : infinity;
            data.t[k].required = -infinity;
          }

          data.t[k].slack = infinity;
          data.t[k].pinC = (GATE_INPUT == data.nType) ? data.t[k].pin->c[data.isRise] : 0.0;
          data.t[k].wireC = 0.0;
        }
      }
      , galois::loopname("TimingGraphInitialize")
      , galois::steal()
  );

  computeTopoL();
  computeRevTopoL();
  std::cout << "Levelization done.\n" << std::endl;
  print();
}

void TimingGraph::setConstraints(SDC& sdc) {
}

void TimingGraph::computeDriveC(GNode n) {
  auto& data = g.getData(n);

  for (auto e: g.edges(n)) {
    auto& eData = g.getEdgeData(e);

    auto succ = g.getEdgeDst(e);
    auto& succData = g.getData(succ);

    for (size_t k = 0; k < engine->numCorners; k++) {
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

  for (size_t k = 0; k < engine->numCorners; k++) {
    MyFloat loadC = 0.0;
    if (BALANCED_TREE == engine->libs[k]->wireTreeType) {
      loadC = data.t[k].pinC;
    }
    else if (WORST_CASE_TREE == engine->libs[k]->wireTreeType) {
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

  auto pred = g.getEdgeDst(ie);
  auto& predData = g.getData(pred);
  auto inPin = predData.t[k].pin;

  Parameter param = {
    {INPUT_NET_TRANSITION,         predData.t[k].slew},
    {TOTAL_OUTPUT_NET_CAPACITANCE, data.t[k].pinC + data.t[k].wireC}
  };

  auto& ieData = g.getEdgeData(ie);

  if (MAX_DELAY_MODE == engine->modes[k]) {
    auto delayResult = outPin->extractMax(param, DELAY, inPin, predData.isRise, data.isRise);
    auto delay = delayResult.first;
    auto& when = delayResult.second;
    ieData.t[k].delay = delay;
    if (data.t[k].arrival < predData.t[k].arrival + delay) {
      data.t[k].arrival = predData.t[k].arrival + delay;
      if (engine->isExactSlew) {
        data.t[k].slew = outPin->extract(param, SLEW, inPin, predData.isRise, data.isRise, when);
      }
    }
    if (!engine->isExactSlew) {
      auto slew = outPin->extractMax(param, SLEW, inPin, predData.isRise, data.isRise).first;
      if (data.t[k].slew < slew) {
        data.t[k].slew = slew;
      }
    }
  }
  else {
    auto delayResult = outPin->extractMin(param, DELAY, inPin, predData.isRise, data.isRise);
    auto delay = delayResult.first;
    auto& when = delayResult.second;
    ieData.t[k].delay = delay;
    if (data.t[k].arrival > predData.t[k].arrival + delay) {
      data.t[k].arrival = predData.t[k].arrival + delay;
      if (engine->isExactSlew) {
        data.t[k].slew = outPin->extract(param, SLEW, inPin, predData.isRise, data.isRise, when);
      }
    }
    if (!engine->isExactSlew) {
      auto slew = outPin->extractMin(param, SLEW, inPin, predData.isRise, data.isRise).first;
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

  using LIFO = galois::worklists::PerThreadChunkLIFO<>;
  using OBIM
      = galois::worklists::OrderedByIntegerMetric<decltype(topoLIndexer), LIFO>
        ::template with_barrier<true>::type
//        ::template with_monotonic<true>::type
        ;

  galois::for_each(
    galois::iterate(fFront),
    [&] (GNode n, auto& ctx) {
      auto& data = g.getData(n);

      switch (data.nType) {
      case GATE_INPUT:
      case PRIMARY_OUTPUT:
        // should have only one incoming neighbor of wire
        for (auto ie: g.in_edges(n)) {
          this->computeArrivalByWire(n, ie);
        }
        break;
      case GATE_OUTPUT:
      case PRIMARY_INPUT:
        this->computeDriveC(n);
        for (auto ie: g.in_edges(n)) {
          for (size_t k = 0; k < engine->numCorners; k++) {
            this->computeArrivalByTimingArc(n, ie, k);
          }
        }
        break;
      default:
        break;
      }

      data.flag.store(false);

      // schedule outgoing neighbors
      for (auto e: g.edges(n)) {
        auto succ = g.getEdgeDst(e);
        auto& succData = g.getData(succ);
        if (/*(1 == succData.topoL - data.topoL) &&*/ true) {
          auto& succInQueue = succData.flag;
          if (!succInQueue) {
            bool succQueued = false;
            if (succInQueue.compare_exchange_strong(succQueued, true)) {
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

  std::cout << "ComputeArrivalTime done." << std::endl;
  print();
}

std::string TimingGraph::getNodeName(GNode n) {
  auto& data = g.getData(n, unprotected);

  std::string nName;
  switch (data.nType) {
  case POWER_VDD:
  case POWER_GND:
  case DUMMY_POWER:
    nName = "Power ";
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "r" : "f";
    break;
#if 0
  case DUMMY_SOURCE:
    nName = "Dummy input";
    break;
  case DUMMY_SINK:
    nName = "Dummy output";
    break;
#endif
  case PRIMARY_OUTPUT:
    nName = "Primary output " + data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "r" : "f";
    break;
  case PRIMARY_INPUT:
    nName = "Primary input " + data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "r" : "f";
    break;
  case GATE_OUTPUT:
    nName = "Gate output ";
    nName += data.pin->gate->name;
    nName += "/";
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "r" : "f";
    break;
  case GATE_INPUT:
    nName = "Gate input ";
    nName += data.pin->gate->name;
    nName += "/";
    nName += data.pin->name;
    nName += ", ";
    nName += (data.isRise) ? "r" : "f";
    break;
  default:
    nName = "(NOT_HANDLED_PIN_TYPE)";
    break;
  }

  return nName;
}

void TimingGraph::print(std::ostream& os) {
  os << "Timing graph for module " << m.name << std::endl;

  g.sortAllEdgesByDst();

  for (auto n: g) {
    auto& data = g.getData(n, unprotected);

    os << "  " << getNodeName(n) << std::endl;
    os << "    topoL = " << data.topoL << ", revTopoL = " << data.revTopoL << std::endl;
    os << "    outDeg = " << std::distance(g.edge_begin(n, unprotected), g.edge_end(n, unprotected));
    os << ", inDeg = " << std::distance(g.in_edge_begin(n, unprotected), g.in_edge_end(n, unprotected)) << std::endl;
    for (size_t k = 0; k < engine->numCorners; k++) {
      os << "    corner " << k;
      os << ": arrival = " << data.t[k].arrival;
      os << ", slew = " << data.t[k].slew;
      os << ", pinC = " << data.t[k].pinC;
      os << ", wireC = " << data.t[k].wireC;
      os << std::endl;
    }

    for (auto ie: g.in_edges(n)) {
      auto& eData = g.getEdgeData(ie);
      auto w = eData.wire;
      if (w) {
        os << "    Wire " << w->name;
      }
      else {
        os << "    Timing arc";
        if (eData.isConstraint) {
          os << " (constraint)";
        }
      }
      os << " from " << getNodeName(g.getEdgeDst(ie)) << std::endl;

      for (size_t k = 0; k < engine->numCorners; k++) {
        os << "    corner " << k;
        os << ": delay = " << eData.t[k].delay;
        os << std::endl;
      }
    }

    for (auto e: g.edges(n)) {
      auto& eData = g.getEdgeData(e);
      auto w = eData.wire;
      if (w) {
        os << "    Wire " << w->name;
      }
      else {
        os << "    Timing arc";
        if (eData.isConstraint) {
          os << " (constraint)";
        }
      }
      os << " to " << getNodeName(g.getEdgeDst(e)) << std::endl;

      for (size_t k = 0; k < engine->numCorners; k++) {
        os << "    corner " << k;
        os << ": delay = " << eData.t[k].delay;
        os << std::endl;
      }
    }
  } // end for n
}

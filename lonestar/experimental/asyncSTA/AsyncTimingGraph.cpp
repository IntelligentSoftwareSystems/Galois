#include "AsyncTimingEngine.h"
#include "AsyncTimingGraph.h"
#include "galois/Bag.h"
#include "galois/Reduction.h"

#include "galois/runtime/Profile.h"

#include <unordered_set>
#include <string>
#include <iostream>
#include <map>

static auto unprotected = galois::MethodFlag::UNPROTECTED;
static std::string name0 = "1\'b0";
static std::string name1 = "1\'b1";

void AsyncTimingGraph::addPin(VerilogPin* pin) {
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
      data.nType = (1 == j) ? POWER_VDD : DUMMY_POWER;
    }
    else if (m.isOutPin(pin)) {
      data.nType = PRIMARY_OUTPUT;
    }
    else {
      data.nType = PRIMARY_INPUT;
    }

    data.t.insert(data.t.begin(), engine->numCorners, NodeAsyncTiming());
    for (size_t k = 0; k < engine->numCorners; k++) {
      data.t[k].pin = nullptr;
    }
  }
}

void AsyncTimingGraph::addGate(VerilogGate* gate) {
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

      data.t.insert(data.t.begin(), engine->numCorners, NodeAsyncTiming());
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

  auto addAsyncTimingArc =
      [&] (GNode i, GNode o, bool isTicked) {
        auto e = g.addMultiEdge(i, o, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = nullptr;
        eData.isTicked = isTicked;
        eData.t.insert(eData.t.begin(), engine->numCorners, EdgeAsyncTiming());
        for (size_t k = 0; k < engine->numCorners; k++) {
          eData.t[k].wireLoad = nullptr;
          eData.t[k].delay = 0.0;
        }
      };

  auto arcSet = engine->arcs->findArcSetForModule(&m);

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
          // only connect legal & required timing arcs
          if (op->isEdgeDefined(ip, j, i) && arcSet->isRequiredArc(ivp, j, ovp, i)) {
            addAsyncTimingArc(in, on, arcSet->isTickedArc(ivp, j, ovp, i));
          }
        } // end for j
      } // end for i
    } // end for iv
  } // end for ov
}

void AsyncTimingGraph::setWireLoad(WireLoad** wWL, WireLoad* wl) {
  assert(wWL);
  *wWL = (nullptr == wl) ? idealWireLoad : wl;
}

void AsyncTimingGraph::addWire(VerilogPin* p) {
  // p is the source of a wire
  for (size_t j = 0; j < 2; j++) {
    auto src = nodeMap[p][j];
    for (auto to: p->wire->pins) {
      if (to != p) {
        auto dst = nodeMap[to][j];
        auto e = g.addMultiEdge(src, dst, unprotected);
        auto& eData = g.getEdgeData(e);
        eData.wire = p->wire;
        eData.t.insert(eData.t.begin(), engine->numCorners, EdgeAsyncTiming());
        for (size_t k = 0; k < engine->numCorners; k++) {
          setWireLoad(&(eData.t[k].wireLoad), engine->libs[k]->defaultWireLoad);
          eData.t[k].delay = 0.0;
        }
      }
    }
  }
}

void AsyncTimingGraph::construct() {
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
//  outDegHistogram();
//  inDegHistogram();
}

size_t AsyncTimingGraph::outDegree(GNode n) {
  return std::distance(g.edge_begin(n, unprotected), g.edge_end(n, unprotected));
}

size_t AsyncTimingGraph::inDegree(GNode n) {
  return std::distance(g.in_edge_begin(n, unprotected), g.in_edge_end(n, unprotected));
}

void AsyncTimingGraph::initialize() {
  galois::do_all(
      galois::iterate(g),
      [&] (GNode n) {
        auto& data = g.getData(n, unprotected);

        // for timing computation
        for (size_t k = 0; k < engine->numCorners; k++) {
          data.t[k].pinC = (GATE_INPUT == data.nType) ? data.t[k].pin->c[data.isRise] : 0.0;
          data.t[k].wireC = 0.0;
          data.t[k].slew = 0.0;
          data.t[k].tmpSlew[0] = 0.0;
          data.t[k].tmpSlew[1] = 0.0;
        }
      }
      , galois::loopname("Initialization")
      , galois::steal()
  );
}

void AsyncTimingGraph::computeDriveC(GNode n) {
  auto& data = g.getData(n);

  for (auto e: g.edges(n, unprotected)) {
    auto& eData = g.getEdgeData(e);

    auto succ = g.getEdgeDst(e);
    auto& succData = g.getData(succ, unprotected);

    for (size_t k = 0; k < engine->numCorners; k++) {
      if (e == g.edge_begin(n, unprotected)) {
        data.t[k].wireC = (engine->isWireIdeal) ? idealWireLoad->wireC(eData.wire) :
            eData.t[k].wireLoad->wireC(eData.wire);
        data.t[k].pinC = succData.t[k].pinC;
      }
      else {
        data.t[k].pinC += succData.t[k].pinC;
      }
    }
  }
}

void AsyncTimingGraph::computeExtremeSlew(GNode n, galois::PerIterAllocTy& alloc) {
  auto& data = g.getData(n);

  for (auto ie: g.in_edges(n, unprotected)) {
    auto pred = g.getEdgeDst(ie);
    auto& predData = g.getData(pred, unprotected);
    auto& ieData = g.getEdgeData(ie);

    for (size_t k = 0; k < engine->numCorners; k++) {
      // from a wire. take the predecessor's slew
      if (ieData.wire) {
        data.t[k].slew = predData.t[k].slew;
      }
      // from a timing arc. compute and take the extreme one
      else {
        Parameter param(alloc);
        param[INPUT_NET_TRANSITION] = predData.t[k].slew;
        param[TOTAL_OUTPUT_NET_CAPACITANCE] = data.t[k].pinC + data.t[k].wireC;

        auto outPin = data.t[k].pin;
        auto inPin = predData.t[k].pin;
        if (MAX_DELAY_MODE == engine->modes[k]) {
          auto slew = outPin->extractMax(param, SLEW, inPin, predData.isRise, data.isRise, alloc).first;
          if (data.t[k].slew < slew) {
            data.t[k].slew = slew;
          }
        }
        else {
          auto slew = outPin->extractMin(param, SLEW, inPin, predData.isRise, data.isRise, alloc).first;
          if (data.t[k].slew > slew) {
            data.t[k].slew = slew;
          }
        }
      } // end else for ieDada.wire
    } // end for k
  } // end for ie
}

void AsyncTimingGraph::computeExtremeDelay(GNode n, galois::PerIterAllocTy& alloc) {
  auto& data = g.getData(n);

  for (auto ie: g.in_edges(n, unprotected)) {
    auto pred = g.getEdgeDst(ie);
    auto& predData = g.getData(pred, unprotected);
    auto& ieData = g.getEdgeData(ie);

    for (size_t k = 0; k < engine->numCorners; k++) {
      // from a wire
      if (ieData.wire) {
        MyFloat delay = 0.0;

        if (engine->isWireIdeal) {
          delay = idealWireLoad->wireDelay(0.0, ieData.wire, data.pin);
        }
        else {
          auto ieWL = ieData.t[k].wireLoad;
          MyFloat loadC = data.t[k].pinC;
          if (dynamic_cast<PreLayoutWireLoad*>(ieWL)) {
            if (WORST_CASE_TREE == engine->libs[k]->wireTreeType) {
              loadC = predData.t[k].pinC; // the sum of all pin capacitance in the net
            }
          }
          delay = ieWL->wireDelay(loadC, ieData.wire, data.pin);
        }

        ieData.t[k].delay = delay;
      }
      // from a timing arc
      else {
        Parameter param(alloc);
        param[INPUT_NET_TRANSITION] = predData.t[k].slew;
        param[TOTAL_OUTPUT_NET_CAPACITANCE] = data.t[k].pinC + data.t[k].wireC;

        auto outPin = data.t[k].pin;
        auto inPin = predData.t[k].pin;

        if (MAX_DELAY_MODE == engine->modes[k]) {
          auto delay = outPin->extractMax(param, DELAY, inPin, predData.isRise, data.isRise, alloc).first;
          ieData.t[k].delay = delay;
        }
        else {
          auto delay = outPin->extractMin(param, DELAY, inPin, predData.isRise, data.isRise, alloc).first;
          ieData.t[k].delay = delay;
        }
      } // end else for (ieData.wire)
    } // end for k
  } // end for ie
}

std::string AsyncTimingGraph::getNodeName(GNode n) {
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

void AsyncTimingGraph::print(std::ostream& os) {
  os << "AsyncTiming graph for module " << m.name << std::endl;

  g.sortAllEdgesByDst();

  for (auto n: g) {
    auto& data = g.getData(n, unprotected);

    os << "  " << getNodeName(n) << std::endl;
//    os << "    topoL = " << data.topoL << ", revTopoL = " << data.revTopoL << std::endl;
    os << "    outDeg = " << outDegree(n);
    os << ", inDeg = " << inDegree(n) << std::endl;
    for (size_t k = 0; k < engine->numCorners; k++) {
      os << "    corner " << k;
      os << ": slew = " << data.t[k].slew;
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
        if (eData.isTicked) {
          os << " (ticked)";
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
        if (eData.isTicked) {
          os << " (ticked)";
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

#include "StaticTimingAnalysis.h"
#include "Verilog.h"
#include "CellLib.h"

#include "galois/Bag.h"
#include "galois/Timer.h"

#include <iostream>
#include <string>

struct ComputeRequiredTime {
  Graph& g;
  ComputeRequiredTime(Graph& g): g(g) {}

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    auto& data = g.getData(n);

    if (data.isDummy) {
      // dummy output, i.e. dummySink
      if (data.isOutput) {
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
      // skip dummy input, i.e. dummySrc
    }

    else if (data.isPrimary) {
      // primary output
      if (data.isOutput) {
        data.rise.slack = data.rise.requiredTime - data.rise.arrivalTime;
        data.fall.slack = data.fall.requiredTime - data.fall.arrivalTime;
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
      // primary input
      else {
        for (auto oe: g.edges(n)) {
          auto& outData = g.getData(g.getEdgeDst(oe));

          auto riseRequiredTime = outData.rise.requiredTime;
          if (data.rise.requiredTime > riseRequiredTime) {
            data.rise.requiredTime = riseRequiredTime;
            data.rise.slack = riseRequiredTime - data.rise.arrivalTime;
          }

          auto fallRequiredTime = outData.fall.requiredTime;
          if (data.fall.requiredTime > fallRequiredTime) {
            data.fall.requiredTime = fallRequiredTime;
            data.fall.slack = fallRequiredTime - data.fall.arrivalTime;
          }
        }
      }
    }

    // gate output
    else if (data.isOutput) {
      // lock all incoming neighbors
      g.in_edges(n);

      bool changed = false;
      for (auto oe: g.edges(n)) {
        auto& outData = g.getData(g.getEdgeDst(oe));
        auto& oeData = g.getEdgeData(oe);

        auto riseRequiredTime = outData.rise.requiredTime - oeData.riseDelay;
        if (data.rise.requiredTime > riseRequiredTime) {
          data.rise.requiredTime = riseRequiredTime;
          data.rise.slack = riseRequiredTime - data.rise.arrivalTime;
          changed = true;
        }

        auto fallRequiredTime = outData.fall.requiredTime - oeData.fallDelay;
        if (data.fall.requiredTime > fallRequiredTime) {
          data.fall.requiredTime = fallRequiredTime;
          data.fall.slack = fallRequiredTime - data.fall.arrivalTime;
          changed = true;
        }
      }

      if (changed) {
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
    }

    // gate input
    else {
      // lock all incoming neighbors
      g.in_edges(n);

      bool changed = false;
      for (auto oe: g.edges(n)) {
        auto& oeData= g.getEdgeData(oe);
        auto& outData = g.getData(g.getEdgeDst(oe));
        auto cellOutPin = outData.pin->gate->cell->outPins.at(outData.pin->name);

        auto requiredTimeForOutRise = outData.rise.requiredTime - oeData.riseDelay;
        // positive unate for outgoing rising edge
        if (cellOutPin->cellRise.count({data.pin->name, TIMING_SENSE_POSITIVE_UNATE})) {
          if (data.rise.requiredTime > requiredTimeForOutRise) {
            data.rise.requiredTime = requiredTimeForOutRise;
            data.rise.slack = requiredTimeForOutRise - data.rise.arrivalTime;
            changed = true;
          }
        }
        // negative unate for outgoing rising edge
        if (cellOutPin->cellRise.count({data.pin->name, TIMING_SENSE_NEGATIVE_UNATE})) {
          if (data.fall.requiredTime > requiredTimeForOutRise) {
            data.fall.requiredTime = requiredTimeForOutRise;
            data.fall.slack = requiredTimeForOutRise - data.fall.arrivalTime;
            changed = true;
          }
        }

        auto requiredTimeForOutFall = outData.fall.requiredTime - oeData.fallDelay;
        // positive unate for outgoing falling edge
        if (cellOutPin->cellFall.count({data.pin->name, TIMING_SENSE_POSITIVE_UNATE})) {
          if (data.fall.requiredTime > requiredTimeForOutFall) {
            data.fall.requiredTime = requiredTimeForOutFall;
            data.fall.slack = requiredTimeForOutFall - data.fall.arrivalTime;
            changed = true;
          }
        }
        // negative unate for outgoing falling edge
        if (cellOutPin->cellFall.count({data.pin->name, TIMING_SENSE_NEGATIVE_UNATE})) {
          if (data.rise.requiredTime > requiredTimeForOutFall) {
            data.rise.requiredTime = requiredTimeForOutFall;
            data.rise.slack = requiredTimeForOutFall - data.rise.arrivalTime;
            changed = true;
          }
        }
      }

      if (changed) {
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
    }
  } // end operator()
}; // end struct ComputeRequiredTime

static void computeRequiredTime(Graph& g, GNode dummySink) {
  galois::InsertBag<GNode> work;
  work.push_back(dummySink);
  galois::for_each_local(work, ComputeRequiredTime{g}, galois::loopname("ComputeRequiredTime"), galois::timeit());
}

struct ComputeArrivalTimeAndPower {
  Graph& g;
  ComputeArrivalTimeAndPower(Graph& g): g(g) {}

  void updateEdgeDelay(Edge& eData, float pinC) {
    auto wire = eData.wire;
    auto wireDeg = wire->leaves.size();

    // for balanced-case RC tree
    auto wireR = wire->wireLoad->wireResistance(wireDeg) / (float)(wireDeg);
    auto wireC = wire->wireLoad->wireCapacitance(wireDeg) / (float)(wireDeg);
    auto wireDelay = wireR * (wireC + pinC);

    eData.riseDelay = wireDelay;
    eData.fallDelay = wireDelay;
  }

  void updateGateInputAndPrimaryOutput(Node& data, Node& inData) {
    data.rise.slew = inData.rise.slew;
    data.rise.arrivalTime = inData.rise.arrivalTime;
    data.fall.slew = inData.fall.slew;
    data.fall.arrivalTime = inData.fall.arrivalTime;
  }

  void updateGateOutput(TimingPowerInfo& info, float pinC, float netC, float& edgeDelay,
    std::string inPinName, TimingPowerInfo& inPosInfo, TimingPowerInfo& inNegInfo,
    CellPin::MapOfTableSet& delayMap,
    CellPin::MapOfTableSet& transitionMap,
    CellPin::MapOfTableSet& powerMap)
  {
    auto posMapI = delayMap.find({inPinName, TIMING_SENSE_POSITIVE_UNATE});
    std::vector<float> posVTotalC = {inPosInfo.slew, pinC + netC};
    std::pair<float, std::string> pos = {-std::numeric_limits<float>::infinity(), ""};
    if (posMapI != delayMap.end()) {
      pos = extractMaxFromTableSet(posMapI->second, posVTotalC);
    }
    float posArrivalTime = inPosInfo.arrivalTime + pos.first;

    auto negMapI = delayMap.find({inPinName, TIMING_SENSE_NEGATIVE_UNATE});
    std::vector<float> negVTotalC = {inNegInfo.slew, pinC + netC};
    std::pair<float, std::string> neg = {-std::numeric_limits<float>::infinity(), ""};
    if (negMapI != delayMap.end()) {
      neg = extractMaxFromTableSet(negMapI->second, negVTotalC);
    }
    float negArrivalTime = inNegInfo.arrivalTime + neg.first;

    // take the parameters for larger arrival time
    bool isPos = (posArrivalTime >= negArrivalTime);
    auto arrivalTime = (isPos) ? posArrivalTime : negArrivalTime;
    auto delay = (isPos) ? pos.first : neg.first;
    auto when = (isPos) ? pos.second : neg.second;
    auto t = (isPos) ? TIMING_SENSE_POSITIVE_UNATE : TIMING_SENSE_NEGATIVE_UNATE;
    auto& inInfo = (isPos) ? inPosInfo : inNegInfo;
    auto& vTotalC = (isPos) ? posVTotalC : negVTotalC;

    if (info.arrivalTime < arrivalTime) {
      // update critical path
      edgeDelay = delay;
      info.arrivalTime = arrivalTime;
      info.slew = extractMaxFromTableSet(transitionMap.at({inPinName, t}), vTotalC).first;

      // power follows critical path
      auto& powerTables = powerMap.at({inPinName, t});
      std::vector<float> vPinC = {inInfo.slew, pinC};
      info.internalPower = extractMaxFromTableSet(powerTables, vPinC).first;
      info.netPower = extractMaxFromTableSet(powerTables, vTotalC).first - info.internalPower;
    }
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    auto& data = g.getData(n);

    if (data.isDummy) {
      // dummy input, i.e. dummySrc
      if (!data.isOutput) {
        for (auto oe: g.edges(n)) {
          ctx.push(g.getEdgeDst(oe));
        }
      }
      // skip dummy output, i.e. dummySink
    }

    else if (data.isPrimary) {
      // primary output
      if (data.isOutput) {
        for (auto ie: g.in_edges(n)) {
          updateGateInputAndPrimaryOutput(data, g.getData(g.getEdgeDst(ie)));
        }
      }
      // primary input
      else {
        for (auto oe: g.edges(n)) {
          ctx.push(g.getEdgeDst(oe));
        }
      }
    }

    // gate input
    else if (!data.isOutput) {
      // lock outgoing neighbors
      g.edges(n);

      auto pin = data.pin;
      auto pinC = pin->gate->cell->inPins.at(pin->name)->capacitance;
      for (auto ie: g.in_edges(n)) {
        auto& inData = g.getData(g.getEdgeDst(ie));
        updateGateInputAndPrimaryOutput(data, inData);

        // don't consider wire delay for primary inputs
        if (!inData.isPrimary) {
          auto& ieData = g.getEdgeData(ie);
          updateEdgeDelay(ieData, pinC);
          data.rise.arrivalTime += ieData.riseDelay;
          data.fall.arrivalTime += ieData.fallDelay;
        }
      }

      for (auto oe: g.edges(n)) {
        auto outNgh = g.getEdgeDst(oe);
        auto& outData = g.getData(outNgh);
        // schedule only after all precondition cleared for preserving topological levels
        outData.precondition -= 1;
        if (0 == outData.precondition) {
          ctx.push(outNgh);
        }
      }
    }

    // gate output
    else {
      // lock incoming neighbors
      g.in_edges(n);

      data.totalPinC = 0.0;
      for (auto oe: g.edges(n)) {
        auto& outData = g.getData(g.getEdgeDst(oe));
        auto outPin = outData.pin;
        if (outPin->gate) {
          data.totalPinC += outPin->gate->cell->inPins.at(outPin->name)->capacitance;
        }
        else {
          // primary output, already recorded
          data.totalPinC += outData.totalPinC;
        }
      }

      auto cellOutPin = data.pin->gate->cell->outPins.at(data.pin->name);
      for (auto ie: g.in_edges(n)) {
        auto& inData = g.getData(g.getEdgeDst(ie));
        auto& ieData = g.getEdgeData(ie);

        // rising edge
        updateGateOutput(data.rise, data.totalPinC, data.totalNetC, ieData.riseDelay,
          inData.pin->name, inData.rise, inData.fall,
          cellOutPin->cellRise, cellOutPin->riseTransition, cellOutPin->risePower);

        // falling edge
        updateGateOutput(data.fall, data.totalPinC, data.totalNetC, ieData.fallDelay,
          inData.pin->name, inData.fall, inData.rise,
          cellOutPin->cellFall, cellOutPin->fallTransition, cellOutPin->fallPower);
      } // end for ie

      for (auto oe: g.edges(n)) {
        ctx.push(g.getEdgeDst(oe));
      }
    } // end else (data.isOutput)
  } // end operator()
}; // end struct ComputeArrivalTimeAndPower

struct SetForwardPrecondition {
  Graph& g;
  SetForwardPrecondition(Graph& g): g(g) {}

  void operator()(GNode n) {
    auto& data = g.getData(n);
    if (data.isOutput && !data.isDummy && !data.isPrimary) {
      data.precondition = std::distance(g.in_edge_begin(n), g.in_edge_end(n));
    }
  }
};

static void computeArrivalTimeAndPower(Graph& g, GNode dummySrc) {
  galois::do_all_local(g, SetForwardPrecondition{g}, galois::do_all_steal<true>());
  galois::InsertBag<GNode> work;
  work.push_back(dummySrc);
  galois::for_each_local(work, ComputeArrivalTimeAndPower{g}, galois::loopname("ComputeArrivalTimeAndPower"), galois::timeit());
}

void doStaticTimingAnalysis(CircuitGraph& graph) {
  computeArrivalTimeAndPower(graph.g, graph.dummySrc);
  computeRequiredTime(graph.g, graph.dummySink);
}

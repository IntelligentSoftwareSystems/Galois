#include "StaticTimingAnalysis.h"
#include "Verilog.h"
#include "CellLib.h"

#include "Galois/Bag.h"
#include "Galois/Statistic.h"

#include <iostream>
#include <string>

struct ComputeRequiredTime {
  Graph& g;
  ComputeRequiredTime(Graph& g): g(g) {}

  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
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

        auto riseRequiredTime = outData.rise.requiredTime;
        if (data.rise.requiredTime > riseRequiredTime) {
          data.rise.requiredTime = riseRequiredTime;
          data.rise.slack = riseRequiredTime - data.rise.arrivalTime;
          changed = true;
        }

        auto fallRequiredTime = outData.fall.requiredTime;
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
        auto& outData = g.getData(g.getEdgeDst(oe));
        auto& oeData= g.getEdgeData(oe);

        auto outRiseRequiredTime = outData.rise.requiredTime - oeData.riseDelay;
        auto outFallRequiredTime = outData.fall.requiredTime - oeData.fallDelay;
        auto tSense = outData.pin->gate->cell->outPins.at(outData.pin->name)->tSense.at(data.pin->name);

        auto riseRequiredTime = (TIMING_SENSE_POSITIVE_UNATE == tSense) ? outRiseRequiredTime : outFallRequiredTime;
        if (data.rise.requiredTime > riseRequiredTime) {
          data.rise.requiredTime = riseRequiredTime;
          data.rise.slack = riseRequiredTime - data.rise.arrivalTime;
          changed = true;
        }

        auto fallRequiredTime = (TIMING_SENSE_POSITIVE_UNATE == tSense) ? outFallRequiredTime : outRiseRequiredTime;
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
  } // end operator()
}; // end struct ComputeRequiredTime

static void computeRequiredTime(Graph& g, GNode dummySink) {
  Galois::StatTimer TRequiredTime("RequiredTime");
  TRequiredTime.start();
  Galois::InsertBag<GNode> work;
  work.push_back(dummySink);
  Galois::for_each_local(work, ComputeRequiredTime{g}, Galois::loopname("ComputeRequiredTime"));
  TRequiredTime.stop();
}

struct ComputeArrivalTimeAndPower {
  Graph& g;
  ComputeArrivalTimeAndPower(Graph& g): g(g) {}

  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
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
          auto& inData = g.getData(g.getEdgeDst(ie));
          data.rise.slew = inData.rise.slew;
          data.rise.arrivalTime = inData.rise.arrivalTime;
          data.fall.slew = inData.fall.slew;
          data.fall.arrivalTime = inData.fall.arrivalTime;
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

      for (auto ie: g.in_edges(n)) {
        auto& inData = g.getData(g.getEdgeDst(ie));
        data.rise.slew = inData.rise.slew;
        data.rise.arrivalTime = inData.rise.arrivalTime;
        data.fall.slew = inData.fall.slew;
        data.fall.arrivalTime = inData.fall.arrivalTime;
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
        auto tSense = cellOutPin->tSense.at(inData.pin->name);

        // rising edge
        auto cellRise = cellOutPin->cellRise.at(inData.pin->name);
        TimingPowerInfo *infoForRise = (TIMING_SENSE_POSITIVE_UNATE == tSense) ? &(inData.rise) : &(inData.fall);
        std::vector<float> riseVTotalC = {infoForRise->slew, data.totalPinC + data.totalNetC};
        ieData.riseDelay = cellRise->lookup(riseVTotalC);

        auto newRiseArrivalTime = infoForRise->arrivalTime + ieData.riseDelay;
        if (data.rise.arrivalTime < newRiseArrivalTime) {
          // update critical path
          data.rise.arrivalTime = newRiseArrivalTime;
          auto riseTransition = cellOutPin->riseTransition.at(inData.pin->name);
          data.rise.slew = riseTransition->lookup(riseVTotalC);

          // power follows critical path
          auto risePower = cellOutPin->risePower.at(inData.pin->name);
          std::vector<float> riseVPinC = {infoForRise->slew, data.totalPinC};
          data.rise.internalPower = risePower->lookup(riseVPinC);
          data.rise.netPower = risePower->lookup(riseVTotalC) - data.rise.internalPower;
        }

        // falling edge
        auto cellFall = cellOutPin->cellFall.at(inData.pin->name);
        TimingPowerInfo *infoForFall = (TIMING_SENSE_POSITIVE_UNATE == tSense) ? &(inData.fall) : &(inData.rise);
        std::vector<float> fallVTotalC = {infoForFall->slew, data.totalPinC + data.totalNetC};
        ieData.fallDelay = cellFall->lookup(fallVTotalC);

        auto newFallArrivalTime = infoForFall->arrivalTime + ieData.fallDelay;
        if (data.fall.arrivalTime < newFallArrivalTime) {
          // update critical path
          data.fall.arrivalTime = newFallArrivalTime;
          auto fallTransition = cellOutPin->fallTransition.at(inData.pin->name);
          data.fall.slew = fallTransition->lookup(fallVTotalC);

          // power follows critical path
          auto fallPower = cellOutPin->fallPower.at(inData.pin->name);
          std::vector<float> fallVPinC = {infoForFall->slew, data.totalPinC};
          data.fall.internalPower = fallPower->lookup(fallVPinC);
          data.fall.netPower = fallPower->lookup(fallVTotalC) - data.fall.internalPower;
        }
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
  Galois::StatTimer TArrivalTimeAndPower("ArrivalTimeAndPower");
  TArrivalTimeAndPower.start();
  Galois::do_all_local(g, SetForwardPrecondition{g}, Galois::do_all_steal<true>());
  Galois::InsertBag<GNode> work;
  work.push_back(dummySrc);
  Galois::for_each_local(work, ComputeArrivalTimeAndPower{g}, Galois::loopname("ComputeArrivalTimeAndPower"));
  TArrivalTimeAndPower.stop();
}

void doStaticTimingAnalysis(CircuitGraph& graph) {
  computeArrivalTimeAndPower(graph.g, graph.dummySrc);
  computeRequiredTime(graph.g, graph.dummySink);
}

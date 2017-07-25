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
        data.slack = data.requiredTime - data.arrivalTime;
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
      // primary input
      else {
        for (auto oe: g.edges(n)) {
          auto requiredTime = g.getData(g.getEdgeDst(oe)).requiredTime;
          if (requiredTime < data.requiredTime) {
            data.requiredTime = requiredTime;
            data.slack = requiredTime - data.arrivalTime;
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
        auto requiredTime = g.getData(g.getEdgeDst(oe)).requiredTime;
        if (requiredTime < data.requiredTime) {
          data.requiredTime = requiredTime;
          data.slack = requiredTime - data.arrivalTime;
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
      for (auto e: g.edges(n)) {
        float requiredTime = g.getData(g.getEdgeDst(e)).requiredTime - g.getEdgeData(e).delay;
        if (requiredTime < data.requiredTime) {
          data.requiredTime = requiredTime;
          data.slack = requiredTime - data.arrivalTime;
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

static void computeRequiredTime(Graph& g) {
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
          data.slew = inData.slew;
          data.isRise = inData.isRise;
          data.arrivalTime = inData.arrivalTime;
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
        data.slew = inData.slew;
        data.isRise = inData.isRise;
        data.arrivalTime = inData.arrivalTime;
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
        auto pin = outData.pin;
        if (pin->gate) {
          data.totalPinC += pin->gate->cell->cellPins.at(pin->name)->capacitance;
        }
        else {
          // primary output, already recorded
          data.totalPinC += outData.totalPinC;
        }
      }

      auto outCellPin = data.pin->gate->cell->outPins.at(data.pin->name);
      data.arrivalTime = -std::numeric_limits<float>::infinity();
      for (auto ie: g.in_edges(n)) {
        auto& inData = g.getData(g.getEdgeDst(ie));
        auto pin = inData.pin;
        auto isInRise = inData.isRise;
        LUT *cellLUT = nullptr, *transitionLUT = nullptr, *powerLUT = nullptr; 
        auto tSense = outCellPin->tSense.at(pin->name);

        if ((TIMING_SENSE_POSITIVE_UNATE == tSense && isInRise) ||
            (TIMING_SENSE_NEGATIVE_UNATE == tSense && !isInRise)) {
          cellLUT = outCellPin->cellRise.at(pin->name);
          transitionLUT = outCellPin->riseTransition.at(pin->name);
          powerLUT = outCellPin->risePower.at(pin->name);
        }
        else if ((TIMING_SENSE_POSITIVE_UNATE == tSense && !isInRise) ||
                 (TIMING_SENSE_NEGATIVE_UNATE == tSense && isInRise)) {
          cellLUT = outCellPin->cellFall.at(pin->name);
          transitionLUT = outCellPin->fallTransition.at(pin->name);
          powerLUT = outCellPin->fallPower.at(pin->name);
        }

        float totalC = data.totalPinC + data.totalNetC;
        auto inSlew = inData.slew;
        std::vector<float> v = {inSlew, totalC};
        auto& ieData = g.getEdgeData(ie);
        ieData.delay = cellLUT->lookup(v);
        auto newArrivalTime = inData.arrivalTime + ieData.delay;

        if (data.arrivalTime < newArrivalTime) {
          // update critical path
          data.arrivalTime = newArrivalTime;
          data.isRise = (TIMING_SENSE_POSITIVE_UNATE == tSense) ? isInRise : !isInRise;
          data.slew = transitionLUT->lookup(v);

          // power follows critical path
          std::vector<float> vPinC = {inSlew, data.totalPinC};
          data.internalPower = powerLUT->lookup(vPinC);
          std::vector<float> vNetC = {inSlew, data.totalNetC};
          data.netPower = powerLUT->lookup(vNetC);
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

static void computeArrivalTimeAndPower(Graph& g) {
  Galois::StatTimer TArrivalTimeAndPower("ArrivalTimeAndPower");
  TArrivalTimeAndPower.start();
  Galois::do_all_local(g, SetForwardPrecondition{g}, Galois::do_all_steal<true>());
  Galois::InsertBag<GNode> work;
  work.push_back(dummySrc);
  Galois::for_each_local(work, ComputeArrivalTimeAndPower{g}, Galois::loopname("ComputeArrivalTimeAndPower"));
  TArrivalTimeAndPower.stop();
}

void doStaticTimingAnalysis(Graph& graph) {
  computeArrivalTimeAndPower(graph);
  computeRequiredTime(graph);
}

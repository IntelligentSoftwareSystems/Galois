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

    if (data.isPrimaryOutput) {
      data.slack = data.requiredTime - data.arrivalTime;
      for (auto ie: g.in_edges(n)) {
        ctx.push(g.getEdgeDst(ie));
      }
    }

    else if (data.isGateOutput) {
      bool changed = false;
      for (auto e: g.edges(n)) {
        float requiredTime = g.getData(g.getEdgeDst(e)).requiredTime;
        if (data.requiredTime > requiredTime) {
          data.requiredTime = requiredTime;
          data.slack = data.requiredTime - data.arrivalTime;
          changed = true;
        }
      }
      if (changed) {
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
    }

    else if (data.isGateInput) {
      bool changed = false;
      for (auto e: g.edges(n)) {
        float requiredTime = g.getData(g.getEdgeDst(e)).requiredTime - g.getEdgeData(e).delay;
        if (data.requiredTime > requiredTime) {
          data.requiredTime = requiredTime;
          data.slack = data.requiredTime - data.arrivalTime;
          changed = true;
        }
      }
      if (changed) {
        for (auto ie: g.in_edges(n)) {
          ctx.push(g.getEdgeDst(ie));
        }
      }
    }

    else if (data.isPrimaryInput) {
      for (auto e: g.edges(n)) {
        float requiredTime = g.getData(g.getEdgeDst(e)).requiredTime;
        if (data.requiredTime > requiredTime) {
          data.requiredTime = requiredTime;
          data.slack = data.requiredTime - data.arrivalTime;
        }
      }     
    }
  } // end operator()
}; // end struct ComputeRequiredTime

static void computeRequiredTime(Graph& g) {
  Galois::StatTimer TRequiredTime("RequiredTime");
  TRequiredTime.start();

  // enqueue all primary outputs
  Galois::InsertBag<GNode> work;
  for (auto ie: g.in_edges(dummySink)) {
    work.push_back(g.getEdgeDst(ie));
  }

  Galois::for_each_local(work, ComputeRequiredTime{g}, Galois::loopname("ComputeRequiredTime"));
  TRequiredTime.stop();
}

struct ComputeArrivalTimeAndPower {
  Graph& g;
  ComputeArrivalTimeAndPower(Graph& g): g(g) {}

  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
    auto& data = g.getData(n);

    if (!data.isGateOutput) {
      if (!data.isPrimaryInput) {
        for (auto ie: g.in_edges(n)) {
          auto& inData = g.getData(g.getEdgeDst(ie));
          data.slew = inData.slew;
          data.isRise = inData.isRise;
          data.arrivalTime = inData.arrivalTime;
        }     
      }
      if (!data.isPrimaryOutput) {
        for (auto oe: g.edges(n)) {
          ctx.push(g.getEdgeDst(oe));
        }
      }
    }

    // gate outputs
    else {
      data.totalPinC = 0.0;
      for (auto oe: g.edges(n)) {
        auto& oData = g.getData(g.getEdgeDst(oe));
        auto pin = oData.pin;
        if (pin->gate) {
          data.totalPinC += pin->gate->cell->cellPins.at(pin->name)->capacitance;
        }
        else {
          // primary output, already recorded
          data.totalPinC += oData.totalPinC;
        }
      }

      auto outCellPin = data.pin->gate->cell->outPins.at(data.pin->name);
      bool changed = false;
      for (auto ie: g.in_edges(n)) {
        auto& inData = g.getData(g.getEdgeDst(ie));
        auto inSlew = inData.slew;
        if (0.0 == inSlew) {
          // skip out-of-topological-order accesses
          continue;
        }

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
        std::vector<float> v = {inSlew, totalC};
//        std::cout << pin->gate->name << "." << pin->name << ": slew = " << inSlew << std::endl;
//        std::cout << data.pin->gate->name << "." << data.pin->name << ": C = " << totalC << std::endl;
        auto& ieData = g.getEdgeData(ie);
        ieData.delay = cellLUT->lookup(v);
        auto newArrivalTime = inData.arrivalTime + ieData.delay;

        if (data.arrivalTime < newArrivalTime) {
          // update critical path
          data.arrivalTime = newArrivalTime;
          data.isRise = (TIMING_SENSE_POSITIVE_UNATE == tSense) ? isInRise : !isInRise;
          data.slew = transitionLUT->lookup(v);
          changed = true;

          // power follows critical path
          std::vector<float> vPinC = {inSlew, data.totalPinC};
          data.internalPower = powerLUT->lookup(vPinC);
          std::vector<float> vNetC = {inSlew, data.totalNetC};
          data.netPower = powerLUT->lookup(vNetC);
        }
      } // end for ie

      if (changed) {
        for (auto oe: g.edges(n)) {
          ctx.push(g.getEdgeDst(oe));
        }
      }
    } // end else (data.isGateOutput)
  } // end operator()
}; // end struct ComputeArrivalTimeAndPower

static void computeArrivalTimeAndPower(Graph& g) {
  Galois::StatTimer TArrivalTimeAndPower("ArrivalTimeAndPower");
  TArrivalTimeAndPower.start();

  // enqueue all primary inputs
  Galois::InsertBag<GNode> work;
  for (auto e: g.edges(dummySrc)) {
    work.push_back(g.getEdgeDst(e));
  }

  Galois::for_each_local(work, ComputeArrivalTimeAndPower{g}, Galois::loopname("ComputeArrivalTimeAndPower"));
  TArrivalTimeAndPower.stop();
}

void doStaticTimingAnalysis(Graph& graph) {
  computeArrivalTimeAndPower(graph);
  computeRequiredTime(graph);
}

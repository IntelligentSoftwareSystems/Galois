#ifndef GALOIS_EDA_TIMING_ENGINE_H
#define GALOIS_EDA_TIMING_ENGINE_H

#include "CellLib.h"
#include "Verilog.h"
#include "Sdc.h"
#include "TimingGraph.h"
#include "TimingDefinition.h"

#include <vector>
#include <unordered_map>

struct TimingPathNode {
  VerilogPin* pin;
  bool isRise;
  MyFloat arrival;
  MyFloat required;
  MyFloat slack;
};

using TimingPath = std::vector<TimingPathNode>;

struct TimingEngine {
  VerilogDesign& v;
  std::vector<CellLib*>& libs;
  std::vector<TimingMode>& modes;
  // sdc files

  // slew merging
  //   by best/worst slew if false (default)
  //   by best/worst arrival if true
  bool isExactSlew;

  std::unordered_map<VerilogModule*, TimingGraph*> modules;

private:
  TimingGraph* findTimingGraph(VerilogModule* m);

public:
  TimingEngine(VerilogDesign& v, std::vector<CellLib*>& libs, std::vector<TimingMode>& modes, bool isExactSlew = false);
  ~TimingEngine();

  void update(VerilogModule* m); // for incremental timing analysis
  void constrain(VerilogModule* m, SDC& sdc);  // add constraints to the module
  void time(VerilogModule* m);  // timing analysis from scratch

  MyFloat reportArrivalTime(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner);
  MyFloat reportSlack(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner);
  std::vector<TimingPath> reportTopKCriticalPaths(VerilogModule* m, size_t corner, size_t k = 1);
};

#endif // GALOIS_EDA_TIMING_ENGINE_H

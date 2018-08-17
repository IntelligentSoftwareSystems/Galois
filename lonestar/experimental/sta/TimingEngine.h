#ifndef GALOIS_EDA_TIMING_ENGINE_H
#define GALOIS_EDA_TIMING_ENGINE_H

#include "CellLib.h"
#include "Verilog.h"
#include "TimingGraph.h"
#include "TimingMode.h"

#include <vector>
#include <unordered_map>

struct TimingPathNode {
  VerilogPin* pin;
  bool isRise;
  float arrival;
  float required;
  float slack;
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
  void time(VerilogModule* m);  // timing analysis from scratch

  float reportArrivalTime(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner);
  float reportSlack(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner);
  std::vector<TimingPath> reportTopKCriticalPaths(VerilogModule* m, size_t corner, size_t k = 1);
};

#endif // GALOIS_EDA_TIMING_ENGINE_H

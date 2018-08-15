#ifndef GALOIS_EDA_TIMING_ENGINE_H
#define GALOIS_EDA_TIMING_ENGINE_H

#include "CellLib.h"
#include "Verilog.h"
#include "TimingGraph.h"

#include <vector>
#include <unordered_map>

struct TimingEngine {
  VerilogDesign& v;
  std::vector<CellLib*>& libs;
  // sdc

  std::unordered_map<VerilogModule*, TimingGraph*> modules;

public:
  TimingEngine(VerilogDesign& v, std::vector<CellLib*>& libs);
  ~TimingEngine();

  void update(VerilogModule* m); // for incremental timing analysis
  void time(VerilogModule* m);  // timing analysis from scratch

  float reportArrivalTime(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner);
  float reportSlack(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner);
  void reportTopKCriticalPaths(VerilogModule* m, size_t corner, size_t k = 1);
};

#endif // GALOIS_EDA_TIMING_ENGINE_H

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
  std::vector<CellLib*> libs;
  std::vector<TimingMode> modes;
  using Corner = std::pair<CellLib*, TimingMode>;
  std::unordered_map<Corner, size_t, boost::hash<Corner> > cornerMap;
  size_t numCorners;
  VerilogDesign* v;

  // slew merging
  //   by best/worst slew if false (default)
  //   by best/worst arrival if true
  bool isExactSlew;

  // when true, use idealWireLoad for all wires
  // default to false
  bool isWireIdeal;

  std::unordered_map<VerilogModule*, TimingGraph*> modules;

private:
  TimingGraph* findTimingGraph(VerilogModule* m);
  void clearTimingGraphs();

public:
  TimingEngine(bool isExactSlew = false): numCorners(0), isExactSlew(isExactSlew), isWireIdeal(false) {}
  ~TimingEngine() { clearTimingGraphs(); }

  void useIdealWire(bool flag) { isWireIdeal = flag; }

  void addCellLib(CellLib* lib, TimingMode mode) {
    auto corner = std::make_pair(lib, mode);
    if (!cornerMap.count(corner)) {
      libs.push_back(lib);
      modes.push_back(mode);
      cornerMap[corner] = numCorners;
      numCorners += 1;
    }
  }

  void readDesign(VerilogDesign* design);
  void constrain(VerilogModule* m, SDC& sdc);  // add constraints to the module
  void time(VerilogModule* m, TimingPropAlgo algo);  // timing analysis from scratch

  MyFloat reportArrivalTime(VerilogModule* m, VerilogPin* p, bool isRise, CellLib* lib, TimingMode mode);
  MyFloat reportSlack(VerilogModule* m, VerilogPin* p, bool isRise, CellLib* lib, TimingMode mode);
  std::vector<TimingPath> reportTopKCriticalPaths(VerilogModule* m, CellLib* lib, TimingMode mode, size_t k = 1);
};

#endif // GALOIS_EDA_TIMING_ENGINE_H

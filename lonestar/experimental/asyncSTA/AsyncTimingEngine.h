#ifndef GALOIS_EDA_ASYNC_TIMING_ENGINE_H
#define GALOIS_EDA_ASYNC_TIMING_ENGINE_H

#include "TimingDefinition.h"
#include "CellLib.h"
#include "Verilog.h"
#include "AsyncTimingGraph.h"
#include "AsyncTimingArcSet.h"

#include <vector>
#include <unordered_map>

struct AsyncTimingEngine {
  std::vector<CellLib*> libs;
  std::vector<TimingMode> modes;
  using Corner = std::pair<CellLib*, TimingMode>;
  std::unordered_map<Corner, size_t, boost::hash<Corner> > cornerMap;
  size_t numCorners;
  VerilogDesign* v;

  // when true, use idealWireLoad for all wires
  // default to false
  bool isWireIdeal;

  std::unordered_map<VerilogModule*, AsyncTimingGraph*> modules;

private:
  AsyncTimingGraph* findAsyncTimingGraph(VerilogModule* m);
  void clearAsyncTimingGraphs();

public:
  AsyncTimingEngine(): numCorners(0), isWireIdeal(false) {}
  ~AsyncTimingEngine() { clearAsyncTimingGraphs(); }

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
  void time(VerilogModule* m);  // timing analysis from scratch
};

#endif // GALOIS_EDA_ASYNC_TIMING_ENGINE_H

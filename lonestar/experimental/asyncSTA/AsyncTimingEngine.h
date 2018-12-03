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
  std::unordered_map<CellLib*, size_t> cornerMap;
  size_t numCorners;
  VerilogDesign* v;
  AsyncTimingArcCollection* arcs;

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

  void addCellLib(CellLib* lib) {
    if (!cornerMap.count(lib)) {
      libs.push_back(lib);
      cornerMap[lib] = numCorners;
      numCorners += 1;
    }
  }

  void readDesign(VerilogDesign* design, AsyncTimingArcCollection* arcCollection);
  void time(VerilogModule* m);  // timing analysis from scratch
};

#endif // GALOIS_EDA_ASYNC_TIMING_ENGINE_H

#include "TimingEngine.h"

void TimingEngine::readDesign(VerilogDesign* design) {
  clearTimingGraphs();
  assert(numCorners); // need cell lib info for TimingGraphs
  assert(design);
  v = design;

  for (auto i: v->modules) {
    auto m = i.second;
    if (!m->isHierarchical()) {
      TimingGraph* g = new TimingGraph(*m, this);
      modules[m] = g;
      g->construct();
    }
    else {
      std::cout << "Not supported: module " << m->name << " is hierarchical" << std::endl;
    }
  }
}

void TimingEngine::clearTimingGraphs() {
  for (auto& i: modules) {
    delete i.second;
  }
  modules.clear();
}

TimingGraph* TimingEngine::findTimingGraph(VerilogModule* m) {
  auto it = modules.find(m);
  if (modules.find(m) != modules.end()) {
    return it->second;
  }
  else {
    std::cout << "Module " << m->name << " does not exist or is hierarchical." << std::endl;
    return nullptr;
  }
}

void TimingEngine::constrain(VerilogModule* m, SDC& sdc) {
  auto g = findTimingGraph(m);
  if (g) {
    g->initialize();
    g->setConstraints(sdc);
  }
}

void TimingEngine::time(VerilogModule* m) {
  auto g = findTimingGraph(m);
  if (g) {
    g->computeForward();
    g->computeBackward();
    g->print();
  }
}

MyFloat TimingEngine::reportArrivalTime(VerilogModule* m, VerilogPin* p, bool isRise, CellLib* lib, TimingMode mode) {
  return 0.0;
}

MyFloat TimingEngine::reportSlack(VerilogModule* m, VerilogPin* p, bool isRise, CellLib* lib, TimingMode mode) {
  return 0.0;
}

std::vector<TimingPath> TimingEngine::reportTopKCriticalPaths(VerilogModule* m, CellLib* lib, TimingMode mode, size_t k) {
  return {};
}

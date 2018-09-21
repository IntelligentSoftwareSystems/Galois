#include "TimingEngine.h"

TimingEngine::TimingEngine(VerilogDesign& v, bool isExactSlew)
  : v(v), isExactSlew(isExactSlew)
{
  for (auto i: v.modules) {
    auto m = i.second;
    if (m->isFlattened()) {
      TimingGraph* g = new TimingGraph(*m, libs, modes, isExactSlew);
      modules[m] = g;
      g->construct();
    }
    else {
      std::cout << "Not supported: module " << m->name << " is not flattened" << std::endl;
    }
  }
}

TimingEngine::~TimingEngine() {
  for (auto& i: modules) {
    delete i.second;
  }
}

TimingGraph* TimingEngine::findTimingGraph(VerilogModule* m) {
  auto it = modules.find(m);
  if (modules.find(m) != modules.end()) {
    return it->second;
  }
  else {
    std::cout << "Module " << m->name << " does not exist or is not flattened." << std::endl;
    return nullptr;
  }
}

void TimingEngine::update(VerilogModule* m) {

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
    g->computeArrivalTime();
  }
}

MyFloat TimingEngine::reportArrivalTime(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner) {
  return 0.0;
}

MyFloat TimingEngine::reportSlack(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner) {
  return 0.0;
}

std::vector<TimingPath> TimingEngine::reportTopKCriticalPaths(VerilogModule* m, size_t corner, size_t k) {
  return {};
}

#include "TimingEngine.h"

TimingEngine::TimingEngine(VerilogDesign& v, std::vector<CellLib*>& libs)
  : v(v), libs(libs)
{
  for (auto i: v.modules) {
    auto m = i.second;
    if (m->isFlattened()) {
      TimingGraph* g = new TimingGraph{*m, libs};
      modules[m] = g;
      g->construct();
      g->print();
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

void TimingEngine::update(VerilogModule* m) {

}

void TimingEngine::time(VerilogModule* m) {

}

float TimingEngine::reportArrivalTime(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner) {
  return 0.0;
}

float TimingEngine::reportSlack(VerilogModule* m, VerilogPin* p, bool isRise, size_t corner) {
  return 0.0;
}

void TimingEngine::reportTopKCriticalPaths(VerilogModule* m, size_t corner, size_t k) {

}

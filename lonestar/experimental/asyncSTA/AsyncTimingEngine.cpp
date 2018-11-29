#include "AsyncTimingEngine.h"
#include "galois/Timer.h"

void AsyncTimingEngine::readDesign(VerilogDesign* design, AsyncTimingArcCollection* arcCollection) {
  clearAsyncTimingGraphs();
  assert(numCorners); // need cell lib info for AsyncTimingGraphs
  assert(design);
  v = design;
  arcs = arcCollection;

  for (auto i: v->modules) {
    auto m = i.second;
    if (m->isHierarchical()) {
      std::cout << "Not supported: module " << m->name << " is hierarchical" << std::endl;
    }
    else if (arcs->findArcSetForModule(m)) {
      AsyncTimingGraph* g = new AsyncTimingGraph(*m, this);
      modules[m] = g;
      g->construct();
    }
    else {
      std::cout << "No timing arc set exist for module " << m->name << std::endl;
    }
  }
}

void AsyncTimingEngine::clearAsyncTimingGraphs() {
  for (auto& i: modules) {
    delete i.second;
  }
  modules.clear();
}

AsyncTimingGraph* AsyncTimingEngine::findAsyncTimingGraph(VerilogModule* m) {
  auto it = modules.find(m);
  if (modules.find(m) != modules.end()) {
    return it->second;
  }
  else {
    std::cout << "Module " << m->name << " does not exist or is hierarchical." << std::endl;
    return nullptr;
  }
}

void AsyncTimingEngine::time(VerilogModule* m) {
  auto g = findAsyncTimingGraph(m);
  if (g) {
    galois::StatTimer Tmain;
    Tmain.start();
    // compute slew
    // compute delay
    // compute maximum cycle ratio
    Tmain.stop();

//    g->print();
  }
}

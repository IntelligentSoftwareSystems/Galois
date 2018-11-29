#include "CellLib.h"
#include "Verilog.h"
#include "AsyncTimingEngine.h"

#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <vector>

static const char* name = "async_sta";
static const char* url = nullptr;
static const char* desc = "Asynchronous Static Timing Analysis";

namespace cll = llvm::cl;
static cll::opt<std::string>
    libName("lib", cll::desc("path to .lib (Liberty file)"), cll::Required);
static cll::opt<std::string>
    verilogName(cll::Positional, cll::desc("path to .v (Verilog file)"), cll::Required);

int main(int argc, char *argv[]) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  CellLib lib;
  lib.parse(libName);
//  lib.print();
  std::cout << "Parsed cell library " << name << std::endl;

  VerilogDesign design;
  design.parse(verilogName);
  std::cout << "Parsed " << design.modules.size() << " Verilog module(s) in " << verilogName << std::endl;
//  design.print();
  design.buildDependency();
//  std::cout << "design is " << (design.isHierarchical() ? "" : "not ") << "hierarchical." << std::endl;
//  std::cout << "design has " << design.roots.size() << " top-level module(s)." << std::endl;
  if (design.isHierarchical() || (design.roots.size() > 1)) {
    std::cout << "Abort: Not supporting multiple/hierarchical modules for now." << std::endl;
    return 0;
  }

  auto m = *(design.roots.begin()); // top-level module

  AsyncTimingArcCollection arcs(design);
  auto arcSet = arcs.findArcSetForModule(m);
  // add required timing arcs and ticked timing arcs here

  AsyncTimingEngine engine;
  engine.useIdealWire(true);
  engine.addCellLib(&lib, MAX_DELAY_MODE);
  engine.readDesign(&design, &arcs);

  engine.time(m);
  std::cout << "Timed Verilog module " << m->name;
  std::cout << ": " << m->numPorts() << " ports";
  std::cout << ", " << m->numGates() << " gates";
  std::cout << ", " << m->numInternalPins() << " internal pins";
  std::cout << ", " << m->numWires() << " nets";
  std::cout << std::endl;

  return 0;
}

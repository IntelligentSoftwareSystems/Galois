#include "CellLib.h"
#include "Verilog.h"
#include "Sdc.h"
#include "TimingEngine.h"

#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <vector>

static const char* name = "sta";
static const char* url = nullptr;
static const char* desc = "Static Timing Analysis";

namespace cll = llvm::cl;
static cll::opt<std::string>
    cellLibName("lib", cll::desc("path to .lib (cell library)"), cll::Required);
static cll::opt<std::string>
    verilogName(cll::Positional, cll::desc("path to .v (Verilog file)"), cll::Required);
static cll::opt<std::string>
    sdcName("sdc", cll::desc("path to .sdc (Synopsis design constraints)"));

int main(int argc, char *argv[]) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  CellLib lib;
  lib.parse(cellLibName);
//  lib.print();

#if 0
  auto wireLoad = lib.defaultWireLoad;
  std::cout << "default wire delay (drive c=0.0, deg=2) = " << wireLoad->wireDelay(0.0, 2) << std::endl;
  std::cout << "default wire delay (drive c=0.0, deg=10) = " << wireLoad->wireDelay(0.0, 10) << std::endl;
  std::cout << "default wire delay (drive c=0.0, deg=12) = " << wireLoad->wireDelay(0.0, 12) << std::endl;

  auto invX1 = lib.findCell("INV_X1");
  auto outPin = invX1->findCellPin("ZN");
  auto inPin = invX1->findCellPin("A");
  std::vector<MyFloat> v = {0.0, 4.0};
  auto res = outPin->extractMax(v, TABLE_DELAY, inPin, true, true);
  std::cout << "invX1.riseDelay(slew=0.0, drive c=4.0) = " << res.first << std::endl;
#endif

  VerilogDesign design;
  design.parse(verilogName);
//  design.print();
  design.buildHierarchy();
//  std::cout << "design is " << (design.isFlattened() ? "" : "not ") << "flattened." << std::endl;
//  std::cout << "design has " << design.roots.size() << " top-level module(s)." << std::endl;
  if (!design.isFlattened() || (design.roots.size() > 1)) {
    std::cout << "Abort: Not supporting multiple/hierarchical modules for now." << std::endl;
    return 0;
  }

  std::vector<CellLib*> libs;
  libs.insert(libs.begin(), 1, &lib);
//  libs.insert(libs.begin(), 2, &lib);

  auto m = *(design.roots.begin());

  SDC sdc(libs, *m);
  sdc.parse(sdcName);
//  sdc.print();

#if 1
//  std::vector<TimingMode> modes = {TIMING_MODE_MAX_DELAY, TIMING_MODE_MIN_DELAY};
  std::vector<TimingMode> modes = {TIMING_MODE_MAX_DELAY};
  TimingEngine engine(design, libs, modes);
  engine.constrain(m, sdc);
  engine.time(m);
#endif

  return 0;
}

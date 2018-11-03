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
  lib.print();

#if 1
  auto and2X1 = lib.findCell("AND2_X1");
  auto pinZN = and2X1->findCellPin("ZN");
  auto pinA1 = and2X1->findCellPin("A1");
  auto pinA2 = and2X1->findCellPin("A2");
  std::cout << "AND2_X1: (A1-f, ZN-f) is " << pinZN->isEdgeDefined(pinA1, false, false) << std::endl;
  std::cout << "AND2_X1: (A1-f, ZN-r) is " << pinZN->isEdgeDefined(pinA1, false, true) << std::endl;
  std::cout << "AND2_X1: (A1-r, ZN-f) is " << pinZN->isEdgeDefined(pinA1, true, false) << std::endl;
  std::cout << "AND2_X1: (A1-r, ZN-r) is " << pinZN->isEdgeDefined(pinA1, true, true) << std::endl;
  std::cout << "AND2_X1: (A2-f, ZN-f) is " << pinZN->isEdgeDefined(pinA2, false, false) << std::endl;
  std::cout << "AND2_X1: (A2-f, ZN-r) is " << pinZN->isEdgeDefined(pinA2, false, true) << std::endl;
  std::cout << "AND2_X1: (A2-r, ZN-f) is " << pinZN->isEdgeDefined(pinA2, true, false) << std::endl;
  std::cout << "AND2_X1: (A2-r, ZN-r) is " << pinZN->isEdgeDefined(pinA2, true, true) << std::endl;

  auto nand2X1 = lib.findCell("NAND2_X1");
  pinZN = nand2X1->findCellPin("ZN");
  pinA1 = nand2X1->findCellPin("A1");
  pinA2 = nand2X1->findCellPin("A2");
  std::cout << "NAND2_X1: (A1-f, ZN-f) is " << pinZN->isEdgeDefined(pinA1, false, false) << std::endl;
  std::cout << "NAND2_X1: (A1-f, ZN-r) is " << pinZN->isEdgeDefined(pinA1, false, true) << std::endl;
  std::cout << "NAND2_X1: (A1-r, ZN-f) is " << pinZN->isEdgeDefined(pinA1, true, false) << std::endl;
  std::cout << "NAND2_X1: (A1-r, ZN-r) is " << pinZN->isEdgeDefined(pinA1, true, true) << std::endl;
  std::cout << "NAND2_X1: (A2-f, ZN-f) is " << pinZN->isEdgeDefined(pinA2, false, false) << std::endl;
  std::cout << "NAND2_X1: (A2-f, ZN-r) is " << pinZN->isEdgeDefined(pinA2, false, true) << std::endl;
  std::cout << "NAND2_X1: (A2-r, ZN-f) is " << pinZN->isEdgeDefined(pinA2, true, false) << std::endl;
  std::cout << "NAND2_X1: (A2-r, ZN-r) is " << pinZN->isEdgeDefined(pinA2, true, true) << std::endl;

  auto xor2X1 = lib.findCell("XOR2_X1");
  auto pinZ = xor2X1->findCellPin("Z");
  auto pinA = xor2X1->findCellPin("A");
  auto pinB = xor2X1->findCellPin("B");
  std::cout << "XOR2_X1: (A-f, Z-f) is " << pinZ->isEdgeDefined(pinA, false, false) << std::endl;
  std::cout << "XOR2_X1: (A-f, Z-r) is " << pinZ->isEdgeDefined(pinA, false, true) << std::endl;
  std::cout << "XOR2_X1: (A-r, Z-f) is " << pinZ->isEdgeDefined(pinA, true, false) << std::endl;
  std::cout << "XOR2_X1: (A-r, Z-r) is " << pinZ->isEdgeDefined(pinA, true, true) << std::endl;
  std::cout << "XOR2_X1: (B-f, Z-f) is " << pinZ->isEdgeDefined(pinB, false, false) << std::endl;
  std::cout << "XOR2_X1: (B-f, Z-r) is " << pinZ->isEdgeDefined(pinB, false, true) << std::endl;
  std::cout << "XOR2_X1: (B-r, Z-f) is " << pinZ->isEdgeDefined(pinB, true, false) << std::endl;
  std::cout << "XOR2_X1: (B-r, Z-r) is " << pinZ->isEdgeDefined(pinB, true, true) << std::endl;

  VerilogWire* wire = new VerilogWire;
  std::vector<VerilogPin*> pins;
  for (size_t i = 0; i < 12; ++i) {
    pins.push_back(new VerilogPin);
  }

  auto wireLoad = lib.defaultWireLoad;

  for (size_t i = 0; i < 2; ++i) {
    wire->addPin(pins[i]);
  }
  std::cout << "default wire delay (drive c=0.0, deg=2) = " << wireLoad->wireDelay(0.0, wire, nullptr) << std::endl;

  for (size_t i = 2; i < 10; ++i) {
    wire->addPin(pins[i]);
  }
  std::cout << "default wire delay (drive c=0.0, deg=10) = " << wireLoad->wireDelay(0.0, wire, nullptr) << std::endl;

  for (size_t i = 10; i < 12; ++i) {
    wire->addPin(pins[i]);
  }
  std::cout << "default wire delay (drive c=0.0, deg=12) = " << wireLoad->wireDelay(0.0, wire, nullptr) << std::endl;

  for (auto& p: pins) {
    delete p;
  }
  delete wire;

  auto invX1 = lib.findCell("INV_X1");
  auto outPin = invX1->findCellPin("ZN");
  auto inPin = invX1->findCellPin("A");
  Parameter param = {
    {INPUT_NET_TRANSITION,         0.0},
    {TOTAL_OUTPUT_NET_CAPACITANCE, 4.0}
  };
  auto res = outPin->extractMax(param, DELAY, inPin, false, true);
  std::cout << "invX1.riseDelay(slew=0.0, drive c=4.0) = " << res.first << std::endl;
#endif
/*
  VerilogDesign design;
  design.parse(verilogName);
//  design.print();
  design.buildDependency();
//  std::cout << "design is " << (design.isHierarchical() ? "" : "not ") << "hierarchical." << std::endl;
//  std::cout << "design has " << design.roots.size() << " top-level module(s)." << std::endl;
  if (design.isHierarchical() || (design.roots.size() > 1)) {
    std::cout << "Abort: Not supporting multiple/hierarchical modules for now." << std::endl;
    return 0;
  }

  TimingEngine engine;
  engine.addCellLib(&lib, TIMING_MODE_MAX_DELAY);
//  engine.addCellLib(&lib, TIMING_MODE_MIN_DELAY);
  engine.readDesign(&design);

  auto m = *(design.roots.begin());

  SDC sdc(engine.libs, *m);
  sdc.parse(sdcName);
//  sdc.print();

  engine.constrain(m, sdc);
  engine.time(m);
*/
  return 0;
}

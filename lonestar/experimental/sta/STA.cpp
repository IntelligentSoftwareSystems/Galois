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
    fastLibName("libFast", cll::desc("path to .lib for the fast corner"));
static cll::opt<std::string>
    slowLibName("libSlow", cll::desc("path to .lib for the slow corner"));
static cll::opt<std::string>
    verilogName(cll::Positional, cll::desc("path to .v (Verilog file)"), cll::Required);
static cll::opt<std::string>
    sdcName("sdc", cll::desc("path to .sdc (Synopsys design constraints)"), cll::Required);
static cll::opt<TimingPropAlgo>
    algo("algo", cll::desc("Choose an algorithm:"),
         cll::values(clEnumVal(TopoBarrier, "TopoBarrier"),
                     clEnumVal(ByDependency, "ByDependency"),
                     clEnumVal(Unordered, "Unordered"),
                     clEnumVal(TopoSoftPriority, "TopoSoftPriority"),
                     clEnumValEnd),
         cll::init(TopoBarrier));

int main(int argc, char *argv[]) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  CellLib fLib;
  CellLib sLib;
  CellLib* fastLib = nullptr;
  CellLib* slowLib = nullptr;

  size_t libSpecified = 0x0;
  libSpecified |= !fastLibName.empty() << 0x1;
  libSpecified |= !slowLibName.empty();

  auto parseAndSetLib =
      [&] (CellLib& lib, CellLib** libPtr, std::string name) {
        lib.parse(name);
        *libPtr = &lib;
        std::cout << "Parsed cell library " << name << std::endl;
      };

  switch (libSpecified) {
  // fastLib and slowLib
  case 0x3:
    parseAndSetLib(fLib, &fastLib, fastLibName);
    if (fastLibName == slowLibName) {
      slowLib = &fLib;
    }
    else {
      parseAndSetLib(sLib, &slowLib, slowLibName);
    }
    break;
  // fastLib only
  case 0x2:
    parseAndSetLib(fLib, &fastLib, fastLibName);
    break;
  // slowLib only
  case 0x1:
    parseAndSetLib(sLib, &slowLib, slowLibName);
    break;
  // no libs are specified
  case 0x0:
  default:
    std::cout << "Need to specify at least one .lib file via -libSlow or -libFast." << std::endl;
    return 0;
  }

#if 0
  if (fastLib) {
    fLib.print();
  }
  if (slowLib && slowLib != fastLib) {
    sLib.print();
  }
#endif

#if 0
  // cell lib test
  // assume the input is NanGate 45nm NLDM .lib at typical corner for fLib
  auto and2X1 = fLib.findCell("AND2_X1");
  auto pinZN = and2X1->findCellPin("ZN");
  auto pinA1 = and2X1->findCellPin("A1");
  auto pinA2 = and2X1->findCellPin("A2");
  std::cout << "AND2_X1: (A1-f, ZN-f) is " << pinZN->isEdgeDefined(pinA1, false, false) << std::endl; // true
  std::cout << "AND2_X1: (A1-f, ZN-r) is " << pinZN->isEdgeDefined(pinA1, false, true) << std::endl;
  std::cout << "AND2_X1: (A1-r, ZN-f) is " << pinZN->isEdgeDefined(pinA1, true, false) << std::endl;
  std::cout << "AND2_X1: (A1-r, ZN-r) is " << pinZN->isEdgeDefined(pinA1, true, true) << std::endl;   // true
  std::cout << "AND2_X1: (A2-f, ZN-f) is " << pinZN->isEdgeDefined(pinA2, false, false) << std::endl; // true
  std::cout << "AND2_X1: (A2-f, ZN-r) is " << pinZN->isEdgeDefined(pinA2, false, true) << std::endl;
  std::cout << "AND2_X1: (A2-r, ZN-f) is " << pinZN->isEdgeDefined(pinA2, true, false) << std::endl;
  std::cout << "AND2_X1: (A2-r, ZN-r) is " << pinZN->isEdgeDefined(pinA2, true, true) << std::endl;   // true

  auto nand2X1 = fLib.findCell("NAND2_X1");
  pinZN = nand2X1->findCellPin("ZN");
  pinA1 = nand2X1->findCellPin("A1");
  pinA2 = nand2X1->findCellPin("A2");
  std::cout << "NAND2_X1: (A1-f, ZN-f) is " << pinZN->isEdgeDefined(pinA1, false, false) << std::endl;
  std::cout << "NAND2_X1: (A1-f, ZN-r) is " << pinZN->isEdgeDefined(pinA1, false, true) << std::endl;  // true
  std::cout << "NAND2_X1: (A1-r, ZN-f) is " << pinZN->isEdgeDefined(pinA1, true, false) << std::endl;  // true
  std::cout << "NAND2_X1: (A1-r, ZN-r) is " << pinZN->isEdgeDefined(pinA1, true, true) << std::endl;
  std::cout << "NAND2_X1: (A2-f, ZN-f) is " << pinZN->isEdgeDefined(pinA2, false, false) << std::endl;
  std::cout << "NAND2_X1: (A2-f, ZN-r) is " << pinZN->isEdgeDefined(pinA2, false, true) << std::endl;  // true
  std::cout << "NAND2_X1: (A2-r, ZN-f) is " << pinZN->isEdgeDefined(pinA2, true, false) << std::endl;  // true
  std::cout << "NAND2_X1: (A2-r, ZN-r) is " << pinZN->isEdgeDefined(pinA2, true, true) << std::endl;

  auto xor2X1 = fLib.findCell("XOR2_X1");
  auto pinZ = xor2X1->findCellPin("Z");
  auto pinA = xor2X1->findCellPin("A");
  auto pinB = xor2X1->findCellPin("B");
  std::cout << "XOR2_X1: (A-f, Z-f) is " << pinZ->isEdgeDefined(pinA, false, false) << std::endl; // true
  std::cout << "XOR2_X1: (A-f, Z-r) is " << pinZ->isEdgeDefined(pinA, false, true) << std::endl;  // true
  std::cout << "XOR2_X1: (A-r, Z-f) is " << pinZ->isEdgeDefined(pinA, true, false) << std::endl;  // true
  std::cout << "XOR2_X1: (A-r, Z-r) is " << pinZ->isEdgeDefined(pinA, true, true) << std::endl;   // true
  std::cout << "XOR2_X1: (B-f, Z-f) is " << pinZ->isEdgeDefined(pinB, false, false) << std::endl; // true
  std::cout << "XOR2_X1: (B-f, Z-r) is " << pinZ->isEdgeDefined(pinB, false, true) << std::endl;  // true
  std::cout << "XOR2_X1: (B-r, Z-f) is " << pinZ->isEdgeDefined(pinB, true, false) << std::endl;  // true
  std::cout << "XOR2_X1: (B-r, Z-r) is " << pinZ->isEdgeDefined(pinB, true, true) << std::endl;   // true

  auto dffrsX1 = fLib.findCell("DFFRS_X1");
  auto pinD = dffrsX1->findCellPin("D");
  auto pinQ = dffrsX1->findCellPin("Q");
  auto pinQN = dffrsX1->findCellPin("QN");
  auto pinCK = dffrsX1->findCellPin("CK");
  auto pinSN = dffrsX1->findCellPin("SN");
  auto pinRN = dffrsX1->findCellPin("RN");
  std::cout << "DFFRS_X1: (CK-f, D-f, delay) is " << pinD->isEdgeDefined(pinCK, false, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, D-r, delay) is " << pinD->isEdgeDefined(pinCK, false, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, D-f, delay) is " << pinD->isEdgeDefined(pinCK, true, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, D-r, delay) is " << pinD->isEdgeDefined(pinCK, true, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, D-f, min_constraint) is " << pinD->isEdgeDefined(pinCK, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, D-r, min_constraint) is " << pinD->isEdgeDefined(pinCK, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, D-f, min_constraint) is " << pinD->isEdgeDefined(pinCK, true, false, MIN_CONSTRAINT) << std::endl;  // true
  std::cout << "DFFRS_X1: (CK-r, D-r, min_constraint) is " << pinD->isEdgeDefined(pinCK, true, true, MIN_CONSTRAINT) << std::endl;   // true
  std::cout << "DFFRS_X1: (CK-f, D-f, max_constraint) is " << pinD->isEdgeDefined(pinCK, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, D-r, max_constraint) is " << pinD->isEdgeDefined(pinCK, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, D-f, max_constraint) is " << pinD->isEdgeDefined(pinCK, true, false, MAX_CONSTRAINT) << std::endl;  // true
  std::cout << "DFFRS_X1: (CK-r, D-r, max_constraint) is " << pinD->isEdgeDefined(pinCK, true, true, MAX_CONSTRAINT) << std::endl;   // true
  std::cout << "DFFRS_X1: (CK-f, RN-f, delay) is " << pinRN->isEdgeDefined(pinCK, false, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, RN-r, delay) is " << pinRN->isEdgeDefined(pinCK, false, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, RN-f, delay) is " << pinRN->isEdgeDefined(pinCK, true, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, RN-r, delay) is " << pinRN->isEdgeDefined(pinCK, true, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, RN-f, min_constraint) is " << pinRN->isEdgeDefined(pinCK, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, RN-r, min_constraint) is " << pinRN->isEdgeDefined(pinCK, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, RN-f, min_constraint) is " << pinRN->isEdgeDefined(pinCK, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, RN-r, min_constraint) is " << pinRN->isEdgeDefined(pinCK, true, true, MIN_CONSTRAINT) << std::endl;   // true
  std::cout << "DFFRS_X1: (CK-f, RN-f, max_constraint) is " << pinRN->isEdgeDefined(pinCK, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, RN-r, max_constraint) is " << pinRN->isEdgeDefined(pinCK, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, RN-f, max_constraint) is " << pinRN->isEdgeDefined(pinCK, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, RN-r, max_constraint) is " << pinRN->isEdgeDefined(pinCK, true, true, MAX_CONSTRAINT) << std::endl;   // true
  std::cout << "DFFRS_X1: (CK-f, SN-f, delay) is " << pinSN->isEdgeDefined(pinCK, false, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, SN-r, delay) is " << pinSN->isEdgeDefined(pinCK, false, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, SN-f, delay) is " << pinSN->isEdgeDefined(pinCK, true, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, SN-r, delay) is " << pinSN->isEdgeDefined(pinCK, true, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, SN-f, min_constraint) is " << pinSN->isEdgeDefined(pinCK, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, SN-r, min_constraint) is " << pinSN->isEdgeDefined(pinCK, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, SN-f, min_constraint) is " << pinSN->isEdgeDefined(pinCK, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, SN-r, min_constraint) is " << pinSN->isEdgeDefined(pinCK, true, true, MIN_CONSTRAINT) << std::endl;   // true
  std::cout << "DFFRS_X1: (CK-f, SN-f, max_constraint) is " << pinSN->isEdgeDefined(pinCK, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, SN-r, max_constraint) is " << pinSN->isEdgeDefined(pinCK, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, SN-f, max_constraint) is " << pinSN->isEdgeDefined(pinCK, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, SN-r, max_constraint) is " << pinSN->isEdgeDefined(pinCK, true, true, MAX_CONSTRAINT) << std::endl;   // true
  std::cout << "DFFRS_X1: (CK-f, Q-f, delay) is " << pinQ->isEdgeDefined(pinCK, false, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, Q-r, delay) is " << pinQ->isEdgeDefined(pinCK, false, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, Q-f, delay) is " << pinQ->isEdgeDefined(pinCK, true, false) << std::endl; // true
  std::cout << "DFFRS_X1: (CK-r, Q-r, delay) is " << pinQ->isEdgeDefined(pinCK, true, true) << std::endl;  // true
  std::cout << "DFFRS_X1: (CK-f, Q-f, min_constraint) is " << pinQ->isEdgeDefined(pinCK, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, Q-r, min_constraint) is " << pinQ->isEdgeDefined(pinCK, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, Q-f, min_constraint) is " << pinQ->isEdgeDefined(pinCK, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, Q-r, min_constraint) is " << pinQ->isEdgeDefined(pinCK, true, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, Q-f, max_constraint) is " << pinQ->isEdgeDefined(pinCK, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, Q-r, max_constraint) is " << pinQ->isEdgeDefined(pinCK, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, Q-f, max_constraint) is " << pinQ->isEdgeDefined(pinCK, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, Q-r, max_constraint) is " << pinQ->isEdgeDefined(pinCK, true, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, Q-f, delay) is " << pinQ->isEdgeDefined(pinRN, false, false) << std::endl; // true
  std::cout << "DFFRS_X1: (RN-f, Q-r, delay) is " << pinQ->isEdgeDefined(pinRN, false, true) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, Q-f, delay) is " << pinQ->isEdgeDefined(pinRN, true, false) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, Q-r, delay) is " << pinQ->isEdgeDefined(pinRN, true, true) << std::endl;   // true
  std::cout << "DFFRS_X1: (RN-f, Q-f, min_constraint) is " << pinQ->isEdgeDefined(pinRN, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, Q-r, min_constraint) is " << pinQ->isEdgeDefined(pinRN, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, Q-f, min_constraint) is " << pinQ->isEdgeDefined(pinRN, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, Q-r, min_constraint) is " << pinQ->isEdgeDefined(pinRN, true, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, Q-f, max_constraint) is " << pinQ->isEdgeDefined(pinRN, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, Q-r, max_constraint) is " << pinQ->isEdgeDefined(pinRN, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, Q-f, max_constraint) is " << pinQ->isEdgeDefined(pinRN, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, Q-r, max_constraint) is " << pinQ->isEdgeDefined(pinRN, true, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, Q-f, delay) is " << pinQ->isEdgeDefined(pinSN, false, false) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, Q-r, delay) is " << pinQ->isEdgeDefined(pinSN, false, true) << std::endl;  // true
  std::cout << "DFFRS_X1: (SN-r, Q-f, delay) is " << pinQ->isEdgeDefined(pinSN, true, false) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, Q-r, delay) is " << pinQ->isEdgeDefined(pinSN, true, true) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, Q-f, min_constraint) is " << pinQ->isEdgeDefined(pinSN, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, Q-r, min_constraint) is " << pinQ->isEdgeDefined(pinSN, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, Q-f, min_constraint) is " << pinQ->isEdgeDefined(pinSN, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, Q-r, min_constraint) is " << pinQ->isEdgeDefined(pinSN, true, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, Q-f, max_constraint) is " << pinQ->isEdgeDefined(pinSN, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, Q-r, max_constraint) is " << pinQ->isEdgeDefined(pinSN, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, Q-f, max_constraint) is " << pinQ->isEdgeDefined(pinSN, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, Q-r, max_constraint) is " << pinQ->isEdgeDefined(pinSN, true, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, QN-f, delay) is " << pinQN->isEdgeDefined(pinCK, false, false) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, QN-r, delay) is " << pinQN->isEdgeDefined(pinCK, false, true) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, QN-f, delay) is " << pinQN->isEdgeDefined(pinCK, true, false) << std::endl; // true
  std::cout << "DFFRS_X1: (CK-r, QN-r, delay) is " << pinQN->isEdgeDefined(pinCK, true, true) << std::endl;  // true
  std::cout << "DFFRS_X1: (CK-f, QN-f, min_constraint) is " << pinQN->isEdgeDefined(pinCK, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, QN-r, min_constraint) is " << pinQN->isEdgeDefined(pinCK, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, QN-f, min_constraint) is " << pinQN->isEdgeDefined(pinCK, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, QN-r, min_constraint) is " << pinQN->isEdgeDefined(pinCK, true, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, QN-f, max_constraint) is " << pinQN->isEdgeDefined(pinCK, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-f, QN-r, max_constraint) is " << pinQN->isEdgeDefined(pinCK, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, QN-f, max_constraint) is " << pinQN->isEdgeDefined(pinCK, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (CK-r, QN-r, max_constraint) is " << pinQN->isEdgeDefined(pinCK, true, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, QN-f, delay) is " << pinQN->isEdgeDefined(pinRN, false, false) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, QN-r, delay) is " << pinQN->isEdgeDefined(pinRN, false, true) << std::endl;  // true
  std::cout << "DFFRS_X1: (RN-r, QN-f, delay) is " << pinQN->isEdgeDefined(pinRN, true, false) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, QN-r, delay) is " << pinQN->isEdgeDefined(pinRN, true, true) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, QN-f, min_constraint) is " << pinQN->isEdgeDefined(pinRN, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, QN-r, min_constraint) is " << pinQN->isEdgeDefined(pinRN, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, QN-f, min_constraint) is " << pinQN->isEdgeDefined(pinRN, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, QN-r, min_constraint) is " << pinQN->isEdgeDefined(pinRN, true, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, QN-f, max_constraint) is " << pinQN->isEdgeDefined(pinRN, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-f, QN-r, max_constraint) is " << pinQN->isEdgeDefined(pinRN, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, QN-f, max_constraint) is " << pinQN->isEdgeDefined(pinRN, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (RN-r, QN-r, max_constraint) is " << pinQN->isEdgeDefined(pinRN, true, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, QN-f, delay) is " << pinQN->isEdgeDefined(pinSN, false, false) << std::endl; // true
  std::cout << "DFFRS_X1: (SN-f, QN-r, delay) is " << pinQN->isEdgeDefined(pinSN, false, true) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, QN-f, delay) is " << pinQN->isEdgeDefined(pinSN, true, false) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, QN-r, delay) is " << pinQN->isEdgeDefined(pinSN, true, true) << std::endl;   // true
  std::cout << "DFFRS_X1: (SN-f, QN-f, min_constraint) is " << pinQN->isEdgeDefined(pinSN, false, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, QN-r, min_constraint) is " << pinQN->isEdgeDefined(pinSN, false, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, QN-f, min_constraint) is " << pinQN->isEdgeDefined(pinSN, true, false, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, QN-r, min_constraint) is " << pinQN->isEdgeDefined(pinSN, true, true, MIN_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, QN-f, max_constraint) is " << pinQN->isEdgeDefined(pinSN, false, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-f, QN-r, max_constraint) is " << pinQN->isEdgeDefined(pinSN, false, true, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, QN-f, max_constraint) is " << pinQN->isEdgeDefined(pinSN, true, false, MAX_CONSTRAINT) << std::endl;
  std::cout << "DFFRS_X1: (SN-r, QN-r, max_constraint) is " << pinQN->isEdgeDefined(pinSN, true, true, MAX_CONSTRAINT) << std::endl;

  VerilogWire* wire = new VerilogWire;
  std::vector<VerilogPin*> pins;
  for (size_t i = 0; i < 12; ++i) {
    pins.push_back(new VerilogPin);
  }

  auto wireLoad = fLib.defaultWireLoad;

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

  auto invX1 = fLib.findCell("INV_X1");
  auto outPin = invX1->findCellPin("ZN");
  auto inPin = invX1->findCellPin("A");
  Parameter param = {
    {INPUT_NET_TRANSITION,         0.0},
    {TOTAL_OUTPUT_NET_CAPACITANCE, 4.0}
  };
  auto res = outPin->extractMax(param, DELAY, inPin, false, true);
  std::cout << "invX1.riseDelay(slew=0.0, drive c=4.0) = " << res.first << std::endl;
#endif

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

  auto m = *(design.roots.begin());
  SDC sdc(*m);
  sdc.parse(sdcName);
//  sdc.print();
  std::cout << "Parsed SDC commands in " << sdcName << std::endl;

  TimingEngine engine;
  engine.useIdealWire(true);
  if (slowLib) {
    engine.addCellLib(slowLib, MAX_DELAY_MODE);
  }
  if (fastLib) {
    engine.addCellLib(fastLib, MIN_DELAY_MODE);
  }
  engine.readDesign(&design);
  engine.constrain(m, sdc);
  engine.time(m, algo);
  std::cout << "Timed Verilog module " << m->name;
  std::cout << ": " << m->numPorts() << " ports";
  std::cout << ", " << m->numGates() << " gates";
  std::cout << ", " << m->numInternalPins() << " internal pins";
  std::cout << ", " << m->numWires() << " nets";
  std::cout << std::endl;

  return 0;
}

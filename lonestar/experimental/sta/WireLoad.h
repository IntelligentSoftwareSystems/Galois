#ifndef GALOIS_EDA_WIRELOAD_H
#define GALOIS_EDA_WIRELOAD_H

#include <string>
#include <iostream>

#include "Verilog.h"
#include "TimingDefinition.h"

// abstract class for the interface of wire load models
struct WireLoad {
  std::string name;

public:
  virtual MyFloat wireC(VerilogWire* wire) = 0;
  virtual MyFloat wireDelay(MyFloat loadC, VerilogWire* wire, VerilogPin* vPin) = 0;
  virtual void print(std::ostream& os = std::cout) = 0;

  // for derived classes
  virtual ~WireLoad() {}
};

// used for ideal wire. do not instantiate this class in user code.
struct IdealWireLoad: public WireLoad {
  MyFloat wireC(VerilogWire*) { return 0.0; }
  MyFloat wireDelay(MyFloat, VerilogWire*, VerilogPin*) { return 0.0; }
  void print(std::ostream& os = std::cout) { }
};

extern WireLoad* idealWireLoad;

#endif // GALOIS_EDA_WIRELOAD_H

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

#endif // GALOIS_EDA_WIRELOAD_H

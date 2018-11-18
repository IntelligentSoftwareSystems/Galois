#ifndef GALOIS_EDA_WIRELOAD_H
#define GALOIS_EDA_WIRELOAD_H

#include <string>
#include <iostream>
#include <boost/functional/hash.hpp>

#include "Verilog.h"
#include "TimingDefinition.h"

// abstract class for the interface of wire load models
struct WireLoad {
  std::string name;

public:
  virtual MyFloat wireC(VerilogWire* wire) = 0;
  virtual MyFloat wireDelay(MyFloat loadC, VerilogWire* wire, VerilogPin* vPin) = 0;
  virtual void setWireC(VerilogWire* wire, MyFloat value) = 0;
  virtual void setWireDelay(VerilogWire* wire, VerilogPin* vPin, MyFloat) = 0;
  virtual void print(std::ostream& os = std::cout) = 0;

  // for derived classes
  virtual ~WireLoad() {}
};

// used for ideal wire. do not instantiate this class in user code.
struct IdealWireLoad: public WireLoad {
  MyFloat wireC(VerilogWire*) { return 0.0; }
  MyFloat wireDelay(MyFloat, VerilogWire*, VerilogPin*) { return 0.0; }
  void setWireC(VerilogWire*, MyFloat) { }
  void setWireDelay(VerilogWire*, VerilogPin*, MyFloat) { }
  void print(std::ostream& os = std::cout) { }
};

// used for user-specified values. do not instantiate this class in user code.
struct SDFWireLoad: public WireLoad {
  std::unordered_map<VerilogWire*, MyFloat> c;
  using WireLeg = std::pair<VerilogWire*, VerilogPin*>;
  std::unordered_map<WireLeg, MyFloat, boost::hash<WireLeg>> delay;

  MyFloat wireC(VerilogWire* wire) {
    return (c.count(wire)) ? c[wire] : 0.0;
  }

  MyFloat wireDelay(MyFloat, VerilogWire* wire, VerilogPin* vPin) {
    auto leg = std::make_pair(wire, vPin);
    return (delay.count(leg)) ? delay[leg] : 0.0;
  }

  void setWireC(VerilogWire* wire, MyFloat value) {
    c[wire] = value;
  }

  void setWireDelay(VerilogWire* wire, VerilogPin* vPin, MyFloat value) {
    auto leg = std::make_pair(wire, vPin);
    delay[leg] = value;
  }

  void print(std::ostream& os = std::cout) { }
};

extern WireLoad* idealWireLoad;
extern WireLoad* sdfWireLoad;

#endif // GALOIS_EDA_WIRELOAD_H

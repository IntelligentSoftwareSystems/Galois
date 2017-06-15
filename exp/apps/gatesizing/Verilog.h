#include <string>
#include <unordered_map>
#include <unordered_set>

#include "CellLib.h"

#ifndef GALOIS_VERILOG_H
#define GALOIS_VERILOG_H

struct VerilogGate;
struct VerilogWire;

struct VerilogPin {
  std::string name;
  VerilogGate *gate;
  VerilogWire *wire;
};

struct VerilogWire {
  std::string name;
  VerilogPin *root;
  WireLoad *wireLoad;
  std::unordered_set<VerilogPin *> leaves;
};

struct VerilogGate {
  std::string name;
  Cell *cell;
  std::unordered_set<VerilogPin *> inPins, outPins;
};

struct VerilogModule {
  std::string name;
  std::unordered_map<std::string, VerilogPin *> inputs, outputs;
  std::unordered_map<std::string, VerilogGate *> gates;
  std::unordered_map<std::string, VerilogWire *> wires;

  VerilogModule(std::string inName, CellLib& cellLib);
  ~VerilogModule();

  void printVerilogModuleDebug();
};

#endif // GALOIS_VERILOG_H


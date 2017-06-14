#include <string>
#include <unordered_map>
#include <unordered_set>

#ifndef GALOIS_VERILOG_H
#define GALOIS_VERILOG_H

struct VerilogGate;
struct VerilogWire;

struct VerilogPin {
  VerilogGate *gate;
  VerilogWire *wire;
  std::string name;
};

struct VerilogWire {
  std::string name;
  VerilogPin *root;
  std::unordered_set<VerilogPin *> leaves;
};

struct VerilogGate {
  std::string name;
  std::string typeName;
  VerilogPin *outPin;
  std::unordered_set<VerilogPin *> inPins;
};

struct VerilogModule {
  std::string name;
  std::unordered_map<std::string, VerilogPin *> inputs, outputs;
  std::unordered_map<std::string, VerilogGate *> gates;
  std::unordered_map<std::string, VerilogWire *> wires;

  VerilogModule(std::string inName);
  ~VerilogModule();

  void printVerilogModule();
};

#endif // GALOIS_VERILOG_H


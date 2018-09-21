#ifndef GALOIS_EDA_VERILOG_H
#define GALOIS_EDA_VERILOG_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cassert>

#include "Tokenizer.h"

// forward declaration
struct VerilogDesign;
struct VerilogModule;
struct VerilogGate;
struct VerilogPin;
struct VerilogWire;

struct VerilogParser {
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;
  VerilogDesign& design;
  VerilogModule* curModule;

private:
  // for token stream
  void tokenizeFile(std::string inName);
  bool isEndOfTokenStream() { return curToken == tokens.end(); }
  bool isEndOfStatement() { return (isEndOfTokenStream() || ";" == *curToken); };

  // for parsing Verilog constructs
  void parseModule();
  void parseWires();
  void parseInPins();
  void parseOutPins();
  void parseAssign();
  void parseGate();
  Token getVarName();

public:
  VerilogParser(VerilogDesign& d): design(d) {}
  void parse(std::string inName);
};

struct VerilogPin {
  std::string name;
  VerilogModule* module;
  VerilogGate* gate;
  VerilogWire* wire;

public:
  void print(std::ostream& os = std::cout);
};

struct VerilogGate {
  std::string name;
  std::string cellType;
  VerilogModule* module;
  std::unordered_map<std::string, VerilogPin*> pins;

public:
  void print(std::ostream& os = std::cout);
  ~VerilogGate();

  VerilogPin* addPin(std::string name) {
    VerilogPin* pin = new VerilogPin;
    pin->name = name;
    pin->gate = this;
    pin->module = module;
    pin->wire = nullptr;
    pins[name] = pin;
    return pin;
  }

  VerilogPin* findPin(std::string name) {
    auto it = pins.find(name);
    return (it == pins.end()) ? nullptr : it->second;
  }
};

struct VerilogWire {
  std::string name;
  std::unordered_set<VerilogPin*> pins;
  VerilogModule* module;

public:
  void print(std::ostream& os = std::cout);

  void addPin(VerilogPin* pin) { pins.insert(pin); }

  size_t deg() { return pins.size(); }
  size_t outDeg() { return (deg() - 1); }
};

struct VerilogModule {
  std::string name;

  // components of a module
  std::unordered_map<std::string, VerilogGate*> gates;
  std::unordered_map<std::string, VerilogWire*> wires;
  std::unordered_map<std::string, VerilogPin*> pins;
  std::unordered_set<VerilogPin*> inPins;
  std::unordered_set<VerilogPin*> outPins;

  // for dependencies among modules
  std::unordered_set<VerilogModule*> pred;
  std::unordered_set<VerilogModule*> succ;

public:
  void print(std::ostream& os = std::cout);
  VerilogModule();
  ~VerilogModule();

  VerilogGate* addGate(std::string name, std::string cellType) {
    VerilogGate* gate = new VerilogGate;
    gate->name = name;
    gate->cellType = cellType;
    gate->module = this;
    gates[name] = gate;
    return gate;
  }

  VerilogGate* findGate(std::string name) {
    auto it = gates.find(name);
    return (it == gates.end()) ? nullptr : it->second;
  }

  VerilogPin* addPin(std::string name) {
    VerilogPin* pin = new VerilogPin;
    pin->name = name;
    pin->module = this;
    pin->gate = nullptr;
    pin->wire = nullptr;
    pins[name] = pin;
    return pin;
  }

  VerilogPin* findPin(std::string name) {
    auto it = pins.find(name);
    return (it == pins.end()) ? nullptr : it->second;
  }

  void addInPin(VerilogPin* pin) {
    assert(pins.count(pin->name));
    inPins.insert(pin);
  }

  bool isInPin(VerilogPin* pin) {
    return inPins.count(pin);
  }

  void addOutPin(VerilogPin* pin) {
    assert(pins.count(pin->name));
    outPins.insert(pin);
  }

  bool isOutPin(VerilogPin* pin) {
    return outPins.count(pin);
  }

  VerilogWire* addWire(std::string name) {
    VerilogWire* wire = new VerilogWire;
    wire->name = name;
    wire->module = this;
    wires[name] = wire;
    return wire;
  }

  VerilogWire* findWire(std::string name);
  bool isHierarchical();
};

struct VerilogDesign {
  std::unordered_map<std::string, VerilogModule*> modules;
  std::unordered_set<VerilogModule*> roots;

private:
  void clear();
  void clearDependency();

public:
  void parse(std::string inName, bool toClear = false);
  void print(std::ostream& os = std::cout);

  void buildDependency();
  bool isHierarchical();

  ~VerilogDesign() { clear(); }

  VerilogModule* addModule(std::string name) {
    VerilogModule* module = new VerilogModule;
    module->name = name;
    modules[name] = module;
    return module;
  }

  VerilogModule* findModule(std::string name) {
    auto it = modules.find(name);
    return (it == modules.end()) ? nullptr : it->second;
  }
};

#endif // GALOIS_EDA_VERILOG_H

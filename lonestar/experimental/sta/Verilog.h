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
struct VerilogModule;
struct VerilogPin;
struct VerilogWire;

struct VerilogParser {
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;
  VerilogModule* module;

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
  void parseSubModule();
  Token getVarName();

public:
  VerilogParser(VerilogModule* module): module(module) {}
  void parse(std::string inName);
};

struct VerilogPin {
  std::string name;
  VerilogModule* module;
  VerilogWire* wire;

public:
  void print(std::ostream& os = std::cout);
};

struct VerilogWire {
  std::string name;
  std::unordered_set<VerilogPin*> pins;
  VerilogModule* module;

public:
  void print(std::ostream& os = std::cout);
  void addPin(VerilogPin* pin);

  size_t deg() { return pins.size(); }
  size_t outDeg() { return (deg() - 1); }
};

struct VerilogModule {
public:
  using Name2ModuleMap = std::unordered_map<std::string, VerilogModule*>;
  using Name2PinMap = std::unordered_map<std::string, VerilogPin*>;
  using Name2WireMap = std::unordered_map<std::string, VerilogWire*>;
  using PinSet = std::unordered_set<VerilogPin*>;

public:
  std::string name;
  std::string cellType;
  Name2ModuleMap subModules;
  Name2PinMap pins;
  PinSet inPins;
  PinSet outPins;
  Name2WireMap wires;
  VerilogModule* parentModule;

private:
  void setup(bool isTopModule);
  void clear();

public:
  void parse(std::string inName, bool toClear = false);
  void print(std::ostream& os = std::cout);
  VerilogModule(bool isTopModule = true);
  ~VerilogModule();

  VerilogModule* addSubModule(std::string name, std::string cellType);
  VerilogModule* findSubModule(std::string name);

  VerilogPin* addPin(std::string name);
  VerilogPin* findPin(std::string name);

  void addInPin(VerilogPin* pin) {
    assert(pins.count(pin));
    inPins.insert(pin);
  }

  void addOutPin(VerilogPin* pin) {
    assert(pins.count(pin));
    outPins.insert(pin);
  }

  VerilogWire* addWire(std::string name);
  VerilogWire* findWire(std::string name);
};

#endif // GALOIS_EDA_VERILOG_H

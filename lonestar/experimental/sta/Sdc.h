#ifndef GALOIS_EDA_SDC_H
#define GALOIS_EDA_SDC_H

#include "TimingDefinition.h"
#include "Tokenizer.h"
#include "CellLib.h"
#include "Verilog.h"
#include "Clock.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <limits>
#include <string>

struct SDC;

struct SDCParser {
  SDC& sdc;
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;

private:
  // for token stream
  void tokenizeFile(std::string inName);
  bool isEndOfTokenStream() { return curToken == tokens.end(); }
  bool isAttributeOfCommand() { return ('-' == (*curToken)[0] || '[' == (*curToken)[0]); }
  bool isEndOfCommand() { return (isEndOfTokenStream() || !isAttributeOfCommand()); }

  // for parsing sdc commands
  std::unordered_set<VerilogPin*> getPorts();
  void collectPort(std::unordered_set<VerilogPin*>& ports, Token name);
  void parseCreateClock();
  void parseSetInputDelay();
  void parseSetInputTransition();
  void parseSetLoad();
  void parseSetOutputDelay();
  Token getVarName();

public:
  SDCParser(SDC& sdc): sdc(sdc) {}

  void parse(std::string inName);
};

struct SDCEnvAtPort {
  // index: MAX_DELAY_MODE/MIN_DELAY_MODE, fall/rise
  MyFloat inputDelay[2][2];
  MyFloat inputSlew[2][2];
  MyFloat outputDelay[2][2];
  MyFloat outputLoad;
  Clock* clk;
  VerilogPin* pin;

public:
  void print(std::ostream& os = std::cout);
};

struct SDC {
private:
  void clear();

public:
  VerilogModule& m;

  std::unordered_map<std::string, Clock*> clocks;
  std::unordered_map<VerilogPin*, SDCEnvAtPort*> envAtPorts;

public:
  void parse(std::string inName, bool toClear = false);
  void print(std::ostream& os = std::cout);

  SDCEnvAtPort* getEnvAtPort(VerilogPin* pin) {
    MyFloat infinity = std::numeric_limits<MyFloat>::infinity();
    if (!envAtPorts.count(pin)) {
      auto env = new SDCEnvAtPort;
      for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
          env->inputDelay[i][j] = infinity;
          env->inputSlew[i][j] = infinity;
          env->outputDelay[i][j] = infinity;
        }
      }
      env->outputLoad = infinity;
      env->clk = nullptr;
      env->pin = nullptr;
      envAtPorts[pin] = env;
    }
    return envAtPorts[pin];
  }

  Clock* getClock(std::string name) {
    if (!clocks.count(name)) {
      clocks[name] = new Clock;
    }
    return clocks[name];
  }

  SDC(VerilogModule& m): m(m) {}
  ~SDC() { clear(); }
};

#endif // GALOIS_EDA_SDC_H

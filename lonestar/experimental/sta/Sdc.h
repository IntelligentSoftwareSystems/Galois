#ifndef GALOIS_EDA_SDC_H
#define GALOIS_EDA_SDC_H

#include "TimingDefinition.h"
#include "Tokenizer.h"
#include "CellLib.h"
#include "Verilog.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <limits>
#include <string>

struct SDC;

struct SDCDrivingCell {
  CellPin* toCellPin;
  CellPin* fromCellPin;
  MyFloat slew[2];

public:
  void print(std::ostream& os);
};

struct SDCClockEdge {
  MyFloat t;
  bool isRise;

public:
  void print(std::ostream& os);
};

struct SDCClock {
  MyFloat period;
  std::vector<SDCClockEdge> waveform;
  VerilogPin* src;
  std::string name;

public:
  void print(std::ostream& os);
};

struct SDCParser {
  SDC& sdc;
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;
  CellLib* defaultLib;
  CellLib* curLib;

private:
  // for token stream
  void tokenizeFile(std::string inName);
  bool isEndOfTokenStream() { return curToken == tokens.end(); }
  bool isAttributeOfCommand() { return ('-' == (*curToken)[0] || '[' == (*curToken)[0]); }
  bool isEndOfCommand() { return (isEndOfTokenStream() || !isAttributeOfCommand()); }

  // for parsing sdc commands
  std::unordered_set<VerilogPin*> getPorts();
  void collectPort(std::unordered_set<VerilogPin*>& ports, Token name);
  void parseClock();
  void parseDrivingCell();
  void parseLoad();
  void parseMaxDelay();
  Token getVarName();

public:
  SDCParser(SDC& sdc): sdc(sdc) {}

  void parse(std::string inName);
};

struct SDC {
private:
  void clear();

public:
  std::vector<CellLib*>& libs;
  VerilogModule& m;
  std::unordered_map<std::string, CellLib*> mapName2Libs;

  // boundary conditions
  std::unordered_map<VerilogPin*, SDCDrivingCell*> mapPin2DrivingCells;
  std::unordered_set<SDCDrivingCell*> drivingCells;
  std::unordered_map<VerilogPin*, MyFloat> pinLoads;

  // usual delay constraints
  MyFloat maxDelayPI2PO;
  MyFloat maxDelayPI2RI;
  MyFloat maxDelayRO2RI;
  MyFloat maxDelayRO2PO;

  std::unordered_map<std::string, SDCClock*> clocks;

public:
  void parse(std::string inName, bool toClear = false);
  void print(std::ostream& os = std::cout);

  SDCDrivingCell* addDrivingCell() {
    SDCDrivingCell* c = new SDCDrivingCell;
    drivingCells.insert(c);
    return c;
  }

  void attachPin2DrivingCell(VerilogPin* pin, SDCDrivingCell* c) {
    mapPin2DrivingCells[pin] = c;
  }

  SDCDrivingCell* findDrivingCell(VerilogPin* pin) {
    return mapPin2DrivingCells.at(pin);
  }

  SDC(std::vector<CellLib*>& libs, VerilogModule& m): libs(libs), m(m) {
    for (auto i: libs) {
      mapName2Libs[i->name] = i;
    }

    maxDelayPI2PO = std::numeric_limits<MyFloat>::infinity();
    maxDelayPI2RI = std::numeric_limits<MyFloat>::infinity();
    maxDelayRO2RI = std::numeric_limits<MyFloat>::infinity();
    maxDelayRO2PO = std::numeric_limits<MyFloat>::infinity();
  }

  ~SDC() { clear(); }
};

#endif // GALOIS_EDA_SDC_H

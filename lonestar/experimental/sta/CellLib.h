#ifndef GALOIS_EDA_CELL_LIB_H
#define GALOIS_EDA_CELL_LIB_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>

#include "Tokenizer.h"

// forward declaration
struct Lut;
struct CellLib;
struct CellPin;
struct Cell;

enum TableType {
  TABLE_DELAY = 0,
  TABLE_SLEW,
};

enum WireTreeType {
  TREE_TYPE_BEST_CASE = 0,
  TREE_TYPE_BALANCED,
  TREE_TYPE_WORST_CASE,
};

struct CellLibParser {
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;
  CellLib* lib;

private:
  // for token stream
  void tokenizeFile(std::string inName);
  Token getBooleanExpression();
  bool isEndOfTokenStream() { return curToken == tokens.end(); }
  bool isEndOfGroup() { return (isEndOfTokenStream() || "}" == *curToken); }

  // for skipping statements
  void skipAttribute();
  void skipGroup();
  void skip(bool isTopCall = true);

  // for parsing group statements
  void parseWireLoad();
  void parseLutTemplate();
  void parseLut(Lut* lut);
  void parseTiming(CellPin* pin);
  void parseCellPin(Cell* cell);
  void parseCellLeakagePower(Cell* cell);
  void parseCell();
  void parseOperatingConditions();
  void parseCellLibrary();

public:
  CellLibParser(CellLib* lib): lib(lib) {}
  void parse(std::string inName);
};

struct LutTemplate {
  std::string name;
  std::vector<std::string> var;
  std::vector<size_t> dim;
  CellLib* lib;

public:
  void print(std::ostream& os = std::cout);
};

struct Lut {
  LutTemplate* lutTemplate;

  std::vector<std::vector<float>> index;
  std::vector<float> value;

private:
  float lookupInternal(std::vector<float>& param, std::vector<std::pair<size_t, size_t>>& bound, std::vector<size_t>& diff, size_t start, size_t i);

public:
  float lookup(std::vector<float>& param);
  void print(std::string attr, std::ostream& os = std::cout);
};

struct CellPin {
  std::string name;

  // direction
  // inout = isInput & isOutput, input = isInput & !isOutput,
  // output = !isInput & isOutput, internal = !isInput & !isOutput
  bool isInput;
  bool isOutput;
  bool isClock;

  float c[2];
  float maxC;
  Cell* cell;
  std::string func;

  // order of keys:
  // pin, fall/rise, delay/slew(/power in the future), unateness, when
  using MapWhen2Lut = std::unordered_map<std::string, Lut*>;
  using MapFromPin = std::unordered_map<CellPin*, MapWhen2Lut[2][2][2]>;

  MapFromPin tables;

public:
  void print(std::ostream& os = std::cout);
  bool isUnateAtEdge(CellPin* inPin, bool isNeg, bool isRise);
  float extract(std::vector<float>& param, TableType index, CellPin* inPin, bool isNeg, bool isRise, std::string when);
  std::pair<float, std::string> extractMax(std::vector<float>& param, TableType index, CellPin* inPin, bool isNeg, bool isRise);
  std::pair<float, std::string> extractMin(std::vector<float>& param, TableType index, CellPin* inPin, bool isNeg, bool isRise);

  void addLut(Lut* lut, TableType tType, bool isRise, CellPin* relatedPin, std::string when, bool isPos, bool isNeg) {
    if (isPos) {
      tables[relatedPin][isRise][tType][0][when] = lut;
    }
    if (isNeg) {
      tables[relatedPin][isRise][tType][1][when] = lut;
    }
  }
};

struct Cell {
  std::string name;
  size_t driveStrength;
  float area;
  float cellLeakagePower;
  std::unordered_map<std::string, float> leakagePower;
  CellLib* lib;

  using MapCellPin = std::unordered_map<std::string, CellPin*>;

  MapCellPin pins;
  MapCellPin outPins;
  MapCellPin inPins;
  MapCellPin internalPins;

public:
  void print(std::ostream& os = std::cout);

  void addInOutPin(CellPin* pin) {
    pin->isInput = true;
    pin->isOutput = true;
    inPins[pin->name] = pin;
    outPins[pin->name] = pin;
  }

  void addInPin(CellPin* pin) {
    pin->isInput = true;
    pin->isOutput = false;
    inPins[pin->name] = pin;
  }

  void addOutPin(CellPin* pin) {
    pin->isInput = false;
    pin->isOutput = true;
    outPins[pin->name] = pin;
  }

  void addInternalPin(CellPin* pin) {
    pin->isInput = false;
    pin->isOutput = false;
    internalPins[pin->name] = pin;
  }

  CellPin* addCellPin(std::string name) {
    auto pin = new CellPin;
    pin->name = name;
    pin->cell = this;
    pins[name] = pin;
    return pin;
  }

  CellPin* findCellPin(std::string name) {
    auto it = pins.find(name);
    return (it == pins.end()) ? nullptr : it->second;
  }

  void addLeakagePower(std::string when, float value) {
    leakagePower[when] = value;
  }
};

struct WireLoad {
  std::string name;
  float c;
  float r;
  float slope;
  std::map<size_t, float> fanoutLength;
  CellLib* lib;

private:
  float wireLength(size_t deg);

public:
  float wireR(size_t deg);
  float wireC(size_t deg);
  float wireDelay(float loadC, size_t deg);
  void print(std::ostream& os = std::cout);

  void addFanoutLength(size_t fanout, float length) {
    fanoutLength[fanout] = length;
  }
};

struct CellLib {
  std::string name;
  std::string opCond;
  WireLoad* defaultWireLoad;
  float defaultInoutPinCap;
  float defaultInputPinCap;
  float defaultOutputPinCap;
  float defaultMaxSlew;
  WireTreeType wireTreeType;

  std::unordered_map<std::string, WireLoad*> wireLoads;
  std::unordered_map<std::string, Cell*> cells;
  std::unordered_map<std::string, LutTemplate*> lutTemplates;

  // set of LUTs to avoid double free
  // lookup should be done from pins
  std::unordered_set<Lut*> luts;

private:
  void clear();
  void setup();

public:
  void parse(std::string inName, bool toClear = false);
  void print(std::ostream& os = std::cout);
  CellLib();
  ~CellLib();

  Cell* addCell(std::string name) {
    Cell* cell = new Cell;
    cell->name = name;
    cell->lib = this;
    cells[name] = cell;
    return cell;
  }

  Cell* findCell(std::string name) {
    auto it = cells.find(name);
    return (it == cells.end()) ? nullptr : it->second;
  }

  WireLoad* addWireLoad(std::string name) {
    WireLoad* wireLoad = new WireLoad;
    wireLoad->name = name;
    wireLoad->lib = this;
    wireLoads[name] = wireLoad;
    return wireLoad;
  }

  WireLoad* findWireLoad(std::string name) {
    auto it = wireLoads.find(name);
    return (it == wireLoads.end()) ? nullptr : it->second;
  }

  LutTemplate* addLutTemplate(std::string name) {
    LutTemplate* lutTemplate = new LutTemplate;
    lutTemplate->name = name;
    lutTemplate->lib = this;
    lutTemplates[name] = lutTemplate;
    return lutTemplate;
  }

  LutTemplate* findLutTemplate(std::string name) {
    auto it = lutTemplates.find(name);
    return (it == lutTemplates.end()) ? nullptr : it->second;
  }

  Lut* addLut() {
    Lut* lut = new Lut;
    luts.insert(lut);
    return lut;
  }
};

#endif // GALOIS_EDA_CELL_LIB_H

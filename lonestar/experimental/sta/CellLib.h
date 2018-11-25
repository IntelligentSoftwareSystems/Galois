#ifndef GALOIS_EDA_CELL_LIB_H
#define GALOIS_EDA_CELL_LIB_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include <limits>

#include "TimingDefinition.h"
#include "Tokenizer.h"
#include "WireLoad.h"
#include "Verilog.h"

#include "galois/Galois.h"

// forward declaration
struct Lut;
struct TimingTable;
struct PowerTable;
struct CellLib;
struct CellPin;
struct Cell;

enum TableType {
  DELAY = 0,
  SLEW,
  MIN_CONSTRAINT,
  MAX_CONSTRAINT,
//  POWER,
};

enum WireTreeType {
  BEST_CASE_TREE = 0,
  BALANCED_TREE,
  WORST_CASE_TREE,
};

enum VariableType {
  INPUT_TRANSITION_TIME = 0,
  CONSTRAINED_PIN_TRANSITION,
  RELATED_PIN_TRANSITION,
  TOTAL_OUTPUT_NET_CAPACITANCE,
  INPUT_NET_TRANSITION,
  TIME,
};

enum TimingSense {
  POSITIVE_UNATE = 0,
  NEGATIVE_UNATE,
  NON_UNATE,
  NOT_APPLICABLE,
};

enum TimingType {
  // combinational delay arcs
  COMBINATIONAL = 0,
  COMBINATIONAL_RISE,
  COMBINATIONAL_FALL,
  // three-state delay arcs
  THREE_STATE_DISABLE,
  THREE_STATE_DISABLE_RISE,
  THREE_STATE_DISABLE_FALL,
  THREE_STATE_ENABLE,
  THREE_STATE_ENABLE_RISE,
  THREE_STATE_ENABLE_FALL,
  // sequential delay arcs
  RISING_EDGE,
  FALLING_EDGE,
  // async delay arcs
  PRESET,
  CLEAR,
  // sequential constraints
  HOLD_RISING,
  HOLD_FALLING,
  SETUP_RISING,
  SETUP_FALLING,
  RECOVERY_RISING,
  RECOVERY_FALLING,
  REMOVAL_RISING,
  REMOVAL_FALLING,
  // clock waveform constraints
  MIN_PULSE_WIDTH,
  MINIMUM_PERIOD,
  // clock skew constraints
  SKEW_RISING,
  SKEW_FALLING,
  MAX_CLOCK_TREE_PATH,
  MIN_CLOCK_TREE_PATH,
  // non-sequential constraints
  NON_SEQ_SETUP_RISING,
  NON_SEQ_SETUP_FALLING,
  NON_SEQ_HOLD_RISING,
  NON_SEQ_HOLD_FALLING,
  // no-change constraints
  NOCHANGE_HIGH_HIGH,
  NOCHANGE_HIGH_LOW,
  NOCHANGE_LOW_HIGH,
  NOCHANGE_LOW_LOW,
};

enum PinDirection {
  INPUT = 0,
  OUTPUT,
  INOUT,
  INTERNAL,
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
  void parseLutTemplate(bool isForPower = false);
  void parseLut(Lut* lut, bool isForPower = false);
  void parseTiming(TimingTable* tTable);
  void parseInternalPower(PowerTable* pTable);
  void parseCellPin(Cell* cell);
  void parseCellLeakagePower(Cell* cell);
  void parseCell();
  void parseOperatingConditions();
  void parseCellLibrary();

public:
  CellLibParser(CellLib* lib): lib(lib) {}
  void parse(std::string inName);
};

using VecOfMyFloat = std::vector<MyFloat>;

struct LutTemplate {
  std::vector<size_t> shape;
  std::vector<size_t> stride;
  std::vector<VariableType> var;
  std::vector<VecOfMyFloat> index;
  std::string name;
  bool isForPower;
  CellLib* lib;

public:
  void print(std::ostream& os = std::cout);
  void wrapUpConstruction();
};

using Parameter =
    std::unordered_map<
      VariableType,
      MyFloat,
      std::hash<VariableType>,
      std::equal_to<VariableType>,
      galois::PerIterAllocTy::rebind<std::pair<const VariableType, MyFloat>>::other
    >;

struct Lut {
  LutTemplate* lutTemplate;

  std::vector<VecOfMyFloat> index;
  VecOfMyFloat value;

private:
  MyFloat lookupInternal(std::vector<MyFloat, galois::PerIterAllocTy::rebind<MyFloat>::other>& param, 
                         std::vector<std::pair<size_t, size_t>, galois::PerIterAllocTy::rebind<std::pair<size_t, size_t>>::other>& bound,
                         std::vector<size_t>& stride, size_t start, size_t lv);

public:
  MyFloat lookup(Parameter& param, galois::PerIterAllocTy& alloc);
  void print(std::string attr, std::ostream& os = std::cout);
  void wrapUpConstruction();
};

struct TimingTable {
  CellPin* relatedPin;
  TimingType tType;
  TimingSense unate;
  std::string when;
  std::string nameOfRelatedPin;

  // tables. ~[0] for falling, ~[1] for rising
  Lut* delay[2];
  Lut* slew[2];
  Lut* constraint[2];

  // end point of an arc, i.e. the pin owning the tables
  CellPin* endPin;

public:
  ~TimingTable();
  void print(std::ostream& os = std::cout);
  void wrapUpConstruction();
};

struct PowerTable {
  CellPin* relatedPin;
  std::string when;
  Lut* internalPower[2];
  CellPin* endPin;
  std::string nameOfRelatedPin;

public:
  ~PowerTable();
  void print(std::ostream& os = std::cout);
  void wrapUpConstruction();
};

struct CellPin {
  std::string name;

  PinDirection dir;
  bool isClock;
  bool isClockGated;

  MyFloat c[2]; // fall/rise capaitance
  MyFloat maxC; // maximum capacitance
  MyFloat minC;
  Cell* cell;
  std::string func;
  std::string func_up;
  std::string func_down;

  std::unordered_set<TimingTable*> timings;
  std::unordered_set<PowerTable*> powers;

  // organization of tables for lookup
  // order of keys:
  //   1. incoming pin
  //   2. inRise
  //   3. meRise
  //   4. delay/slew/min_constraint/max_constraint
  //   5. when
  using InnerTimingMap = std::unordered_map<std::string, Lut*>;
  std::unordered_map<CellPin*, InnerTimingMap[2][2][4]> timingMap;

public:
  ~CellPin();
  void print(std::ostream& os = std::cout);
  void wrapUpConstruction();
  bool isEdgeDefined(CellPin* inPin, bool isInRise, bool isMeRise, TableType index = DELAY);
  MyFloat extract(Parameter& param, TableType index, CellPin* inPin, bool isInRise, bool isMeRise, std::string when, galois::PerIterAllocTy& alloc);
  std::pair<MyFloat, std::string> extractMax(Parameter& param, TableType index, CellPin* inPin, bool isInRise, bool isMeRise, galois::PerIterAllocTy& alloc);
  std::pair<MyFloat, std::string> extractMin(Parameter& param, TableType index, CellPin* inPin, bool isInRise, bool isMeRise, galois::PerIterAllocTy& alloc);

  PinDirection pinDir() { return dir; }
  bool isClockPin() { return isClock; }

  TimingTable* addTimingTable() {
    TimingTable* tTable = new TimingTable;
    tTable->endPin = this;
    tTable->relatedPin = nullptr;
    tTable->unate = NOT_APPLICABLE;
    tTable->tType = COMBINATIONAL;
    for (int i = 0; i < 2; i++) {
      tTable->delay[i] = nullptr;
      tTable->slew[i] = nullptr;
      tTable->constraint[i] = nullptr;
    }
    timings.insert(tTable);
    return tTable;
  }

  PowerTable* addPowerTable() {
    PowerTable* pTable = new PowerTable;
    pTable->endPin = this;
    pTable->relatedPin = nullptr;
    for (int i = 0; i < 2; i++) {
      pTable->internalPower[i] = nullptr;
    }
    powers.insert(pTable);
    return pTable;
  }
};

struct Cell {
  std::string name;
  size_t driveStrength;
  MyFloat area;
  MyFloat cellLeakagePower;
  std::unordered_map<std::string, MyFloat> leakagePower;
  CellLib* lib;

  std::unordered_map<std::string, CellPin*> pins;

  using PinSet = std::unordered_set<CellPin*>;
  PinSet outPins;
  PinSet inPins;
  PinSet internalPins;
  PinSet clockPins;

public:
  ~Cell();
  void wrapUpConstruction();
  void print(std::ostream& os = std::cout);

  void addInOutPin(CellPin* pin) {
    pin->dir = INOUT;
    inPins.insert(pin);
    outPins.insert(pin);
  }

  void addInPin(CellPin* pin) {
    pin->dir = INPUT;
    inPins.insert(pin);
  }

  void addOutPin(CellPin* pin) {
    pin->dir = OUTPUT;
    outPins.insert(pin);
  }

  void addInternalPin(CellPin* pin) {
    pin->dir = INTERNAL;
    internalPins.insert(pin);
  }

  void addClockPin(CellPin* pin, bool isGated = false) {
    pin->isClock = true;
    pin->isClockGated = isGated;
    clockPins.insert(pin);
  }

  CellPin* addCellPin(std::string name) {
    auto pin = new CellPin;
    pin->name = name;
    pin->cell = this;
    pin->c[0] = std::numeric_limits<MyFloat>::infinity();
    pin->c[1] = std::numeric_limits<MyFloat>::infinity();
    pin->maxC = 0.0;
    pin->minC = 0.0;
    pin->isClock = false;
    pin->isClockGated = false;
    pins[name] = pin;
    return pin;
  }

  CellPin* findCellPin(std::string name) {
    auto it = pins.find(name);
    return (it == pins.end()) ? nullptr : it->second;
  }

  void addLeakagePower(std::string when, MyFloat value) {
    leakagePower[when] = value;
  }

  bool isSequentialCell() { return !clockPins.empty(); }
};

// compute wire value as in cell lib
// if user needs to set values, use SDFWireLoad instead
struct PreLayoutWireLoad: public WireLoad {
  MyFloat c;
  MyFloat r;
  MyFloat slope;
  std::map<size_t, MyFloat> fanoutLength;
  CellLib* lib;

private:
  MyFloat wireLength(size_t deg);

public:
  MyFloat wireC(VerilogWire* wire);
  MyFloat wireDelay(MyFloat loadC, VerilogWire* wire, VerilogPin* vPin);
  void print(std::ostream& os = std::cout);

  void setWireC(VerilogWire*, MyFloat) { }
  void setWireDelay(VerilogWire*, VerilogPin*, MyFloat) { }

  void addFanoutLength(size_t fanout, MyFloat length) {
    fanoutLength[fanout] = length;
  }
};

struct CellLib {
  std::string name;
  std::string opCond;
  PreLayoutWireLoad* defaultWireLoad;
  MyFloat defaultInoutPinCap;
  MyFloat defaultInputPinCap;
  MyFloat defaultOutputPinCap;
  MyFloat defaultMaxSlew;
  WireTreeType wireTreeType;

  std::unordered_map<std::string, PreLayoutWireLoad*> wireLoads;
  std::unordered_map<std::string, Cell*> cells;
  std::unordered_map<std::string, LutTemplate*> lutTemplates;
  std::unordered_map<std::string, LutTemplate*> powerLutTemplates;

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
    cell->driveStrength = 0;
    cell->area = 0.0;
    cell->cellLeakagePower = 0.0;
    cells[name] = cell;
    return cell;
  }

  Cell* findCell(std::string name) {
    auto it = cells.find(name);
    return (it == cells.end()) ? nullptr : it->second;
  }

  PreLayoutWireLoad* addWireLoad(std::string name) {
    PreLayoutWireLoad* wireLoad = new PreLayoutWireLoad;
    wireLoad->name = name;
    wireLoad->lib = this;
    wireLoad->c = 0.0;
    wireLoad->r = 0.0;
    wireLoad->slope = 0.0;
    wireLoads[name] = wireLoad;
    return wireLoad;
  }

  PreLayoutWireLoad* findWireLoad(std::string name) {
    auto it = wireLoads.find(name);
    return (it == wireLoads.end()) ? nullptr : it->second;
  }

  LutTemplate* addLutTemplate(std::string name, bool isForPower = false) {
    LutTemplate* lutTemplate = new LutTemplate;
    lutTemplate->name = name;
    lutTemplate->lib = this;
    lutTemplate->isForPower = isForPower;
    auto& map = (isForPower) ? powerLutTemplates : lutTemplates;
    map[name] = lutTemplate;
    return lutTemplate;
  }

  LutTemplate* findLutTemplate(std::string name, bool isForPower = false) {
    auto& map = (isForPower) ? powerLutTemplates : lutTemplates;
    auto it = map.find(name);
    return (it == map.end()) ? nullptr : it->second;
  }
};

#endif // GALOIS_EDA_CELL_LIB_H

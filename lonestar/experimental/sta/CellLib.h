#ifndef GALOIS_EDA_CELL_LIB_H
#define GALOIS_EDA_CELL_LIB_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <utility>
#include <iostream>

#include <boost/functional/hash.hpp>

#include "Tokenizer.h"

// forward declaration
struct Lut;
struct CellLib;
struct CellPin;
struct Cell;

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
  void skipGroupStatement();
  void skip(bool isTopCall = true);

  // for parsing group statements
  void parseWireLoad();
  void parseLutTemplate();
  void parseLut(Lut* lut);
  void parseTiming(CellPin* pin);
  void parseCellPin(Cell* cell);
  void parseCellLeakagePower(Cell* cell);
  void parseCell();
  void parseOperatingCondition();
  void parseCellLibrary();

public:
  CellLibParser(CellLib* lib): lib(lib) {}
  void parse(std::string inName);
};

struct LutTemplate {
  std::string name;
  std::vector<std::string> var;
  std::vector<size_t> dim;

public:
  void print(std::ostream& os = std::cout);
};

struct Lut {
  LutTemplate* lutTemplate;

  std::vector<std::vector<float>> index;
  std::vector<float> value;

public:
  float lookup(std::vector<float>& param);
  void print(std::ostream& os = std::cout);
};

struct CellPin {
  std::string name;

  // direction
  // inout = isInput & isOutput, input = isInput & !isOutput,
  // output = !isInput & isOutput, internal = !isInput & !isOutput
  bool isInput;
  bool isOutput;

  float riseC;
  float fallC;
  float maxC;
  Cell* cell;
  std::string func;

public:
  void print(std::ostream& os = std::cout);
};

struct Cell {
  std::string name;
  size_t driveStrength;
  float area;
  float cellLeakagePower;
  std::unordered_map<std::string, float> leakagePower;

  using MapCellPin = std::unordered_map<std::string, CellPin*>;

  MapCellPin pins;
  MapCellPin outPins;
  MapCellPin inPins;
  MapCellPin internalPins;

public:
  void print(std::ostream& os = std::cout);
};

struct WireLoad {
  std::string name;
  float c;
  float r;
  float slope;
  std::map<size_t, float> fanoutLength;

public:
  float wireR(size_t deg);
  float wireC(size_t deg);
  void print(std::ostream& os = std::cout);
};

struct CellLib {
  std::string name;
  WireLoad* defaultWireLoad;
  float defaultInoutPinCap;
  float defaultInputPinCap;
  float defaultOutputPinCap;
  std::unordered_map<std::string, WireLoad*> wireLoads;
  std::unordered_map<std::string, Cell*> cells;
  std::unordered_map<std::string, LutTemplate*> lutTemplates;

  // set of LUTs to avoid double free
  std::unordered_set<Lut*> luts;

  using TimingArc = std::pair<CellPin*, CellPin*>;
  using MapWhen2Lut = std::unordered_map<std::string, Lut*>;
  using MapLut = std::unordered_map<TimingArc, MapWhen2Lut, boost::hash<TimingArc>>;

  // [0]: positive unate, [1]: negative unate.
  // non-unate will appear in both
  MapLut riseSlew[2];
  MapLut fallSlew[2];
  MapLut cellRise[2];
  MapLut cellFall[2];

private:
  void clear();
  void setup();

public:
  void parse(std::string inName, bool toClear = false);
  void print(std::ostream& os = std::cout);
  bool isPosUnateForFall(CellPin* outPin, CellPin* inPin);
  bool isPosUnateForRise(CellPin* outPin, CellPin* inPin);
  bool isNegUnateForFall(CellPin* outPin, CellPin* inPin);
  bool isNegUnateForRise(CellPin* outPin, CellPin* inPin);
  CellLib();
  ~CellLib();
};

#endif // GALOIS_EDA_CELL_LIB_H

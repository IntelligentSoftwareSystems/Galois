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

struct CellLib;

struct CellLibParser {
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;
  CellLib* lib;

private:
  // for token stream
  void tokenizeFile(std::string inName);
  bool isEndOfTokenStream() { return curToken == tokens.end(); }
  bool isEndOfGroup() { return (isEndOfTokenStream() || "}" == *curToken); }

  // for skipping statements
  void skipAttribute();
  void skipGroupStatement();
  void skip();

  // for parsing group statements
  void parseWireLoad();
  void parseLutTemplate();
  void parseLut();
  void parseCellPin();
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

};

struct Cell {

};

struct CellPin {

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
  std::unordered_map<std::string, WireLoad*> wireLoads;
  std::unordered_map<std::string, Cell*> cells;
  std::unordered_map<std::string, LutTemplate*> lutTemplates;

private:
  void clear();
  void setup();

public:
  void parse(std::string inName, bool toClear = false);
  void print(std::ostream& os = std::cout);
  CellLib();
  ~CellLib();
};

#endif // GALOIS_EDA_CELL_LIB_H

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

class CellLib;

class CellLibParser {
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;
  CellLib* lib;

private:
  void tokenizeFile(std::string inName);
  void skipAttribute();
  void skipGroupStatement();
  void skip();
  void parseLutTemplate();
  void parseLut();
  void parseCellPin();
  void parseCell();
  void parseWireLoad();
  void parseCellLibrary();

public:
  CellLibParser(CellLib* lib): lib(lib) {}
  void parse(std::string inName);
};

class LutTemplate {

};

class Lut {

};

class Cell {

};

class CellPin {

};

class WireLoad {

};

class CellLib {
  friend class CellLibParser;

private:
  std::string name;

public:
  void parse(std::string inName);
  void print(std::ostream& os);
};

#endif // GALOIS_EDA_CELL_LIB_H

#ifndef AIGPARSER_H_
#define AIGPARSER_H_
#include <string>
#include <fstream>
#include <sstream>

#include "semantic_error.h"
#include "syntax_error.h"
#include "unexpected_eof.h"
#include "../misc/util/utilString.h"
#include "../subjectgraph/aig/Aig.h"

class AigParser {

private:
  unsigned currLine;
  unsigned currChar;
  std::ifstream file;
  int m, i, l, o, a;
  std::vector<int> inputs, outputs;
  std::vector<std::tuple<int, int, bool>> latches;
  std::vector<std::tuple<int, int, int>> ands;
  std::vector<std::string> inputNames, latchNames, outputNames;
  std::vector<aig::GNode> nodes;
  std::vector<int> levelHistogram;

  aig::Aig& aig;
  std::string designName;

  unsigned decode();
  bool parseBool(std::string delimChar);
  char parseByte();
  unsigned char parseChar();
  int parseInt(std::string delimChar);
  std::string parseString(std::string delimChar);

  void resize();

  void parseAagHeader();
  void parseAigHeader();
  void parseAagInputs();
  void parseAigInputs();
  void parseAagLatches();
  void parseAigLatches();
  void parseOutputs();
  void parseAagAnds();
  void parseAigAnds();
  void parseSymbolTable();

  void createAig();
  void createConstant();
  void createInputs();
  void createLatches();
  void createOutputs();
  void createAnds();
  void connectAndsWithFanoutMap();

  void connectLatches();
  void connectOutputs();
  void connectAnds();

public:
  AigParser(aig::Aig& aig);
  AigParser(std::string fileName, aig::Aig& aig);
  virtual ~AigParser();

  void open(std::string fileName);
  bool isOpen() const;
  void close();
  void parseAag();
  void parseAig();

  int getI();
  int getL();
  int getO();
  int getA();
  int getE();

  std::vector<int>& getLevelHistogram();
};

#endif /* AIGPARSER_H_ */

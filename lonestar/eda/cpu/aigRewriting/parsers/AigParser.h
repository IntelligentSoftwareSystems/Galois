/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

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

/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/Timer.h"
#include <AigParser.h>
#include <iostream>
#include <algorithm>

AigParser::AigParser(aig::Aig& aig) : aig(aig) {
  currLine = 0;
  currChar = 0;
}

AigParser::AigParser(std::string fileName, aig::Aig& aig) : aig(aig) {
  currLine = 1;
  currChar = 0;
  open(fileName);
}

AigParser::~AigParser() { close(); }

void AigParser::open(std::string fileName) {
  close();
  currLine = 1;
  currChar = 0;
  file.open(fileName.c_str());
}

bool AigParser::isOpen() const { return file.is_open(); }

void AigParser::close() {
  if (isOpen()) {
    file.close();
    currLine = 0;
    currChar = 0;
  }
}

unsigned AigParser::decode() {
  unsigned x = 0, in = 0;
  unsigned char c;
  while ((c = parseByte()) & 0x80)
    x |= (c & 0x7f) << (7 * in++);
  return x | (c << (7 * in));
}

char AigParser::parseByte() {
  char byte;
  file.read(&byte, 1);
  if (file.eof()) {
    throw unexpected_eof(currLine, currChar);
  }
  if (byte == '\n') {
    currLine++;
    currChar = 0;
  } else {
    currChar++;
  }
  return byte;
}

bool AigParser::parseBool(std::string delimChar) {
  bool result;
  int c = parseChar();
  switch (c) {
  case '0':
    result = false;
    break;
  case '1':
    result = true;
    break;
  default:
    throw syntax_error(
        currLine, currChar,
        format("Expected Boolean (0 or 1) but found: ASCII %d", c));
  }
  c = parseChar();
  if (delimChar.find(c) == std::string::npos) {
    throw syntax_error(
        currLine, currChar,
        format("Expected Boolean (0 or 1) but found: ASCII %d", c));
  }
  return result;
}

unsigned char AigParser::parseChar() {
  int result = file.get();
  if (file.eof()) {
    throw unexpected_eof(currLine, currChar);
  }
  if (result == '\r') {
    result = file.get();
    if (file.eof()) {
      throw unexpected_eof(currLine, currChar);
    }
  }
  if (result == '\n') {
    currLine++;
    currChar = 0;
  } else {
    currChar++;
  }
  return result;
}

int AigParser::parseInt(std::string delimChar) {
  unsigned char c;
  int result;
  bool done = false;
  std::stringstream buffer;
  c = parseChar();
  if (!isdigit(c))
    throw syntax_error(currLine, currChar,
                       format("Expected integer but found: ASCII %d", c));
  buffer << c;
  while (!done) {
    c = parseChar();
    if (isdigit(c)) {
      buffer << c;
    } else if (delimChar.find(c) != std::string::npos) {
      buffer >> result;
      done = true;
    } else {
      throw syntax_error(currLine, currChar,
                         format("Expected integer but found: ASCII %d", c));
    }
  }
  return result;
}

std::string AigParser::parseString(std::string delimChar) {
  bool done = false;
  unsigned char c;
  std::stringstream buffer;
  c = parseChar();
  if (delimChar.find(c) != std::string::npos || c == '\0') {
    throw syntax_error(currLine, currChar,
                       format("Expected integer but found: ASCII %d", c));
  }
  buffer << c;
  while (!done) {
    c = parseChar();
    if (delimChar.find(c) != std::string::npos || c == '\0') {
      done = true;
    } else {
      buffer << c;
    }
  }
  return buffer.str();
}

void AigParser::resize() {
  inputs.resize(i);
  inputNames.resize(i);
  latches.resize(l);
  latchNames.resize(l);
  outputs.resize(o);
  outputNames.resize(o);
  ands.resize(a);
  int nNodes = m + o + 1;
  aig.getNodes().resize(nNodes);
  aig.getNodesTravId().resize(nNodes);
  aig.getNodesFanoutMap().resize(nNodes);
}

void AigParser::parseAagHeader() {
  std::string aag = parseString(" ");
  if (aag.compare("aag") != 0) {
    throw syntax_error(1, 0, "Expected aag header");
  }
  m = parseInt(" ");
  i = parseInt(" ");
  l = parseInt(" ");
  o = parseInt(" ");
  a = parseInt("\n");

  if (m != (i + l + a)) {
    throw semantic_error(1, 4, "Incorrect value for M");
  }
  resize();
}

void AigParser::parseAigHeader() {
  std::string aig = parseString(" ");
  if (aig.compare("aig") != 0) {
    throw syntax_error(1, 0, "Expected aig header");
  }
  m = parseInt(" ");
  i = parseInt(" ");
  l = parseInt(" ");
  o = parseInt(" ");
  a = parseInt("\n");

  if (m != (i + l + a)) {
    throw semantic_error(1, 4, "Incorrect value for M");
  }
  resize();
}

void AigParser::parseAagInputs() {
  for (int in = 0; in < i; in++) {
    inputs[in] = parseInt("\n");
  }
}

void AigParser::parseAigInputs() {
  int x = 2;
  for (int in = 0; in < i; in++) {
    inputs[in] = x;
    x          = x + 2;
  }
}

void AigParser::parseAagLatches() {
  unsigned line;
  for (int in = 0; in < l; in++) {
    int lhs = parseInt(" ");
    line    = currLine;
    int rhs = parseInt("\n");
    bool init;
    if (line == currLine) {
      init = parseBool("\n");
    } else {
      init = false;
    }
    latches[in] = std::make_tuple(lhs, rhs, init);
  }
}

void AigParser::parseAigLatches() {
  unsigned line;
  int lhs = i * 2 + 2;
  for (int in = 0; in < l; in++) {
    line    = currLine;
    int rhs = parseInt("\n");
    bool init;
    if (line == currLine) {
      init = parseBool("\n");
    } else {
      init = false;
    }
    latches[in] = std::make_tuple(lhs, rhs, init);
    lhs         = lhs + 2;
  }
}

void AigParser::parseOutputs() {
  for (int in = 0; in < o; in++) {
    outputs[in] = parseInt("\n");
  }
}

void AigParser::parseAagAnds() {
  for (int in = 0; in < a; in++) {
    int lhs  = parseInt(" ");
    int rhs0 = parseInt(" ");
    int rhs1 = parseInt("\n");
    ands[in] = std::make_tuple(lhs, rhs0, rhs1);
  }
}

void AigParser::parseAigAnds() {
  int delta0, delta1;
  int lhs = (i + l) * 2 + 2;
  for (int in = 0; in < a; in++) {
    delta0   = decode();
    delta1   = decode();
    int rhs0 = lhs - delta0;
    if (rhs0 < 0) {
      throw semantic_error(currLine, currChar,
                           format("Negative rhs0: %d", rhs0));
    }
    int rhs1 = rhs0 - delta1;
    if (rhs1 < 0) {
      throw semantic_error(currLine, currChar,
                           format("Negative rhs0: %d", rhs1));
    }
    ands[in] = std::make_tuple(lhs, rhs0, rhs1);
    lhs      = lhs + 2;
  }
}

void AigParser::parseSymbolTable() {
  int c, n;
  while (true) {
    try {
      c = parseChar();
    } catch (unexpected_eof& e) {
      return;
    }
    switch (c) {
    case 'i':
      n = parseInt(" ");
      if (n >= i)
        throw semantic_error(currLine, currChar,
                             "Input number greater than number of inputs");
      inputNames[n] = parseString("\n");
      break;
    case 'l':
      n = parseInt(" ");
      if (n >= l)
        throw semantic_error(currLine, currChar,
                             "Latch number greater than number of latches");
      latchNames[n] = parseString("\n");
      break;
    case 'o':
      n = parseInt(" ");
      if (n >= o)
        throw semantic_error(currLine, currChar,
                             "Output number greater than number of outputs");
      outputNames[n] = parseString("\n");
      break;
    case 'c':
      c = parseChar();
      if (c != '\n' && c != '\r')
        throw syntax_error(currLine, currChar);
      if (file.peek() != 0)
        designName = parseString("\n\0");
      else
        designName = "Unnamed";
      return;
    }
  }
}

void AigParser::parseAag() {
  parseAagHeader();
  parseAagInputs();
  parseAagLatches();
  parseOutputs();
  parseAagAnds();
  parseSymbolTable();
  createAig();
}

void AigParser::parseAig() {
  parseAigHeader();
  parseAigInputs();
  parseAigLatches();
  parseOutputs();
  parseAigAnds();
  parseSymbolTable();
  createAig();
}

void AigParser::createAig() {
  aig.setDesignName(this->designName);
  createConstant();
  createInputs();
  createLatches();
  createOutputs();
  createAnds();
  // connectAnds();
  connectAndsWithFanoutMap();
  connectLatches();
  connectOutputs();
}

void AigParser::createConstant() {
  // Node Data
  aig::NodeData nodeData;
  nodeData.id      = 0;
  nodeData.counter = 0;
  nodeData.type    = aig::NodeType::CONSTZERO;
  nodeData.level   = 0;
  // AIG Node
  aig::Graph& aigGraph = aig.getGraph();
  aig::GNode constNode;
  constNode = aigGraph.createNode(nodeData);
  aigGraph.addNode(constNode);
  aig.getNodes()[nodeData.id] = constNode;
}

void AigParser::createInputs() {
  aig::Graph& aigGraph = aig.getGraph();
  for (int in = 0; in < i; in++) {
    // Node Data
    aig::NodeData nodeData;
    nodeData.id      = inputs[in] / 2;
    nodeData.counter = 0;
    nodeData.type    = aig::NodeType::PI;
    nodeData.level   = 0;
    // AIG Node
    aig::GNode inputNode;
    inputNode = aigGraph.createNode(nodeData);
    aigGraph.addNode(inputNode);
    aig.getInputNodes().push_back(inputNode);
    aig.getNodes()[nodeData.id] = inputNode;
  }

  aig.setInputNames(this->inputNames);
}

void AigParser::createLatches() {
  aig::Graph& aigGraph = aig.getGraph();
  for (int in = 0; in < l; in++) {
    // Node Data
    aig::NodeData nodeData;
    nodeData.id      = (std::get<0>(latches[in]) / 2);
    nodeData.counter = 0;
    nodeData.type    = aig::NodeType::LATCH;
    // NodeData.initialValue = std::get<3>( latches[in] ); // FIXME
    // AIG Node
    aig::GNode latchNode;
    latchNode = aigGraph.createNode(nodeData);
    aigGraph.addNode(latchNode);
    aig.getLatchNodes().push_back(latchNode);
    aig.getNodes()[nodeData.id] = latchNode;
  }

  aig.setLatchNames(this->latchNames);
}

void AigParser::createOutputs() {
  aig::Graph& aigGraph = aig.getGraph();
  for (int in = 0; in < o; in++) {
    // Node Data
    aig::NodeData nodeData;
    nodeData.id      = m + in + 1;
    nodeData.counter = 0;
    nodeData.type    = aig::NodeType::PO;
    // AIG Node
    aig::GNode outputNode;
    outputNode = aigGraph.createNode(nodeData);
    aigGraph.addNode(outputNode);
    aig.getOutputNodes().push_back(outputNode);
    aig.getNodes()[nodeData.id] = outputNode;
  }

  aig.setOutputNames(this->outputNames);
}

void AigParser::createAnds() {
  aig::Graph& aigGraph = aig.getGraph();
  for (int in = 0; in < a; in++) {
    // Node Data
    aig::NodeData nodeData;
    nodeData.id = (std::get<0>(ands[in]) / 2);
    std::stringstream sName;
    nodeData.counter = 0;
    nodeData.type    = aig::NodeType::AND;
    // AIG Node
    aig::GNode andNode;
    andNode = aigGraph.createNode(nodeData);
    aigGraph.addNode(andNode);
    aig.getNodes()[nodeData.id] = andNode;
  }
}

void AigParser::connectLatches() {
  aig::Graph& aigGraph = aig.getGraph();
  for (int in = 0; in < l; in++) {
    int lhs                      = std::get<0>(latches[in]);
    aig::GNode latchNode         = aig.getNodes()[lhs / 2];
    aig::NodeData& latchNodeData = aigGraph.getData(latchNode);

    int rhs                      = std::get<1>(latches[in]);
    aig::GNode inputNode         = aig.getNodes()[rhs / 2];
    aig::NodeData& inputNodeData = aigGraph.getData(inputNode);

    aigGraph.getEdgeData(aigGraph.addEdge(inputNode, latchNode)) = !(rhs % 2);
    latchNodeData.level = 1 + inputNodeData.level;
  }
}

void AigParser::connectOutputs() {
  aig::Graph& aigGraph = aig.getGraph();
  for (int in = 0; in < o; in++) {
    aig::GNode outputNode = aig.getNodes()[m + in + 1];
    aig::NodeData& outputNodeData =
        aigGraph.getData(outputNode, galois::MethodFlag::WRITE);
    // outputNodeData.nFanin = 1;

    int rhs              = outputs[in];
    aig::GNode inputNode = aig.getNodes()[rhs / 2];
    aig::NodeData& inputNodeData =
        aigGraph.getData(inputNode, galois::MethodFlag::WRITE);
    inputNodeData.nFanout += 1;

    aigGraph.getEdgeData(aigGraph.addEdge(inputNode, outputNode)) = !(rhs % 2);
    outputNodeData.level = 1 + inputNodeData.level;
  }
}

void AigParser::connectAnds() {

  aig::Graph& aigGraph = aig.getGraph();

  // Each andDef is composed by three nodes A B C, A is the AND itself, B and C
  // are the two input nodes.
  for (auto andDef : this->ands) {

    int A              = std::get<0>(andDef);
    aig::GNode andNode = aig.getNodes()[A / 2];
    aig::NodeData& andData =
        aigGraph.getData(andNode, galois::MethodFlag::WRITE);
    // andData.nFanin = 2;

    int B              = std::get<1>(andDef);
    aig::GNode lhsNode = aig.getNodes()[B / 2];
    aig::NodeData& lhsData =
        aigGraph.getData(lhsNode, galois::MethodFlag::WRITE);
    lhsData.nFanout += 1;

    int C              = std::get<2>(andDef);
    aig::GNode rhsNode = aig.getNodes()[C / 2];
    aig::NodeData& rhsData =
        aigGraph.getData(rhsNode, galois::MethodFlag::WRITE);
    rhsData.nFanout += 1;

    aigGraph.getEdgeData(aigGraph.addMultiEdge(
        lhsNode, andNode, galois::MethodFlag::UNPROTECTED)) = !(B % 2);
    aigGraph.getEdgeData(aigGraph.addMultiEdge(
        rhsNode, andNode, galois::MethodFlag::UNPROTECTED)) = !(C % 2);

    andData.level = 1 + std::max(lhsData.level, rhsData.level);
  }
}

void AigParser::connectAndsWithFanoutMap() {

  aig::Graph& aigGraph = aig.getGraph();

  this->levelHistogram.resize(50000, 0); // FIXME
  this->levelHistogram[0] = this->i;

  // Each andDef is composed by three nodes A B C, A is the AND itself, B and C
  // are the two input nodes.
  for (auto andDef : this->ands) {

    int A              = std::get<0>(andDef);
    aig::GNode andNode = aig.getNodes()[A / 2];
    aig::NodeData& andData =
        aigGraph.getData(andNode, galois::MethodFlag::WRITE);
    // andData.nFanin = 2;

    int B              = std::get<1>(andDef);
    aig::GNode lhsNode = aig.getNodes()[B / 2];
    aig::NodeData& lhsData =
        aigGraph.getData(lhsNode, galois::MethodFlag::WRITE);
    bool lhsPol = !(B % 2);
    lhsData.nFanout += 1;

    int C              = std::get<2>(andDef);
    aig::GNode rhsNode = aig.getNodes()[C / 2];
    aig::NodeData& rhsData =
        aigGraph.getData(rhsNode, galois::MethodFlag::WRITE);
    bool rhsPol = !(C % 2);
    rhsData.nFanout += 1;

    aigGraph.getEdgeData(aigGraph.addMultiEdge(
        lhsNode, andNode, galois::MethodFlag::UNPROTECTED)) = lhsPol;
    aigGraph.getEdgeData(aigGraph.addMultiEdge(
        rhsNode, andNode, galois::MethodFlag::UNPROTECTED)) = rhsPol;
    aig.insertNodeInFanoutMap(andNode, lhsNode, rhsNode, lhsPol, rhsPol);

    andData.level = 1 + std::max(lhsData.level, rhsData.level);
    this->levelHistogram[andData.level] += 1;
  }

  int i = 0;
  while (i < 50000) {
    if (this->levelHistogram[i] == 0) {
      break;
    }
    i++;
  }
  this->levelHistogram.resize(i);
}

int AigParser::getI() { return i; }

int AigParser::getL() { return l; }

int AigParser::getO() { return o; }

int AigParser::getA() { return a; }

int AigParser::getE() {

  aig::Graph& aigGraph = aig.getGraph();
  int nEdges           = 0;

  for (auto node : aigGraph) {
    nEdges += std::distance(aigGraph.edge_begin(node), aigGraph.edge_end(node));
  }

  return nEdges;
}

std::vector<int>& AigParser::getLevelHistogram() {
  return this->levelHistogram;
}

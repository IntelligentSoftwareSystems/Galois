#ifndef GALOIS_EDA_ASYNC_TIMING_ARC_SET_H
#define GALOIS_EDA_ASYNC_TIMING_ARC_SET_H

#include "Verilog.h"
#include "CellLib.h"

#include <string>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <boost/functional/hash.hpp>

using AsyncTimingNode = std::pair<VerilogPin*, bool>;
using AsyncTimingArc = std::pair<AsyncTimingNode, AsyncTimingNode>;

struct AsyncTimingArcSet {
  std::unordered_set<AsyncTimingArc, boost::hash<AsyncTimingArc>> requiredArcs;
  std::unordered_set<AsyncTimingArc, boost::hash<AsyncTimingArc>> tickedArcs;

public:
  void print(std::ostream& os = std::cout);

  void addRequiredArc(VerilogPin* in, bool inRise, VerilogPin* out, bool outRise) {
    auto inNode = std::make_pair(in, inRise);
    auto outNode = std::make_pair(out, outRise);
    auto arc = std::make_pair(inNode, outNode);
    requiredArcs.insert(arc);
  }

  void addTickedArc(VerilogPin* in, bool inRise, VerilogPin* out, bool outRise) {
    auto inNode = std::make_pair(in, inRise);
    auto outNode = std::make_pair(out, outRise);
    auto arc = std::make_pair(inNode, outNode);
    tickedArcs.insert(arc);
  }

  bool isRequiredArc(VerilogPin* in, bool inRise, VerilogPin* out, bool outRise) {
    auto inNode = std::make_pair(in, inRise);
    auto outNode = std::make_pair(out, outRise);
    auto arc = std::make_pair(inNode, outNode);
    return requiredArcs.count(arc);
  }

  bool isTickedArc(VerilogPin* in, bool inRise, VerilogPin* out, bool outRise) {
    auto inNode = std::make_pair(in, inRise);
    auto outNode = std::make_pair(out, outRise);
    auto arc = std::make_pair(inNode, outNode);
    return tickedArcs.count(arc);
  }
};

struct AsyncTimingArcCollection {
  VerilogDesign& v;
  CellLib& lib;
  std::unordered_map<VerilogModule*, AsyncTimingArcSet*> modules;

private:
  void setup();
  void clear();

public:
  AsyncTimingArcCollection(VerilogDesign& design, CellLib& lib): v(design), lib(lib) { setup(); }
  ~AsyncTimingArcCollection() { clear(); }
  void parse(std::string inName);
  void print(std::ostream& os = std::cout);

  AsyncTimingArcSet* findArcSetForModule(VerilogModule* m) {
    auto it = modules.find(m);
    return (it == modules.end()) ? nullptr : it->second;
  }
};

struct AsyncTimingArcParser {
  std::vector<Token> tokens;
  std::vector<Token>::iterator curToken;
  AsyncTimingArcCollection& c;

private:
  void tokenizeFile(std::string inName);
  bool isEndOfTokenStream() { return curToken == tokens.end(); }
  void parseTimingArc(VerilogModule* m);

public:
  AsyncTimingArcParser(AsyncTimingArcCollection& c): c(c) {}
  void parse(std::string inName);
};

#endif // GALOIS_EDA_ASYNC_TIMING_ARC_SET_H

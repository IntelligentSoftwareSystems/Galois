#ifndef GALOIS_EDA_ASYNC_TIMING_ARC_SET_H
#define GALOIS_EDA_ASYNC_TIMING_ARC_SET_H

#include "Verilog.h"

#include <unordered_set>
#include <unordered_map>
#include <boost/functional/hash.hpp>

using AsyncTimingNode = std::pair<VerilogPin*, bool>;
using AsyncTimingArc = std::pair<AsyncTimingNode, AsyncTimingNode>;

struct AsyncTimingArcSet {
  std::unordered_set<AsyncTimingArc, boost::hash<AsyncTimingArc>> requiredArcs;
  std::unordered_set<AsyncTimingArc, boost::hash<AsyncTimingArc>> tickedArcs;

public:
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
  std::unordered_map<VerilogModule*, AsyncTimingArcSet*> modules;

private:
  void setup();
  void clear();

public:
  AsyncTimingArcCollection(VerilogDesign& design): v(design) { setup(); }
  ~AsyncTimingArcCollection() { clear(); }

  AsyncTimingArcSet* findArcSetForModule(VerilogModule* m) {
    auto it = modules.find(m);
    return (it == modules.end()) ? nullptr : it->second;
  }
};

#endif // GALOIS_EDA_ASYNC_TIMING_ARC_SET_H

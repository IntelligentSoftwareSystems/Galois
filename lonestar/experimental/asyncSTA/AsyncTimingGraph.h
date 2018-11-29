#ifndef GALOIS_EDA_ASYNC_TIMING_GRAPH_H
#define GALOIS_EDA_ASYNC_TIMING_GRAPH_H

#include "galois/Galois.h"
#include "galois/graphs/MorphGraph.h"

#include "CellLib.h"
#include "Verilog.h"
#include "TimingDefinition.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <boost/functional/hash.hpp>

enum TimingNodeType {
  PRIMARY_INPUT = 0,
  PRIMARY_OUTPUT,
  GATE_INPUT,
  GATE_OUTPUT,
  GATE_INTERNAL,
  GATE_INOUT,
  POWER_VDD,
  POWER_GND,
  DUMMY_POWER,
};

struct NodeTiming {
  CellPin* pin;
  MyFloat pinC;
  MyFloat wireC;
  MyFloat slew;
  MyFloat arrival;
  MyFloat required;
  MyFloat slack;
};

struct Node {
  bool isRise;
  TimingNodeType nType;
  size_t topoL;
  size_t revTopoL;
  bool flag;
  std::vector<NodeTiming> t;
  VerilogPin* pin;
};

struct EdgeTiming {
  WireLoad* wireLoad; // nullptr for timing arcs
  MyFloat delay;
};

struct Edge {
  std::vector<EdgeTiming> t;
  VerilogWire* wire;
  bool isConstraint;
};

struct AsyncTimingEngine;

struct AsyncTimingGraph {
public:
  // timing graph is directed and tracks incoming edges
  using Graph = galois::graphs::MorphGraph<Node, Edge, true, true>;
  using GNode = Graph::GraphNode;

public:
  VerilogModule& m;
  AsyncTimingEngine* engine;

  // internal graph
  Graph g;

  // internal mapping
  std::unordered_map<VerilogPin*, GNode[2]> nodeMap;

private:
  // utility functions
  size_t outDegree(GNode n);
  size_t inDegree(GNode n);

  void computeDriveC(GNode n);
  void computeExtremeSlew(GNode n, galois::PerIterAllocTy& alloc);
  void computeExtremeDelay(GNode n, galois::PerIterAllocTy& alloc);

  // graph construction
  void addPin(VerilogPin* p);
  void addGate(VerilogGate* g);
  void addWire(VerilogPin* p);
  void setWireLoad(WireLoad** wWL, WireLoad* wl);

  // debug printing
  std::string getNodeName(GNode n);

public:
  AsyncTimingGraph(VerilogModule& m, AsyncTimingEngine* engine): m(m), engine(engine) {}

  void construct();
  void initialize();
  void print(std::ostream& os = std::cout);
};

#endif // GALOIS_EDA_ASYNC_TIMING_GRAPH_H

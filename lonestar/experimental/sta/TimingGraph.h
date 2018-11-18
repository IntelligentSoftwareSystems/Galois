#ifndef GALOIS_EDA_TIMING_GRAPH_H
#define GALOIS_EDA_TIMING_GRAPH_H

#include "galois/Galois.h"
#include "galois/graphs/MorphGraph.h"

#include "CellLib.h"
#include "Verilog.h"
#include "Sdc.h"
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
  std::atomic<bool> flag;
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

struct TimingEngine;

struct TimingGraph {
public:
  // timing graph is directed and tracks incoming edges
  using Graph = galois::graphs::MorphGraph<Node, Edge, true, true>;
  using GNode = Graph::GraphNode;

public:
  VerilogModule& m;
  TimingEngine* engine;
  Clock* clk;

  // internal graph
  Graph g;

  // internal mapping
  std::unordered_map<VerilogPin*, GNode[2]> nodeMap;

  // forward/backward frontiers
  galois::InsertBag<GNode> fFront;
  galois::InsertBag<GNode> bFront;

private:
  // forward computation
  void computeDriveC(GNode n);
  void computeExtremeSlew(GNode n);
  bool computeDelayAndExactSlew(GNode n);
  void computeConstraint(GNode n);
  void initNodeForward(GNode n);
  void computeTopoL();

  // backward computation
  void computeSlack(GNode n);
  void initNodeBackward(GNode n);
  void computeRevTopoL();

  // initialization
  void initFlag(bool value);

  // graph construction
  void addPin(VerilogPin* p);
  void addGate(VerilogGate* g);
  void addWire(VerilogPin* p);
  void setWireLoad(WireLoad** wWL, WireLoad* wl);

  // debug printing
  std::string getNodeName(GNode n);

public:
  TimingGraph(VerilogModule& m, TimingEngine* engine): m(m), engine(engine) {}

  void construct();
  void initialize();
  void setConstraints(SDC& sdc);
  void computeForward();
  void computeBackward();
  void print(std::ostream& os = std::cout);
};

#endif // GALOIS_EDA_TIMING_GRAPH_H

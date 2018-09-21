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
  bool isOutput;
  bool isPrimary;
  bool isPowerNode;
  bool isClock;
  bool isDummy;
  bool isDrivingCell;
  size_t topoL;
  size_t revTopoL;
  std::atomic<bool> flag;
  std::vector<NodeTiming> t;
  VerilogPin* pin;

public:
  bool isPrimaryInput() {
    return (!isDummy && !isPowerNode && isPrimary && !isOutput);
  }

  bool isPrimaryOutput() {
    return (!isDummy && !isPowerNode && isPrimary && isOutput);
  }

  bool isGateInput() {
    return (!isDummy && !isPowerNode && !isPrimary && !isOutput);
  }

  bool isGateOutput() {
    return (!isDummy && !isPowerNode && !isPrimary && isOutput);
  }

  bool isRealPowerNode() {
    return (isPowerNode && !isDummy);
  }

  bool isRealDummy() {
    return (isDummy && !isPowerNode);
  }

  bool isSequentialInput() {
    return false;
  }

  bool isSequentialOutput() {
    return false;
  }

  bool isPseudoPrimaryInput() {
    return (isPrimaryInput() || isSequentialOutput());
  }

  bool isPseudoPrimaryOutput() {
    return (isPrimaryOutput() || isSequentialInput());
  }

  // TODO: add clock pins of all modules/gates
  bool isTimingSource() {
    return (isPseudoPrimaryInput() || isRealPowerNode());
  }
};

struct EdgeTiming {
  WireLoad* wireLoad; // nullptr for timing arcs
  MyFloat delay;
};

struct Edge {
  std::vector<EdgeTiming> t;
  VerilogWire* wire;
};

struct TimingGraph {
public:
  // timing graph is directed and tracks incoming edges
  using Graph = galois::graphs::MorphGraph<Node, Edge, true, true>;
  using GNode = Graph::GraphNode;

public:
  // external inputs from TimingEngine
  VerilogModule& m;
  std::vector<CellLib*>& libs;
  std::vector<TimingMode>& modes;
  bool isExactSlew;

  // internal graph
  Graph g;

  // internal mapping
  GNode dummySrc;
  GNode dummySink;
  std::unordered_map<VerilogPin*, GNode[2]> nodeMap;

private:
  // arrival time computation
  void computeDriveC(GNode n);
  void computeArrivalByWire(GNode n, Graph::in_edge_iterator ie);
  void computeArrivalByTimingArc(GNode n, Graph::in_edge_iterator ie, size_t k);

  // initialization
  void initFlag(bool value);
  void computeTopoL();
  void computeRevTopoL();

  // graph construction
  void addDummyNodes();
  void addPin(VerilogPin* p);
  void addGate(VerilogGate* g);
  void addWire(VerilogWire* w);

  // debug printing
  std::string getNodeName(GNode n);

public:
  TimingGraph(VerilogModule& m, std::vector<CellLib*>& libs, std::vector<TimingMode>& modes, bool isExactSlew): m(m), libs(libs), modes(modes), isExactSlew(isExactSlew) {}

  void construct();
  void initialize();
  void setConstraints(SDC& sdc);
  void computeArrivalTime();
  void print(std::ostream& os = std::cout);
};

#endif // GALOIS_EDA_TIMING_GRAPH_H

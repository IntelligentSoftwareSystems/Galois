#ifndef GALOIS_EDA_TIMING_GRAPH_H
#define GALOIS_EDA_TIMING_GRAPH_H

#include "galois/Galois.h"
#include "galois/graphs/MorphGraph.h"

#include "CellLib.h"
#include "Verilog.h"

#include <iostream>
#include <vector>
#include <unordered_map>

struct NodeTiming {
  CellPin* pin;
  float slew;
  float arrival;
  float required;
  float slack;
};

struct Node {
  bool isRise;
  bool isOutput;
  bool isPrimary;
  bool isDummy;
  size_t precondition;
  std::vector<NodeTiming> t;
  VerilogPin* pin;
};

struct EdgeTiming {
  WireLoad* wireLoad;
  float delay;
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
  VerilogModule& m;
  std::vector<CellLib*> libs;
  Graph g;
  GNode dummySrc;
  GNode dummySink;
  std::unordered_map<VerilogPin*, GNode[2]> nodeMap;

private:
  void addDummySrc();
  void addDummySink();
  void addPin(VerilogPin* p);
  void addGate(VerilogGate* g);
  void addWire(VerilogWire* w);
  std::string getNodeName(GNode n);

public:
  TimingGraph(VerilogModule& m, std::vector<CellLib*>& libs): m(m), libs(libs) {}

  void construct();
  void initialize();
  void print(std::ostream& os = std::cout);
};

#endif // GALOIS_EDA_TIMING_GRAPH_H

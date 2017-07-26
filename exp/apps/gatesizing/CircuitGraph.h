#include <unordered_map>
#include <utility>

#include "Galois/Graphs/Graph.h"

#include "Verilog.h"
#include "Sdc.h"

#ifndef GALOIS_CIRCUIT_GRAPH_H
#define GALOIS_CIRCUIT_GRAPH_H

struct TimingPowerInfo {
  float slew;
  float arrivalTime, requiredTime, slack;
  float internalPower, netPower;
};

struct Node {
  VerilogPin *pin;
  size_t precondition;
  bool isDummy, isPrimary, isOutput;
  float totalNetC, totalPinC;
  TimingPowerInfo rise, fall;
};

struct Edge {
  VerilogWire *wire;
  float riseDelay, fallDelay;
};

typedef Galois::Graph::FirstGraph<Node, Edge, true, true> Graph;
typedef Graph::GraphNode GNode;

extern Graph graph;
extern std::unordered_map<VerilogPin *, GNode> nodeMap;
extern GNode dummySrc, dummySink;

void constructCircuitGraph(Graph& g, VerilogModule& vModule);
void initializeCircuitGraph(Graph& g, SDC& sdc);
void printCircuitGraph(Graph& g);
std::pair<size_t, size_t> getCircuitGraphStatistics(Graph& g);

#endif // GALOIS_CIRCUIT_GRAPH_H


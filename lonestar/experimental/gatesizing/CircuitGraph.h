#include <unordered_map>
#include <utility>

#include "galois/graphs/MorphGraph.h"

#include "Verilog.h"

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

typedef galois::graphs::MorphGraph<Node, Edge, true, true> Graph;
typedef Graph::GraphNode GNode;

struct CircuitGraph {
  Graph g;
  std::unordered_map<VerilogPin *, GNode> nodeMap;
  GNode dummySrc, dummySink;

public:
  void construct(VerilogModule& vModule);
  void initialize();
  void print();
  std::pair<size_t, size_t> getStatistics();
};

#endif // GALOIS_CIRCUIT_GRAPH_H

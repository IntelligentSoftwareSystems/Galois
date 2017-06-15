#include <unordered_map>

#include "Galois/Graphs/Graph.h"

#include "Verilog.h"

#ifndef GALOIS_CIRCUIT_GRAPH_H
#define GALOIS_CIRCUIT_GRAPH_H

struct Node {
  VerilogPin *pin;
  float slew, capacitance;
  float arrivalTime, requiredTime, slack;
  float internalPower, netPower;
};

struct Edge {
  VerilogWire *wire;
  bool isRise;
};

typedef Galois::Graph::FirstGraph<Node, Edge, true, true> Graph;
typedef Graph::GraphNode GNode;

extern std::unordered_map<VerilogPin *, GNode> nodeMap;
extern GNode dummySrc, dummySink;

void constructCircuitGraph(Graph& g, VerilogModule& vModule);
void printCircuitGraph(Graph& g);

#endif // GALOIS_CIRCUIT_GRAPH_H


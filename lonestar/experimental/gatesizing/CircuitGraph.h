/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
  VerilogPin* pin;
  size_t precondition;
  bool isDummy, isPrimary, isOutput;
  float totalNetC, totalPinC;
  TimingPowerInfo rise, fall;
};

struct Edge {
  VerilogWire* wire;
  float riseDelay, fallDelay;
};

typedef galois::graphs::MorphGraph<Node, Edge, true, true> Graph;
typedef Graph::GraphNode GNode;

struct CircuitGraph {
  Graph g;
  std::unordered_map<VerilogPin*, GNode> nodeMap;
  GNode dummySrc, dummySink;

public:
  void construct(VerilogModule& vModule);
  void initialize();
  void print();
  std::pair<size_t, size_t> getStatistics();
};

#endif // GALOIS_CIRCUIT_GRAPH_H

/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#include <string>

//#include <vector>
//#include <unordered_map>
//typedef std::string KeyTy;
//typedef std::string ValTy;
//typedef std::unordered_map<KeyTy, ValTy> Attr;

struct Node {
  uint32_t label; // maximum of 32 node labels
  uint32_t id;
  uint64_t matched; // maximum of 64 nodes in the query graph
  // TODO: make matched a dynamic bitset
};

struct EdgeData {
  uint32_t label; // maximum  of 32 edge labels
  uint64_t timestamp; // range of timestamp is limited
  uint64_t matched; // maximum of 64 edges in the query graph
  EdgeData(uint32_t l, uint64_t t) : label(l), timestamp(t), matched(0) {}
};

struct MatchedNode {
  uint32_t id;
  //const char* label;
  const char* name;
};

struct MatchedEdge {
  uint64_t timestamp;
  const char* label;
  MatchedNode caused_by;
  MatchedNode acted_on;
};

struct EventLimit { // time-limit of consecutive events (inclusive)
  bool valid;
  uint64_t time; // inclusive
  EventLimit() : valid(false) {}
};

struct EventWindow { // time-span of all events (inclusive)
  bool valid;
  uint64_t startTime; // inclusive
  uint64_t endTime; // inclusive
  EventWindow() : valid(false) {}
};

typedef galois::graphs::LC_CSR_Graph<Node, EdgeData>::with_no_lockable<true>::type::with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode;

struct AttributedGraph {
  Graph graph;
  std::vector<std::string> nodeLabelNames; // maps ID to Name
  std::map<std::string, uint32_t> nodeLabelIDs; // maps Name to ID
  std::vector<std::string> edgeLabelNames; // maps ID to Name
  std::map<std::string, uint32_t> edgeLabelIDs; // maps Name to ID
  std::map<uint32_t, uint32_t> nodeIndices; // maps node UUID/ID to index/GraphNode
  std::vector<std::string> nodeNames; // cannot use LargeArray because serialize does not do deep-copy
  // custom attributes: maps from an attribute name to a vector that contains
  // the attribute for each node/edge
  std::map<std::string, std::vector<std::string>> nodeAttributes;
  std::map<std::string, std::vector<std::string>> edgeAttributes;
};

void runGraphSimulation(Graph& queryGraph, Graph& dataGraph, EventLimit limit, EventWindow window, bool queryNodeHasMoreThan2Edges);

void matchNodeWithRepeatedActions(Graph &graph, uint32_t nodeLabel, uint32_t action, EventWindow window);
void matchNodeWithTwoActions(Graph &graph, uint32_t nodeLabel, uint32_t action1, uint32_t dstNodeLabel1, uint32_t action2, uint32_t dstNodeLabel2, EventWindow window);

void matchNeighbors(Graph& graph, Graph::GraphNode node, uint32_t nodeLabel, uint32_t action, uint32_t neighborLabel, EventWindow window);

size_t countMatchedNodes(Graph& graph);
size_t countMatchedNeighbors(Graph& graph, Graph::GraphNode node);
size_t countMatchedEdges(Graph& graph);
size_t countMatchedNeighborEdges(Graph& graph, Graph::GraphNode node);


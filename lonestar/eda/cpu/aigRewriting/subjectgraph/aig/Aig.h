/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef AIG_AIG_H_
#define AIG_AIG_H_

#include "galois/Galois.h"
#include "galois/runtime/Statistics.h"
#include "galois/graphs/Morph_SepInOut_Graph.h"

#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <utility>
#include <unordered_map>

namespace andInverterGraph {

struct NodeData;

// Nodes hold a NodeData structure, edges hold a boolean value and are
// directional with InOut distinction
typedef galois::graphs::Morph_SepInOut_Graph<NodeData, bool, true, true> Graph;

typedef Graph::GraphNode GNode;

enum NodeType { AND, PI, PO, LATCH, CONSTZERO, CHOICE };

struct NodeData {
  NodeType type;		// AIG node type acording to the NodeType enum
  int id;						// AIG node identifier
  int level;				// AIG node level
  int counter;			// Counter used for controlling graph traversal
  int nFanout;			// AIG node fanout counter
	int nRefs;				// AIG node reference counter for tech mapping
	float reqTime;		// AIG node required time for tech mapping
	GNode choiceList; // Pointer to the first choice node, if it exists
	//bool isCompl;			// Mark is the output is complemented. It is used in choice nodes.

  NodeData() : level(0), counter(0), nFanout(0), nRefs(0), reqTime(std::numeric_limits<float>::max()), choiceList(nullptr) {} //, isCompl(false) {}
};

class Aig {

private:
  Graph graph;
  std::string designName;
  std::vector<GNode> inputNodes;
  std::vector<GNode> latchNodes;
  std::vector<GNode> outputNodes;
  std::vector<std::string> inputNames;
  std::vector<std::string> latchNames;
  std::vector<std::string> outputNames;
  std::vector<GNode> nodes;
  std::vector<std::pair<int, int>> nodesTravId;
  std::vector<std::unordered_multimap<unsigned, GNode>> nodesFanoutMap;
	size_t idCounter;
	float expansionRate;


  void topologicalSortAll(GNode node, std::vector<bool>& visited, std::stack<GNode>& stack);
  void topologicalSortAnds(GNode node, std::vector<bool>& visited, std::stack<GNode>& stack);
	void genericTopologicalSortAnds(GNode node, std::vector<bool>& visited, std::vector<GNode>& sortedNodes);

public:
  
	Aig();
	Aig(float expansionRate);
  virtual ~Aig();

	void resize(int m, int i, int l, int o, bool hasSymbols);
	void resizeNodeVectors(int size);
	void expandNodeVectors(size_t extraSize);
	size_t getNextId();
	GNode createAND(GNode lhsAnd, GNode rhsAnd, bool lhsAndPol, bool rhsAndPol);

	void insertNodeInFanoutMap(GNode andNode, GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol);
  void removeNodeInFanoutMap(GNode removedNode, GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol);
  GNode lookupNodeInFanoutMap(GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol);
  unsigned makeAndHashKey(int lhsId, int rhsId, bool lhsPol, bool rhsPol);

 	void registerTravId(int nodeId, int threadId, int travId);
  bool lookupTravId(int nodeId, int threadId, int travId);

	std::vector<std::pair<int, int>>& getNodesTravId();
  std::unordered_multimap<unsigned, GNode>& getFanoutMap(int nodeId);
  std::vector<std::unordered_multimap<unsigned, GNode>>& getNodesFanoutMap();
  Graph& getGraph();
  std::vector<GNode>& getNodes();
  std::vector<GNode>& getInputNodes();
  std::vector<GNode>& getLatchNodes();
  std::vector<GNode>& getOutputNodes();
  std::vector<std::string>& getInputNames();
  void setInputNames(std::vector<std::string> inputNames);
  std::vector<std::string>& getLatchNames();
  void setLatchNames(std::vector<std::string> latchNames);
  std::vector<std::string>& getOutputNames();
  void setOutputNames(std::vector<std::string> outputNames);
  GNode getConstZero();
  int getNumInputs();
  int getNumLatches();
  int getNumOutputs();
  int getNumAnds();
  int getDepth();
  std::string getDesignName();
  void setDesignName(std::string designName);

	//bool isGNodeComplemented(GNode node);
	//GNode makeGNodeRegular(GNode node);
	//GNode makeGNodeComplemented(GNode node);

  void resetAndIds();
  void resetAndPIsIds();
  void resetAndPOsIds();
  void resetAllIds();

  void resetAllNodeCounters();

  void computeTopologicalSortForAll(std::stack<GNode>& stack);
  void computeTopologicalSortForAnds(std::stack<GNode>& stack);
	void computeGenericTopologicalSortForAnds(std::vector<GNode>& sortedNodes);

  void writeDot(std::string path, std::string dotText);
  std::string toDot();
};

} // namespace andInverterGraph

namespace aig = andInverterGraph;

#endif /* AIG_H_ */

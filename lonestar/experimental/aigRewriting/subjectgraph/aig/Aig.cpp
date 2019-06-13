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

#include "Aig.h"

#include <fstream>
#include <unordered_map>

namespace andInverterGraph {

Aig::Aig() {
	this->idCounter = 0;
	this->expansionRate = 0.2;
}

Aig::Aig(float expansionRate) {
	this->idCounter = 0;
	this->expansionRate = expansionRate;
}

Aig::~Aig() { }

void Aig::resize(int m, int i, int l, int o, int a, bool hasSymbols) {
	this->inputNodes.resize(i);
  this->latchNodes.resize(l);
  this->outputNodes.resize(o);
 	
	if ( hasSymbols ) {
		this->inputNames.resize(i);
 		this->latchNames.resize(l);
 		this->outputNames.resize(o);
	}

  int nNodes = m + o + 1;
 	this->nodes.resize(nNodes);
  this->nodesTravId.resize(nNodes);
  this->nodesFanoutMap.resize(nNodes);
	this->idCounter = nNodes;
}

void Aig::resizeNodeVectors(int size) {
	this->idCounter = size;
  this->nodes.resize(size);
  this->nodesTravId.resize(size);
  this->nodesFanoutMap.resize(size);
}

void Aig::expandNodeVectors(int extraSize) {
	this->idCounter = this->nodes.size();
	int nNodes = this->idCounter + extraSize;
  this->nodes.resize(nNodes);
  this->nodesTravId.resize(nNodes);
  this->nodesFanoutMap.resize(nNodes);
}

int Aig::getNextId() {
	if (this->idCounter == nodes.size()) {
		int extraSize = (int) (this->nodes.size() * this->expansionRate);
		expandNodeVectors(extraSize);
	}
	int nextId = this->idCounter++;
	return nextId;
}

GNode Aig::createAND(GNode lhsAnd, GNode rhsAnd, bool lhsAndPol, bool rhsAndPol) {

  NodeData& lhsAndData = this->graph.getData(lhsAnd, galois::MethodFlag::READ);
  NodeData& rhsAndData = this->graph.getData(rhsAnd, galois::MethodFlag::READ);
  NodeData newAndData;

  newAndData.id      = getNextId();
  newAndData.type    = aig::NodeType::AND;
  newAndData.level   = 1 + std::max(lhsAndData.level, rhsAndData.level);
  newAndData.counter = 0;
	newAndData.nFanout = 0;

 	GNode newAnd = this->graph.createNode(newAndData);
  this->graph.addNode(newAnd);

  this->graph.getEdgeData(graph.addMultiEdge(lhsAnd, newAnd, galois::MethodFlag::WRITE)) = lhsAndPol;
  this->graph.getEdgeData(graph.addMultiEdge(rhsAnd, newAnd, galois::MethodFlag::WRITE)) = rhsAndPol;
  lhsAndData.nFanout++;
  rhsAndData.nFanout++;

  // int faninSize = std::distance( aigGraph.in_edge_begin( newAnd ),
  // aigGraph.in_edge_begin( newAnd ) ); assert( faninSize == 2 );

  this->nodes[newAndData.id] = newAnd;
  this->insertNodeInFanoutMap(newAnd, lhsAnd, rhsAnd, lhsAndPol, rhsAndPol);

  return newAnd;
}

void Aig::insertNodeInFanoutMap(GNode andNode, GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol) {

  NodeData& lhsNodeData = this->graph.getData(lhsNode, galois::MethodFlag::READ);
  NodeData& rhsNodeData = this->graph.getData(rhsNode, galois::MethodFlag::READ);

  unsigned key = makeAndHashKey(lhsNode, rhsNode, lhsNodeData.id, rhsNodeData.id, lhsPol, rhsPol);

  if (lhsNodeData.id < rhsNodeData.id) {
    this->nodesFanoutMap[lhsNodeData.id].emplace(key, andNode);
  } else {
    this->nodesFanoutMap[rhsNodeData.id].emplace(key, andNode);
  }
}

void Aig::removeNodeInFanoutMap(GNode removedNode, GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol) {

  GNode lhsInNode;
  GNode rhsInNode;
  bool lhsInNodePol;
  bool rhsInNodePol;
  int smallestId;

  NodeData& lhsNodeData = this->graph.getData(lhsNode, galois::MethodFlag::READ);
  NodeData& rhsNodeData = this->graph.getData(rhsNode, galois::MethodFlag::READ);

  unsigned key = makeAndHashKey(lhsNode, rhsNode, lhsNodeData.id, rhsNodeData.id, lhsPol, rhsPol);

  if (lhsNodeData.id < rhsNodeData.id) {
    smallestId = lhsNodeData.id;
  } else {
    smallestId = rhsNodeData.id;
  }

  std::unordered_multimap<unsigned, GNode>& fanoutMap = this->nodesFanoutMap[smallestId];
  auto range = fanoutMap.equal_range(key);

  for (auto it = range.first; it != range.second;) {

    GNode fanoutNode = it->second;
    NodeData& fanoutNodeData = this->graph.getData(fanoutNode, galois::MethodFlag::READ);

    if (fanoutNodeData.type != NodeType::AND) {
      it++;
      continue;
    }

    auto inEdge  = this->graph.in_edge_begin(fanoutNode);
    lhsInNode    = this->graph.getEdgeDst(inEdge);
    lhsInNodePol = this->graph.getEdgeData(inEdge);

    if (lhsInNode == lhsNode) {
      inEdge++;
      rhsInNode    = this->graph.getEdgeDst(inEdge);
      rhsInNodePol = this->graph.getEdgeData(inEdge);
    } else {
      rhsInNode    = lhsInNode;
      rhsInNodePol = lhsInNodePol;
      inEdge++;
      lhsInNode    = this->graph.getEdgeDst(inEdge);
      lhsInNodePol = this->graph.getEdgeData(inEdge);
    }

    if ((lhsInNode == lhsNode) && (lhsInNodePol == lhsPol) &&
        (rhsInNode == rhsNode) && (rhsInNodePol == rhsPol) &&
        (fanoutNode == removedNode)) {
      it = fanoutMap.erase(it);
    } else {
      it++;
    }
  }
}

GNode Aig::lookupNodeInFanoutMap(GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol) {

  GNode lhsInNode;
  GNode rhsInNode;
  bool lhsInNodePol;
  bool rhsInNodePol;
  int smallestId;

  NodeData& lhsNodeData = this->graph.getData(lhsNode, galois::MethodFlag::READ);
  NodeData& rhsNodeData = this->graph.getData(rhsNode, galois::MethodFlag::READ);

  unsigned key = makeAndHashKey(lhsNode, rhsNode, lhsNodeData.id, rhsNodeData.id, lhsPol, rhsPol);

  if (lhsNodeData.id < rhsNodeData.id) {
    smallestId = lhsNodeData.id;
  } else {
    smallestId = rhsNodeData.id;
  }

  std::unordered_multimap<unsigned, GNode>& fanoutMap = this->nodesFanoutMap[smallestId];
  auto range = fanoutMap.equal_range(key);

  for (auto it = range.first; it != range.second; it++) {

    GNode fanoutNode = it->second;
    NodeData& fanoutNodeData = this->graph.getData(fanoutNode, galois::MethodFlag::READ);

    if (fanoutNodeData.type != NodeType::AND) {
      continue;
    }

    auto inEdge  = this->graph.in_edge_begin(fanoutNode);
    lhsInNode    = this->graph.getEdgeDst(inEdge);
    lhsInNodePol = this->graph.getEdgeData(inEdge);

    if (lhsInNode == lhsNode) {
      inEdge++;
      rhsInNode    = this->graph.getEdgeDst(inEdge);
      rhsInNodePol = this->graph.getEdgeData(inEdge);
    } else {
      rhsInNode    = lhsInNode;
      rhsInNodePol = lhsInNodePol;
      inEdge++;
      lhsInNode    = this->graph.getEdgeDst(inEdge);
      lhsInNodePol = this->graph.getEdgeData(inEdge);
      assert(lhsInNode == lhsNode);
    }
    if ((lhsInNode == lhsNode) && (lhsInNodePol == lhsPol) &&
        (rhsInNode == rhsNode) && (rhsInNodePol == rhsPol)) {
      return fanoutNode;
    }
  }

  return nullptr;
}

unsigned Aig::makeAndHashKey(GNode lhsNode, GNode rhsNode, int lhsId, int rhsId, bool lhsPol, bool rhsPol) {

  unsigned key = 0;

  if (lhsId < rhsId) {
    key ^= lhsId * 7937;
    key ^= rhsId * 2971;
    key ^= lhsPol ? 911 : 0;
    key ^= rhsPol ? 353 : 0;
  } else {
    key ^= rhsId * 7937;
    key ^= lhsId * 2971;
    key ^= rhsPol ? 911 : 0;
    key ^= lhsPol ? 353 : 0;
  }

  return key;
}

void Aig::registerTravId(int nodeId, int threadId, int travId) {
  this->nodesTravId[nodeId].first  = threadId;
  this->nodesTravId[nodeId].second = travId;
}

bool Aig::lookupTravId(int nodeId, int threadId, int travId) {
  if ((this->nodesTravId[nodeId].first == threadId) && (this->nodesTravId[nodeId].second == travId)) {
    return true;
  } else {
    return false;
  }
}

std::vector<std::pair<int, int>>& Aig::getNodesTravId() {
  return this->nodesTravId;
}

std::unordered_multimap<unsigned, GNode>& Aig::getFanoutMap(int nodeId) {
  return this->nodesFanoutMap[nodeId];
}

std::vector<std::unordered_multimap<unsigned, GNode>>& Aig::getNodesFanoutMap() {
  return this->nodesFanoutMap;
}

Graph& Aig::getGraph() { return this->graph; }

std::vector<GNode>& Aig::getNodes() { return this->nodes; }

std::vector<GNode>& Aig::getInputNodes() { return this->inputNodes; }

std::vector<GNode>& Aig::getLatchNodes() { return this->latchNodes; }

std::vector<GNode>& Aig::getOutputNodes() { return this->outputNodes; }

GNode Aig::getConstZero() { return this->nodes[0]; }

int Aig::getNumInputs() { return this->inputNodes.size(); }

int Aig::getNumLatches() { return this->latchNodes.size(); }

int Aig::getNumOutputs() { return this->outputNodes.size(); }

int Aig::getNumAnds() {
  int nNodes = std::distance(this->graph.begin(), this->graph.end());
  return (nNodes - (getNumInputs() + getNumLatches() + getNumOutputs() + 1)); 
	// +1 is to disconsider the constant node.
}

int Aig::getDepth() {

  resetAllIds();

  int max = -1;

  for (auto po : this->outputNodes) {
    NodeData& poData = this->graph.getData(po, galois::MethodFlag::READ);
    if (max < poData.level) {
      max = poData.level;
    }
  }

  assert(max > -1);

  return max;
}

std::vector<std::string>& Aig::getInputNames() { return this->inputNames; }

void Aig::setInputNames(std::vector<std::string> inputNames) {
  this->inputNames = inputNames;
}

std::vector<std::string>& Aig::getLatchNames() { return this->latchNames; }

void Aig::setLatchNames(std::vector<std::string> latchNames) {
  this->latchNames = latchNames;
}

std::vector<std::string>& Aig::getOutputNames() { return this->outputNames; }

void Aig::setOutputNames(std::vector<std::string> outputNames) {
  this->outputNames = outputNames;
}

std::string Aig::getDesignName() { return this->designName; }

void Aig::setDesignName(std::string designName) {
  this->designName = designName;
}

/*
bool Aig::isGNodeComplemented(GNode node) {
  return (bool)(((unsigned long int)node) & 01u);
}

GNode Aig::makeGNodeRegular(GNode node) {
  return (GNode)((unsigned long int)(node) & ~01u);
}

GNode Aig::makeGNodeComplemented(GNode node) {
  return (GNode)((unsigned long int)(node) ^ 01u);
}
*/

// ########## ALGORITHMES ######## ///

struct ResetNodeCounters {	
  aig::Graph& aigGraph;

  ResetNodeCounters(aig::Graph& aigGraph) : aigGraph(aigGraph) {}

  void operator()(aig::GNode node) {
		aig::NodeData & nodeData = aigGraph.getData( node, galois::MethodFlag::WRITE );
		nodeData.counter = 0;
  }
};

void Aig::resetAllNodeCounters() {
	galois::do_all( galois::iterate( graph ), ResetNodeCounters{ graph }, galois::steal() );
}

void Aig::resetAndIds() {

  std::stack<GNode> stack;

  computeTopologicalSortForAnds(stack);

  int currentId = this->getNumInputs() + this->getNumLatches() + 1;

  while (!stack.empty()) {

    GNode node = stack.top();
    stack.pop();
    NodeData& nodeData       = graph.getData(node, galois::MethodFlag::WRITE);
    nodeData.id              = currentId++;
		nodeData.counter = 0;
    this->nodes[nodeData.id] = node;
  }

  // std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::resetAndPIsIds() {

  std::stack<GNode> stack;

  computeTopologicalSortForAnds(stack);

  int currentId = 1;

  for (GNode pi : this->inputNodes) {
    NodeData& piData = this->graph.getData(pi, galois::MethodFlag::WRITE);
    piData.id        = currentId++;
    this->nodes[piData.id] = pi;
  }

  while (!stack.empty()) {
    GNode node = stack.top();
    stack.pop();
    NodeData& nodeData       = graph.getData(node, galois::MethodFlag::WRITE);
    nodeData.id              = currentId++;
    this->nodes[nodeData.id] = node;
  }

  // std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::resetAndPOsIds() {

  std::stack<GNode> stack;

  computeTopologicalSortForAnds(stack);

  int currentId = this->getNumInputs() + this->getNumLatches() + 1;

  while (!stack.empty()) {
    GNode node = stack.top();
    stack.pop();
    NodeData& nodeData       = graph.getData(node, galois::MethodFlag::WRITE);
    nodeData.id              = currentId++;
    this->nodes[nodeData.id] = node;
  }

  for (GNode po : this->outputNodes) {
    NodeData& poData = this->graph.getData(po, galois::MethodFlag::WRITE);
    poData.id        = currentId++;
    this->nodes[poData.id] = po;
  }

  // std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::resetAllIds() {

  std::stack<GNode> stack;

  computeTopologicalSortForAnds(stack);

  int currentId = 1;

  for (GNode pi : this->inputNodes) {
    NodeData& piData = this->graph.getData(pi, galois::MethodFlag::WRITE);
    piData.id        = currentId++;
    piData.level     = 0;
    this->nodes[piData.id] = pi;
  } 

	for (GNode latch : this->latchNodes) {
    NodeData& latchData = this->graph.getData(latch, galois::MethodFlag::WRITE);
    latchData.id        = currentId++;
    latchData.level     = 0; // FIXME
    this->nodes[latchData.id] = latch;
  }

  while (!stack.empty()) {
    GNode node = stack.top();
    stack.pop();
    NodeData& nodeData = graph.getData(node, galois::MethodFlag::WRITE);
    nodeData.id        = currentId++;

    auto inEdge      = this->graph.in_edge_begin(node);
    GNode lhsNode    = this->graph.getEdgeDst(inEdge);
    NodeData lhsData = this->graph.getData(lhsNode, galois::MethodFlag::READ);
    inEdge++;
    GNode rhsNode    = this->graph.getEdgeDst(inEdge);
    NodeData rhsData = this->graph.getData(rhsNode, galois::MethodFlag::READ);

    nodeData.level = 1 + std::max(lhsData.level, rhsData.level);

    this->nodes[nodeData.id] = node;
  }

  for (GNode po : this->outputNodes) {
    NodeData& poData = this->graph.getData(po, galois::MethodFlag::WRITE);
    poData.id        = currentId++;

    auto inEdge     = this->graph.in_edge_begin(po);
    GNode inNode    = this->graph.getEdgeDst(inEdge);
    NodeData inData = this->graph.getData(inNode, galois::MethodFlag::READ);

    poData.level = inData.level;
    this->nodes[poData.id] = po;
  }

  // std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::computeTopologicalSortForAll(std::stack<GNode>& stack) {

  int size = this->nodes.size();
  std::vector<bool> visited(size, false);

  for (GNode pi : this->inputNodes) {
    for (auto outEdge : this->graph.out_edges(pi)) {

      GNode node         = this->graph.getEdgeDst(outEdge);
      NodeData& nodeData = this->graph.getData(node, galois::MethodFlag::READ);

      if (!visited[nodeData.id]) {
        topologicalSortAll(node, visited, stack);
      }
    }

    stack.push(pi);
  }

	for (GNode latch : this->latchNodes) {
    for (auto outEdge : this->graph.out_edges(latch)) {

      GNode node         = this->graph.getEdgeDst(outEdge);
      NodeData& nodeData = this->graph.getData(node, galois::MethodFlag::READ);

      if (!visited[nodeData.id]) {
        topologicalSortAll(node, visited, stack);
      }
    }

    stack.push(latch);
  }

}

void Aig::topologicalSortAll(GNode node, std::vector<bool>& visited, std::stack<GNode>& stack) {

  NodeData& nodeData   = graph.getData(node, galois::MethodFlag::READ);
  visited[nodeData.id] = true;

  for (auto outEdge : this->graph.out_edges(node)) {

    GNode nextNode = this->graph.getEdgeDst(outEdge);
    NodeData& nextNodeData =
        this->graph.getData(nextNode, galois::MethodFlag::READ);

    if (!visited[nextNodeData.id]) {
      topologicalSortAll(nextNode, visited, stack);
    }
  }

  stack.push(node);
}

void Aig::computeTopologicalSortForAnds(std::stack<GNode>& stack) {

  int size = this->nodes.size();
  std::vector<bool> visited(size, false);

  for (GNode pi : this->inputNodes) {
    for (auto outEdge : this->graph.out_edges(pi)) {

      GNode node         = this->graph.getEdgeDst(outEdge);
      NodeData& nodeData = this->graph.getData(node, galois::MethodFlag::READ);

      if ((!visited[nodeData.id]) && (nodeData.type == NodeType::AND)) {
        topologicalSortAnds(node, visited, stack);
      }
    }
  }

	for (GNode latch : this->latchNodes) {
    for (auto outEdge : this->graph.out_edges(latch)) {

      GNode node         = this->graph.getEdgeDst(outEdge);
      NodeData& nodeData = this->graph.getData(node, galois::MethodFlag::READ);

      if ((!visited[nodeData.id]) && (nodeData.type == NodeType::AND)) {
        topologicalSortAnds(node, visited, stack);
      }
    }
  }
}

void Aig::topologicalSortAnds(GNode node, std::vector<bool>& visited, std::stack<GNode>& stack) {

  NodeData& nodeData   = graph.getData(node, galois::MethodFlag::READ);
  visited[nodeData.id] = true;

  for (auto outEdge : this->graph.out_edges(node)) {

    GNode nextNode = this->graph.getEdgeDst(outEdge);
    NodeData& nextNodeData =
        this->graph.getData(nextNode, galois::MethodFlag::READ);

    if ((!visited[nextNodeData.id]) && (nextNodeData.type == NodeType::AND)) {
      topologicalSortAnds(nextNode, visited, stack);
    }
  }

  stack.push(node);
}

void Aig::computeGenericTopologicalSortForAnds(std::vector<GNode>& sortedNodes) {

  int size = this->nodes.size();
  std::vector<bool> visited(size, false);

  for (GNode pi : this->inputNodes) {
    for (auto outEdge : this->graph.out_edges(pi)) {

      GNode node         = this->graph.getEdgeDst(outEdge);
      NodeData& nodeData = this->graph.getData(node, galois::MethodFlag::UNPROTECTED);

      if ((!visited[nodeData.id]) && (nodeData.type == NodeType::AND)) {
        genericTopologicalSortAnds(node, visited, sortedNodes);
      }
    }
  }

	for (GNode latch : this->latchNodes) {
    for (auto outEdge : this->graph.out_edges(latch)) {

      GNode node         = this->graph.getEdgeDst(outEdge);
      NodeData& nodeData = this->graph.getData(node, galois::MethodFlag::UNPROTECTED);

      if ((!visited[nodeData.id]) && (nodeData.type == NodeType::AND)) {
        genericTopologicalSortAnds(node, visited, sortedNodes);
      }
    }
  }
}

void Aig::genericTopologicalSortAnds(GNode node, std::vector<bool>& visited, std::vector<GNode>& sortedNodes) {

  NodeData& nodeData   = graph.getData(node, galois::MethodFlag::UNPROTECTED);
  visited[nodeData.id] = true;

  for (auto outEdge : this->graph.out_edges(node)) {

    GNode nextNode = this->graph.getEdgeDst(outEdge);
    NodeData& nextNodeData =
        this->graph.getData(nextNode, galois::MethodFlag::UNPROTECTED);

    if ((!visited[nextNodeData.id]) && (nextNodeData.type == NodeType::AND)) {
      genericTopologicalSortAnds(nextNode, visited, sortedNodes);
    }
  }

  sortedNodes.push_back(node);
}

std::string Aig::toDot() {

  // Preprocess PI, LATCH and PO names
  std::unordered_map<int, std::string> piNames;
  for (int i = 0; i < this->inputNodes.size(); i++) {
    aig::NodeData& nodeData = graph.getData(this->inputNodes[i], galois::MethodFlag::READ);
    piNames.insert(std::make_pair(nodeData.id, this->inputNames[i]));
  }

 	std::unordered_map<int, std::string> latchNames;
  for (int i = 0; i < this->latchNodes.size(); i++) {
    aig::NodeData& nodeData = graph.getData(this->latchNodes[i], galois::MethodFlag::READ);
    latchNames.insert(std::make_pair(nodeData.id, this->latchNames[i]));
  }

  std::unordered_map<int, std::string> poNames;
  for (int i = 0; i < this->outputNodes.size(); i++) {
    aig::NodeData& nodeData = graph.getData(this->outputNodes[i], galois::MethodFlag::READ);
    poNames.insert(std::make_pair(nodeData.id, this->outputNames[i]));
  }

  std::stringstream dot, inputs, latches, outputs, ands, edges;

  for (auto node : this->graph) {

    aig::NodeData& nodeData = graph.getData(node, galois::MethodFlag::READ);

    // Write Edges
    for (auto edge : graph.in_edges(node)) {
      aig::GNode dstNode     = graph.getEdgeDst(edge);
      aig::NodeData& dstData = graph.getData(dstNode, galois::MethodFlag::READ);
      bool polarity = graph.getEdgeData(edge, galois::MethodFlag::READ);

      std::string nodeName, dstName;

      if (nodeData.type == NodeType::PI) {
        nodeName = piNames[nodeData.id];
      } else {
				if (nodeData.type == NodeType::LATCH) {
        	nodeName = latchNames[nodeData.id];
				} else {
       		if (nodeData.type == NodeType::PO) {
          	nodeName = poNames[nodeData.id];
        	} else {
          	nodeName = std::to_string(nodeData.id);
        	}
				}
      }

      if (dstData.type == NodeType::PI) {
        dstName = piNames[dstData.id];
      } else {
				if (dstData.type == NodeType::LATCH) {
        	dstName = latchNames[dstData.id];
				} else {
        	if (dstData.type == NodeType::PO) {
          	dstName = poNames[dstData.id];
        	} else {
          	dstName = std::to_string(dstData.id);
        	}
				}
      }

      edges << "\"" << dstName << "\" -> \"" << nodeName << "\"";

      if (polarity) {
        edges << " [penwidth = 3, color=blue]" << std::endl;
      } else {
        edges << " [penwidth = 3, color=red, style=dashed]" << std::endl;
      }
    }

    if (nodeData.type == NodeType::PI) {
      inputs << "\"" << piNames[nodeData.id] << "\"";
      inputs << " [shape=circle, height=1, width=1, penwidth=5 style=filled, "
                "fillcolor=\"#ff8080\", fontsize=20]"
             << std::endl;
      continue;
    }

		if (nodeData.type == NodeType::LATCH) {
      latches << "\"" << latchNames[nodeData.id] << "\"";
      latches << " [shape=square, height=1, width=1, penwidth=5 style=filled, "
                "fillcolor=\"#ff8080\", fontsize=20]"
             << std::endl;
      continue;
    }

    if (nodeData.type == NodeType::PO) {
      outputs << "\"" << poNames[nodeData.id] << "\"";
      outputs << " [shape=circle, height=1, width=1, penwidth=5 style=filled, "
                 "fillcolor=\"#008080\", fontsize=20]"
              << std::endl;
      continue;
    }

    if (nodeData.type == NodeType::AND) {
      ands << "\"" << nodeData.id << "\"";
      ands << " [shape=circle, height=1, width=1, penwidth=5 style=filled, "
              "fillcolor=\"#ffffff\", fontsize=20]"
           << std::endl;
    }
  }

  dot << "digraph aig {" << std::endl;
  dot << "ranksep=1.5;" << std::endl;
  dot << "nodesep=1.5;" << std::endl;
  dot << inputs.str();
  dot << latches.str();
  dot << ands.str();
  dot << outputs.str();
  dot << edges.str();
  dot << "{ rank=source;";
  for (GNode node : this->inputNodes) {
    aig::NodeData& nodeData = graph.getData(node, galois::MethodFlag::READ);
    dot << " \"" << piNames[nodeData.id] << "\"";
  }

  for (GNode node : this->latchNodes) {
    aig::NodeData& nodeData = graph.getData(node, galois::MethodFlag::READ);
    dot << " \"" << latchNames[nodeData.id] << "\"";
  }
  dot << " }" << std::endl;

  dot << "{ rank=sink;";
  for (GNode node : this->outputNodes) {
    aig::NodeData& nodeData = graph.getData(node, galois::MethodFlag::READ);
    dot << " \"" << poNames[nodeData.id] << "\"";
  }
  dot << " }" << std::endl;

  dot << "rankdir=\"BT\"" << std::endl;
  dot << "}" << std::endl;

  return dot.str();
}

void Aig::writeDot(std::string path, std::string dotText) {

  std::ofstream dotFile;
  dotFile.open(path);
  dotFile << dotText;
  dotFile.close();
}

} // namespace andInverterGraph

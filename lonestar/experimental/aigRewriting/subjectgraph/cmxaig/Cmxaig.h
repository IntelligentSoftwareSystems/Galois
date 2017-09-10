/*
 * Cmxaig.h
 *
 *  Created on: 09/04/2015
 *      Author: possani
 */

#ifndef CMXAIG_H_
#define CMXAIG_H_

#include "nodes/Node.h"
#include "nodes/MuxNode.h"
#include "nodes/XorNode.h"
#include "nodes/AndNode.h"
#include "nodes/FunctionNode.h"
#include "nodes/ChoiceNode.h"
#include "nodes/InputNode.h"
#include "nodes/OutputNode.h"
#include "../functional/FunctionHandler.h"
#include "../functional/BitVectorPool.h"

#include <vector>
#include <set>
#include <utility>

namespace SubjectGraph {

typedef Functional::word word;
typedef std::vector<Node*> NodeVector;

typedef struct nodeInfo {
	Node* node;
	word* function;
	bool outPolarity;
} NodeInfo;

class Cmxaig {

	int nVars;
	int nWords;
	Functional::BitVectorPool & functionPool;

	NodeVector inputs;
	NodeVector outputs;
	unsigned nodeCounter;
	unsigned muxCounter;
	unsigned xorCounter;
	unsigned andCounter;
	unsigned choiceCounter;
	unsigned inputCounter;
	unsigned outputCounter;

	unsigned getNextNodeID();
	unsigned getNextMuxID();
	unsigned getNextXorID();
	unsigned getNextAndID();
	unsigned getNextChoiceID();
	unsigned getNextInputID();
	unsigned getNextOutputID();

	void elementsToDot(Node& currentNode, std::set<unsigned>& visited, std::stringstream& dotNodes, std::stringstream& dotEdges);

public:

	Cmxaig(int nVars, int nWords, Functional::BitVectorPool & functionPool);

	Cmxaig(int nVars, int nWords, Functional::BitVectorPool & functionPool, NodeVector& inputs, NodeVector& outputs, unsigned nodeCounter, unsigned muxCounter, unsigned xorCounter, unsigned andCounter, unsigned choiceCounter, unsigned inputCounter, unsigned outputCounter);

	Cmxaig& operator=(const Cmxaig& rhs);

	virtual ~Cmxaig();

	void addMuxNode( FunctionNode* lhs, FunctionNode* rhs, FunctionNode* sel, bool lhsPolarity, bool rhsPolarity, bool selPolarity, NodeInfo & nodeInfo );

	void addXorNode(FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo);

	void addAndNode(FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo);

	void addChoiceNode(Node* inNode, bool inPolarity, word * choiceFunction, NodeInfo & nodeInfo);

	void addInputNode(String name, word * function, NodeInfo & nodeInfo);

	void addInputNode(word * function, NodeInfo & nodeInfo);

	void addOutputNode(FunctionNode* inNode, const bool& inPolarity, NodeInfo & nodeInfo);

	void addOutputNode(FunctionNode* inNode, const bool& inPolarity, String name, NodeInfo & nodeInfo);

	void addConstantOne(word * function, NodeInfo & nodeInfo);

	void addConstantZero(word * function, NodeInfo & nodeInfo);

	void addEdge(Node* from, Node* to, bool edgePolarity);

	void removeNode(Node* node);

	String toDot();

	std::vector<Node*>* findAllNodes();

	void findNodes(Node* currentNode, std::set<unsigned>& visited, std::vector<Node*>& allNodes);

	/******** Getters and Setters ********/

	NodeVector& getInputs();

	void setInputs(NodeVector& inputs);

	NodeVector& getOutputs();

	void setOutputs(NodeVector& outputs);

	unsigned getNodeCounter();

	void setNodeCounter(unsigned nodeCounter);

	unsigned getMuxCounter();

	void setMuxCounter(unsigned muxCounter);

	unsigned getXorCounter();

	void setXorCounter(unsigned xorCounter);

	unsigned getAndCounter();

	void setAndCounter(unsigned andCounter);

	unsigned getChoiceCounter();

	void setChoiceounter(unsigned choiceCounter);

	unsigned getInputCounter();

	void setInputCounter(unsigned inputCounter);

	unsigned getOutputCounter();

	void setOutputCounter(unsigned outputCounter);

};

} // namespace SubjectGraph

#endif /* CMXAIG_H_ */

/*
 * Cmxaig.cpp
 *
 *  Created on: 09/04/2015
 *      Author: possani
 */

#include "Cmxaig.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <utility>
#include <exception>

namespace SubjectGraph {

Cmxaig::Cmxaig(int nVars, int nWords, Functional::BitVectorPool & functionPool ) : nVars( nVars ), nWords( nWords ), functionPool( functionPool ) {
	this->nodeCounter = 0;
	this->muxCounter = 0;
	this->xorCounter = 0;
	this->andCounter = 0;
	this->choiceCounter = 0;
	this->inputCounter = 0;
	this->outputCounter = 0;
}

Cmxaig::Cmxaig(int nVars, int nWords, Functional::BitVectorPool & functionPool, NodeVector& inputs, NodeVector& outputs,
		unsigned nodeCounter, unsigned muxCounter, unsigned xorCounter, unsigned andCounter,
		unsigned choiceCounter, unsigned inputCounter, unsigned outputCounter) : nVars( nVars ), nWords( nWords ), functionPool( functionPool ) {

	this->inputs = inputs;
	this->outputs = outputs;
	this->nodeCounter = nodeCounter;
	this->muxCounter = muxCounter;
	this->xorCounter = xorCounter;
	this->andCounter = andCounter;
	this->choiceCounter = choiceCounter;
	this->inputCounter = inputCounter;
	this->outputCounter = outputCounter;
}

Cmxaig::~Cmxaig() {

	std::set<unsigned> visited;
	std::vector<Node*>* allNodes;

	/* Find all nodes to be deleted */
	allNodes = findAllNodes();

	/* Delete all nodes */
	for(std::vector<Node*>::iterator i = allNodes->begin(); i != allNodes->end(); i++) {
		delete (*i);
	}

	delete allNodes;
}

void Cmxaig::addMuxNode( FunctionNode* lhs, FunctionNode* rhs, FunctionNode* sel, bool lhsPolarity, bool rhsPolarity, bool selPolarity, NodeInfo & nodeInfo ) {

	word * lhsFunction;
	word * rhsFunction;
	word * selFunction;

	if ( lhsPolarity ) {
		lhsFunction = lhs->getFunctionPtr();
	}
	else {
		lhsFunction = this->functionPool.getMemory();
		Functional::NOT( lhsFunction, lhs->getFunctionPtr(), this->nWords );
	}

	if ( rhsPolarity ) {
		rhsFunction = rhs->getFunctionPtr();
	}
	else {
		rhsFunction = this->functionPool.getMemory();
		Functional::NOT( rhsFunction, rhs->getFunctionPtr(), this->nWords );
	}

	selFunction = sel->getFunctionPtr();

	nodeInfo.function = this->functionPool.getMemory();
	nodeInfo.node = new MuxNode(this->getNextNodeID(), this->getNextMuxID());
	nodeInfo.outPolarity = true;

	if (selPolarity) {
		this->addEdge(lhs, nodeInfo.node, lhsPolarity);
		this->addEdge(rhs, nodeInfo.node, rhsPolarity);
		this->addEdge(sel, nodeInfo.node, true);
		Functional::MUX( nodeInfo.function, lhsFunction, rhsFunction, selFunction, this->nWords );
	}
	else {
		this->addEdge(rhs, nodeInfo.node, rhsPolarity);
		this->addEdge(lhs, nodeInfo.node, lhsPolarity);
		this->addEdge(sel, nodeInfo.node, false);
		Functional::MUX( nodeInfo.function, rhsFunction, lhsFunction, selFunction, this->nWords );
	}
}

void Cmxaig::addXorNode( FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo ) {

	word * lhsFunction;
	word * rhsFunction;

	if ( lhsPolarity ) {
		lhsFunction = lhs->getFunctionPtr();
	}
	else {
		lhsFunction = this->functionPool.getMemory();
		Functional::NOT( lhsFunction, lhs->getFunctionPtr(), this->nWords );
	}

	if ( rhsPolarity ) {
		rhsFunction = rhs->getFunctionPtr();
	}
	else {
		rhsFunction = this->functionPool.getMemory();
		Functional::NOT( rhsFunction, rhs->getFunctionPtr(), this->nWords );
	}

	nodeInfo.function = this->functionPool.getMemory();
	Functional::XOR( nodeInfo.function, rhsFunction, lhsFunction, this->nWords );

	nodeInfo.node = new XorNode(this->getNextNodeID(), this->getNextXorID());
	this->addEdge(lhs, nodeInfo.node, lhsPolarity);
	this->addEdge(rhs, nodeInfo.node, rhsPolarity);

	nodeInfo.outPolarity = true;
}

void Cmxaig::addAndNode( FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo ) {

	word * lhsFunction;
	word * rhsFunction;

	if ( lhsPolarity ) {
		lhsFunction = lhs->getFunctionPtr();
	}
	else {
		lhsFunction = this->functionPool.getMemory();
		Functional::NOT( lhsFunction, lhs->getFunctionPtr(), this->nWords );
	}

	if ( rhsPolarity ) {
		rhsFunction = rhs->getFunctionPtr();
	}
	else {
		rhsFunction = this->functionPool.getMemory();
		Functional::NOT( rhsFunction, rhs->getFunctionPtr(), this->nWords );
	}

	nodeInfo.function = this->functionPool.getMemory();
	Functional::AND( nodeInfo.function, rhsFunction, lhsFunction, this->nWords );

	nodeInfo.node = new AndNode(this->getNextNodeID(), this->getNextAndID());
	this->addEdge(lhs, nodeInfo.node , lhsPolarity);
	this->addEdge(rhs, nodeInfo.node , rhsPolarity);

	nodeInfo.outPolarity = true;
}

void Cmxaig::addChoiceNode( Node* inNode, bool inPolarity, word * choiceFunction, NodeInfo & nodeInfo ) {

	if( Functional::isOdd( choiceFunction ) ) {
		nodeInfo.function = functionPool.getMemory();
		Functional::NOT( nodeInfo.function, choiceFunction, this->nWords );
		inPolarity = !inPolarity;
	}
	else {
		nodeInfo.function = choiceFunction;
	}

	nodeInfo.node = new ChoiceNode(this->getNextNodeID(), this->getNextChoiceID(), nodeInfo.function);
	this->addEdge(inNode, nodeInfo.node, inPolarity);

	nodeInfo.outPolarity = inPolarity;
}

void Cmxaig::addInputNode( String name, word * function, NodeInfo & nodeInfo ) {

	nodeInfo.function = function;
	nodeInfo.node = new InputNode(this->getNextNodeID(), this->getNextInputID(), name, function);
	this->inputs.push_back(nodeInfo.node);
	nodeInfo.outPolarity = true;
}

void Cmxaig::addInputNode( word * function, NodeInfo & nodeInfo ) {

	nodeInfo.function = function;
	nodeInfo.node = new InputNode(this->getNextNodeID(), this->getNextInputID(), function);
	this->inputs.push_back(nodeInfo.node);
	nodeInfo.outPolarity = true;
}

void Cmxaig::addOutputNode( FunctionNode* inNode, const bool& inPolarity, NodeInfo & nodeInfo ) {

	nodeInfo.function = inNode->getFunctionPtr();
	nodeInfo.node = new OutputNode(this->getNextNodeID(), this->getNextOutputID());
	this->addEdge(inNode, nodeInfo.node, inPolarity);
	this->outputs.push_back(nodeInfo.node);
	nodeInfo.outPolarity = inPolarity;
}

void Cmxaig::addOutputNode( FunctionNode* inNode, const bool& inPolarity, String name, NodeInfo & nodeInfo ) {

	nodeInfo.function = inNode->getFunctionPtr();
	nodeInfo.node = new OutputNode(this->getNextNodeID(), this->getNextOutputID(), name);
	this->addEdge(inNode, nodeInfo.node, inPolarity);
	this->outputs.push_back(nodeInfo.node);
	nodeInfo.outPolarity = inPolarity;
}

void Cmxaig::addConstantOne( word * function, NodeInfo & nodeInfo ) {

	nodeInfo.function = function;
	nodeInfo.node = new InputNode( this->getNextNodeID(), this->getNextInputID(), "1", function );
	this->inputs.push_back(nodeInfo.node);
	nodeInfo.outPolarity = true;
}

void Cmxaig::addConstantZero(word * function, NodeInfo & nodeInfo) {

	nodeInfo.function = function;
	nodeInfo.node = new InputNode( this->getNextNodeID(), this->getNextInputID(), "0", function );
	this->inputs.push_back(nodeInfo.node);
	nodeInfo.outPolarity = true;
}

void Cmxaig::addEdge(Node* from, Node* to, bool edgePolarity) {
	from->addOutNode(to, edgePolarity);
	to->addInNode(from, edgePolarity);
} 

void Cmxaig::removeNode(Node* node) {

	std::vector<Node*>::iterator it;

	if(node->isInputNode()) {
		std::cout << "WARNING: You are removing the INPUT node: " << node->getName() << std::endl; //FIXME Adicionar Try Catch
		it = std::find(this->getInputs().begin(), this->getInputs().end(), node);
		if(it != this->getInputs().end()) {
			this->getInputs().erase(it);
		}
		else {
			std::cout << "The INPUT node was not fount and cannot be removed: " << node->getName() << std::endl; //FIXME Adicionar Try Catch
			return;
		}
	}
	else {
		if(node->isOutputNode()) {
			std::cout << "WARNING: You are removing the OUTPUT node: " << node->getName() << std::endl; //FIXME Adicionar Try Catch
			it = std::find(this->getInputs().begin(), this->getInputs().end(), node);
			if(it != this->getInputs().end()) {
				this->getInputs().erase(it);
			}
			else {
				std::cout << "The INPUT node was not fount and cannot be removed: " << node->getName() << std::endl; //FIXME Adicionar Try Catch
				return;
			}
		}
	}

	// Remove all pointers from the inputs of node
	for(std::vector<Node*>::iterator inputsIt = node->getInNodes().begin(); inputsIt != node->getInNodes().end(); inputsIt++) {
		(*inputsIt)->removeOutNode(node);
	}

	// Remove all pointers from the outputs of node
	for(std::vector<Node*>::iterator outputsIt = node->getOutNodes().begin(); outputsIt != node->getOutNodes().end(); outputsIt++) {
		(*outputsIt)->removeInNode(node);
	}

	delete node;
}

unsigned Cmxaig::getNextNodeID() {
	unsigned aux = this->nodeCounter;
	this->setNodeCounter(aux+1);
	return aux;
}

unsigned Cmxaig::getNextMuxID(){
	unsigned aux = this->muxCounter;
	this->setMuxCounter(aux+1);
	return aux;
}

unsigned Cmxaig::getNextXorID() {
	unsigned aux = this->xorCounter;
	this->setXorCounter(aux+1);
	return aux;
}

unsigned Cmxaig::getNextAndID() {
	unsigned aux = this->andCounter;
	this->setAndCounter(aux+1);
	return aux;
}

unsigned Cmxaig::getNextChoiceID() {
	unsigned aux  = this->choiceCounter;
	this->setChoiceounter(aux+1);
	return aux;
}

unsigned Cmxaig::getNextInputID() {
	unsigned aux = this->inputCounter;
	this->setInputCounter(aux+1);
	return aux;
}

unsigned Cmxaig::getNextOutputID() {
	unsigned aux = this->outputCounter;
	this->setOutputCounter(aux+1);
	return aux;
}


String Cmxaig::toDot() {

	std::stringstream dotNodes;
	std::stringstream dotEdges;
	std::set<unsigned> visited;

	dotNodes << "digraph cmxaig{" << std::endl;

	for(unsigned i=0; i < this->getOutputs().size(); i++) {
		elementsToDot(*this->getOutputs()[i], visited, dotNodes, dotEdges);
	}

	dotNodes << dotEdges.str();

	dotNodes << "{ rank=source; ";

	for (unsigned i = 0; i < this->getInputs().size(); i++) {
		dotNodes << "\"" << this->getInputs()[i]->getName() << "\" ";
	}

	dotNodes << "}" << std::endl;
	dotNodes << "{ rank=sink; ";

	for (unsigned i = 0; i < this->getOutputs().size(); i++) {
		dotNodes << "\"" << this->getOutputs()[i]->getName() << "\" ";
	}

	dotNodes << "}" << std::endl << "rankdir=\"BT\";" << std::endl << "}" << std::endl;

	return dotNodes.str();
}

void Cmxaig::elementsToDot(Node& currentNode, std::set<unsigned>& visited, std::stringstream& dotNodes, std::stringstream& dotEdges) {

	if(currentNode.isInputNode()) {
		if( visited.find(currentNode.getId()) == visited.end()) {
			dotNodes << currentNode.toDot() << std::endl;
			visited.insert(currentNode.getId());
		}
		return;
	}

	if( visited.find(currentNode.getId()) != visited.end()) {
		return;
	}

	visited.insert(currentNode.getId());

	for(unsigned i=0; i < currentNode.getInNodes().size() ; i++) {

		elementsToDot(*currentNode.getInNodes()[i], visited, dotNodes, dotEdges);

		/* Write Edges */
		dotEdges << "\"" << currentNode.getInNodes()[i]->getName() << "\"" << " -> " << "\"" << currentNode.getName() << "\"";
		if(currentNode.getInEdges()[i]) {
			dotEdges << " [penwidth=3]" << std::endl;
		}
		else {
			dotEdges << " [style=dashed, penwidth=3]" << std::endl;
		}
	}

	/* Write Nodes */
	dotNodes << currentNode.toDot() << std::endl;
}

std::vector<Node*>* Cmxaig::findAllNodes() {

	std::set<unsigned> visited;
	std::vector<Node*>* allNodes = new std::vector<Node*>();

	for(unsigned i=0; i < this->getOutputs().size(); i++) {
		findNodes(this->getOutputs()[i], visited, (*allNodes));
	}

	return allNodes;
}

void Cmxaig::findNodes(Node* currentNode, std::set<unsigned>& visited, std::vector<Node*>& allNodes) {

	if(visited.find(currentNode->getId()) != visited.end()) {
		return;
	}

	if(currentNode->isInputNode()) {
		if( visited.find(currentNode->getId()) == visited.end()) {
			visited.insert(currentNode->getId());
			allNodes.push_back(currentNode);
		}
		return;
	}

	visited.insert(currentNode->getId());

	for(unsigned i=0; i < currentNode->getInNodes().size(); i++) {
		findNodes(currentNode->getInNodes()[i], visited, allNodes);
	}

	allNodes.push_back(currentNode);
}

/******** Getters and Setters ********/

NodeVector& Cmxaig::getInputs() {
	return this->inputs;
}

void Cmxaig::setInputs(NodeVector& inputs) {
	this->inputs = inputs;
}

NodeVector& Cmxaig::getOutputs() {
	return this->outputs;
}

void Cmxaig::setOutputs(NodeVector& outputs) {
	this->outputs = outputs;
}

unsigned Cmxaig::getNodeCounter() {
	return this->nodeCounter;
}

void Cmxaig::setNodeCounter(unsigned nodeCounter) {
	this->nodeCounter = nodeCounter;
}

unsigned Cmxaig::getMuxCounter() {
	return this->muxCounter;
}

void Cmxaig::setMuxCounter(unsigned muxCounter) {
	this->muxCounter = muxCounter;
}

unsigned Cmxaig::getXorCounter() {
	return this->xorCounter;
}

void Cmxaig::setXorCounter(const unsigned xorCounter) {
	this->xorCounter = xorCounter;
}

unsigned Cmxaig::getAndCounter() {
	return this->andCounter;
}

void Cmxaig::setAndCounter(unsigned andCounter) {
	this->andCounter = andCounter;
}

unsigned Cmxaig::getChoiceCounter() {
	return this->choiceCounter;
}

void Cmxaig::setChoiceounter(unsigned choiceCounter) {
	this->choiceCounter = choiceCounter;
}

unsigned Cmxaig::getInputCounter() {
	return this->inputCounter;
}

void Cmxaig::setInputCounter(unsigned inputCounter) {
	this->inputCounter = inputCounter;
}

unsigned Cmxaig::getOutputCounter() {
	return this->outputCounter;
}

void Cmxaig::setOutputCounter(unsigned outputCounter) {
	this->outputCounter = outputCounter;
}

} // namespace SubjectGraph

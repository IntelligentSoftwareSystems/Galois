#include "ChoiceNode.h"

#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <exception>

namespace SubjectGraph {

ChoiceNode::ChoiceNode() {
}

ChoiceNode::ChoiceNode(const unsigned & id, const unsigned & typeId, word * function) : FunctionNode::FunctionNode(id, typeId, function) {
	std::stringstream name;
	name << "C" << typeId;
	//name << "Choice_" << typeId;
	this->setName(name.str());
}

ChoiceNode::ChoiceNode(const ChoiceNode & node) {
	operator=(node);
}

ChoiceNode& ChoiceNode::operator=(const ChoiceNode & rhs) {
	FunctionNode::operator =(rhs);
	return *this;
}

void ChoiceNode::addInNode(Node* inNode, const bool& inPolatiry) {
	this->getInNodes().push_back(inNode);
	this->getInEdges().push_back(inPolatiry);
}

void ChoiceNode::addOutNode(Node* outNode, const bool& outPolatiry) {
	this->getOutNodes().push_back(outNode);
	this->getOutEdges().push_back(outPolatiry);
}

void ChoiceNode::removeInNode(Node* inNode) {
	// Find the in_node and remove it from the inNodes vector
	std::vector<Node*>::iterator it = std::find(this->getInNodes().begin(), this->getInNodes().end(), inNode);
	if(it != this->getInNodes().end()) {
		this->getInNodes().erase(it);
		// Find the index of the edge related to the in_node and remove it from inEdges vector.
		int index = it - this->getInNodes().begin();
		this->getInEdges().erase(this->getInEdges().begin()+index);
	}
	else {
		std::cout << "Input Missing: the input node was found in this CHOICE node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
}

void ChoiceNode::removeOutNode(Node* outNode) {
	// Find the out_node and remove it from the inNodes vector
	std::vector<Node*>::iterator it = std::find(this->getOutNodes().begin(), this->getOutNodes().end(), outNode);

	if(it != this->getOutNodes().end()) {
		this->getOutNodes().erase(it);
		// Find the index of the edge related to the out_node and remove it from inEdges vector.
		int index = it - this->getOutNodes().begin();
		this->getOutEdges().erase(this->getOutEdges().begin()+index);
	}
	else {
		std::cout << "Output Missing: the output node was found in this CHOICE node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
}

bool ChoiceNode::isChoiceNode() const {
	return true;
}

bool ChoiceNode::isInputNode() const {
	return false;
}

String ChoiceNode::toDot() const {
	//return "\"" + getName() + "\"[label=\"" + getName() + "\\n" + getFunctionValue().toHex() + "\", shape=circle, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#013220\", fillcolor=\"#90EE90\", fontsize=14]\n";
	return "\"" + getName() + "\"[label=\"" + getName() + "\", shape=circle, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#013220\", fillcolor=\"#90EE90\", fontsize=30]\n";
}

}  // namespace SubjectGraph

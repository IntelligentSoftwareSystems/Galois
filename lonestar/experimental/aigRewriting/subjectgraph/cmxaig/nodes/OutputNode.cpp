#include "OutputNode.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <exception>

namespace SubjectGraph {

OutputNode::OutputNode() : Node::Node() {
}

OutputNode::OutputNode(const unsigned & id, const unsigned & typeId) : Node::Node(id, typeId) {
	std::stringstream name;
	name << "O" << typeId;
	this->setName(name.str());
}

OutputNode::OutputNode(const unsigned & id, const unsigned & typeId, const String& name) : Node::Node(id, typeId) {
	this->setName(name);
}

OutputNode::OutputNode(const OutputNode & node) {
	operator=(node);
}

OutputNode& OutputNode::operator=(const OutputNode & rhs) {
	Node::operator=(rhs);
	return *this;
}

void OutputNode::addInNode(Node* inNode, const bool& inPolarity) {
	if(this->getInNodes().size() > 0) {
		std::cout << "Input Overflow: Just 1 input is allowed in OUTPUT nodes. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
	else {
		this->getInNodes().push_back(inNode);
		this->getInEdges().push_back(inPolarity);
	}
}

void OutputNode::addOutNode(Node* outNode, const bool& outPolarity) {
	std::cout << "The output cannot be added, outputs are not allowed in OUTPUT nodes. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
}

void OutputNode::removeInNode(Node* inNode) {
	if(this->getInNodes().size() > 0) {
		if(this->getInNodes()[0] == inNode) {
			this->getInNodes().erase(this->getInNodes().begin());
			this->getInEdges().erase(this->getInEdges().begin());
			return;
		}
	}

	std::cout << "Input Missing: the input node was found in this OUTPUT node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch

	return;
}

void OutputNode::removeOutNode(Node* outNode) {
	std::cout << "The output cannot be removed, outputs are not allowed in OUTPUT nodes. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
}

bool OutputNode::isAndNode() const {
	return false;
}

bool OutputNode::isXorNode() const {
	return false;
}

bool OutputNode::isMuxNode() const {
	return false;
}

bool OutputNode::isChoiceNode() const {
	return false;
}

bool OutputNode::isFunctionNode() const {
	return false;
}

bool OutputNode::isInputNode() const {
	return false;
}

bool OutputNode::isOutputNode() const {
	return true;
}

String OutputNode::toDot() const {
	return "\"" + getName() + "\"[shape=circle, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#69359C\", fillcolor=\"#E6E6FA\", fontsize=25]\n";
}

}  // namespace SubjectGraph

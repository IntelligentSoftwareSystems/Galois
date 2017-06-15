#include "AndNode.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <exception>

namespace SubjectGraph {

AndNode::AndNode() {
}

AndNode::AndNode(const unsigned & id, const unsigned & typeId) : Node::Node(id, typeId) {
	std::stringstream name;
	name << "A" << typeId;
	//name << "And_" << typeId;
	this->setName(name.str());
}

AndNode::AndNode(const AndNode& node) {
	operator=(node);
}

AndNode& AndNode::operator=(const AndNode& rhs) {
	Node::operator=(rhs);
	return *this;
}

void AndNode::addInNode(Node* inNode, const bool& inPolatiry) {
	if(this->getInNodes().size() >= 2) {
		std::cout << "Input Overflow: Just 2 inputs are allowed in a AND node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
	else {
		this->getInNodes().push_back(inNode);
		this->getInEdges().push_back(inPolatiry);
	}
}

void AndNode::addOutNode(Node* outNode, const bool& outPolatiry) {
	if(this->getOutNodes().size() >= 1) {
		std::cout << "Outputs Overflow: Just 1 output is allowed in a AND node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
	else {
		this->getOutNodes().push_back(outNode);
		this->getOutEdges().push_back(outPolatiry);
	}
}

void AndNode::removeInNode(Node* inNode) {
	if(this->getInNodes().size() > 0) {
		if(this->getInNodes()[0] == inNode) {
			this->getInNodes().erase(this->getInNodes().begin());
			this->getInEdges().erase(this->getInEdges().begin());
			return;
		}
		else {
			if(this->getInNodes()[1] == inNode) {
				this->getInNodes().erase(this->getInNodes().begin()+1);
				this->getInEdges().erase(this->getInEdges().begin()+1);
				return;
			}
		}
	}

	std::cout << "Input Missing: the input node was found in this AND node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch

	return;
}

void AndNode::removeOutNode(Node* outNode) {
	if(this->getOutNodes().size() > 0) {
		if(this->getOutNodes()[0] == outNode) {
			this->getOutNodes().erase(this->getOutNodes().begin());
			this->getOutEdges().erase(this->getOutEdges().begin());
			return;
		}
	}

	std::cout << "Output Missing: the output node was found in this AND node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch

	return;
}

bool AndNode::isAndNode() const {
	return true;
}

bool AndNode::isXorNode() const {
	return false;
}

bool AndNode::isMuxNode() const {
	return false;
}

bool AndNode::isChoiceNode() const {
	return false;
}

bool AndNode::isFunctionNode() const {
	return false;
}

bool AndNode::isInputNode() const {
	return false;
}

bool AndNode::isOutputNode() const {
	return false;
}

String AndNode::toDot() const {
	return "\"" + getName() + "\"[shape=square, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#00008B\", fillcolor=\"#ADD8E6\", fontsize=30]\n";
}

}  // namespace SubjectGraph

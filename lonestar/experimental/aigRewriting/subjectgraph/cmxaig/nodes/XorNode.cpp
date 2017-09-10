#include "XorNode.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <exception>

namespace SubjectGraph {

XorNode::XorNode() {
}

XorNode::XorNode(const unsigned & id, const unsigned & typeId) : Node::Node(id, typeId) {
	std::stringstream name;
	name << "X" << typeId;
	//name << "Xor_" << typeId;
	this->setName(name.str());
}

XorNode::XorNode(const XorNode& node) {
	operator=(node);
}

XorNode& XorNode::operator=(const XorNode& rhs) {
	Node::operator=(rhs);
	return *this;
}

void XorNode::addInNode(Node* inNode, const bool& inPolatiry) {
	if(this->getInNodes().size() >= 2) {
		std::cout << "Input Overflow: Just 2 inputs are allowed in a XOR node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
	else {
		this->getInNodes().push_back(inNode);
		this->getInEdges().push_back(inPolatiry);
	}
}

void XorNode::addOutNode(Node* outNode, const bool& outPolatiry) {
	if(this->getOutNodes().size() >= 1) {
		std::cout << "Outputs Overflow: Just 1 output is allowed in a XOR node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
	else {
		this->getOutNodes().push_back(outNode);
		this->getOutEdges().push_back(outPolatiry);
	}
}

void XorNode::removeInNode(Node* inNode) {
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

	std::cout << "Input Missing: the input node was found in this XOR node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch

	return;
}

void XorNode::removeOutNode(Node* outNode) {
	if(this->getOutNodes().size() > 0) {
		if(this->getOutNodes()[0] == outNode) {
			this->getOutNodes().erase(this->getOutNodes().begin());
			this->getOutEdges().erase(this->getOutEdges().begin());
			return;
		}
	}

	std::cout << "Output Missing: the output node was found in this XOR node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch

	return;
}

bool XorNode::isAndNode() const {
	return false;
}

bool XorNode::isXorNode() const {
	return true;
}

bool XorNode::isMuxNode() const {
	return false;
}

bool XorNode::isChoiceNode() const {
	return false;
}

bool XorNode::isFunctionNode() const {
	return false;
}

bool XorNode::isInputNode() const {
	return false;
}

bool XorNode::isOutputNode() const {
	return false;
}

String XorNode::toDot() const {
	return "\"" + getName() + "\"[shape=square, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#996515\", fillcolor=\"#FFFFED\", fontsize=30]\n";
}

}  // namespace SubjectGraph

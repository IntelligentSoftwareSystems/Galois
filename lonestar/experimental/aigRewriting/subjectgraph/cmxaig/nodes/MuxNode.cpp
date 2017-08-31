#include "MuxNode.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <exception>

namespace SubjectGraph {

MuxNode::MuxNode() {
}

MuxNode::MuxNode(const unsigned & id, const unsigned & type_id) : Node::Node(id, type_id) {
	std::stringstream name;
	name << "M" << type_id;
	//name << "Mux_" << type_id;
	this->setName(name.str());
}

MuxNode::MuxNode(const MuxNode & node) {
	operator=(node);
}

MuxNode& MuxNode::operator=(const MuxNode & rhs) {
	Node::operator=(rhs);
	return *this;
}

void MuxNode::addInNode(Node* in_node, const bool & isInvEdge) {
	if(this->getInNodes().size() >= 3) {
		std::cout << "Input Overflow: Just 3 inputs are allowed in a MUX node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
	else {
		this->getInNodes().push_back(in_node);
		this->getInEdges().push_back(isInvEdge);
	}
}

void MuxNode::addOutNode(Node* out_node, const bool& isInvEdge) {
	if(this->getOutNodes().size() >= 1) {
		std::cout << "Outputs Overflow: Just 1 output is allowed in a MUX node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
	else {
		this->getOutNodes().push_back(out_node);
		this->getOutEdges().push_back(isInvEdge);
	}
}

void MuxNode::removeInNode(Node* in_node) {
	if(this->getInNodes().size() > 0) {
		if(this->getInNodes()[0] == in_node) {
			this->getInNodes().erase(this->getInNodes().begin());
			this->getInEdges().erase(this->getInEdges().begin());
			return;
		}
		else {
			if(this->getInNodes()[1] == in_node) {
				this->getInNodes().erase(this->getInNodes().begin()+1);
				this->getInEdges().erase(this->getInEdges().begin()+1);
				return;
			}
			else {
				if(this->getInNodes()[2] == in_node) {
					this->getInNodes().erase(this->getInNodes().begin()+2);
					this->getInEdges().erase(this->getInEdges().begin()+2);
					return;
				}
			}
		}
	}

	std::cout << "Input Missing: the input node was found in this MUX node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch

	return;

}

void MuxNode::removeOutNode(Node* out_node) {
	if(this->getOutNodes().size() > 0) {
		if(this->getOutNodes()[0] == out_node) {
			this->getOutNodes().erase(this->getOutNodes().begin());
			this->getOutEdges().erase(this->getOutEdges().begin());
			return;
		}
	}

	std::cout << "Output Missing: the output node was found in this MUX node. Node: " << this->getName() << std::endl; //FIXME Adicionar Try Catch

	return;
}

bool MuxNode::isAndNode() const {
	return false;
}

bool MuxNode::isXorNode() const {
	return false;
}

bool MuxNode::isMuxNode() const {
	return true;
}

bool MuxNode::isChoiceNode() const {
	return false;
}

bool MuxNode::isFunctionNode() const {
	return false;
}

bool MuxNode::isInputNode() const {
	return false;
}

bool MuxNode::isOutputNode() const {
	return false;
}

String MuxNode::toDot() const {
	return "\"" + getName() + "\"[shape=square, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#8B0000\", fillcolor=\"#FFB6C1\", fontsize=30]\n";
}

}  // namespace SubjectGraph

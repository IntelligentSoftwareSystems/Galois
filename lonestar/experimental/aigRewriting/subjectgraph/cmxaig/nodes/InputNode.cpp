#include "InputNode.h"
#include "../../functional/FunctionHandler.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <exception>

namespace SubjectGraph {

typedef Functional::word word;

InputNode::InputNode() {
}

InputNode::InputNode(const unsigned & id, const unsigned & typeId, word * function) : FunctionNode::FunctionNode(id, typeId, function) {
	std::stringstream name;
	name << "In_" << this->getTypeId();
	this->setName(name.str());
}

InputNode::InputNode(const unsigned & id, const unsigned& typeId, String name, word * function) : FunctionNode::FunctionNode(id, typeId, function) {
	this->setName( name );
}

InputNode::InputNode(const InputNode & node) {
	operator=(node);
}

InputNode& InputNode::operator=(const InputNode & rhs) {
	FunctionNode::operator=(rhs);
	return *this;
}

void InputNode::addInNode(Node* inNode, const bool & inPolarity) {
	std::cout << "The input cannot be added, inputs are not allowed in INPUT nodes. Input: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
}

void InputNode::addOutNode(Node* outNode, const bool & outPolarity) {
	this->getOutNodes().push_back(outNode);
	this->getOutEdges().push_back(outPolarity);
}

void InputNode::removeInNode(Node* inNode) {
	std::cout << "The input cannot be removed, inputs are not allowed in INPUT nodes. Input: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
}

void InputNode::removeOutNode(Node* outNode) {
	// Find the out_node and remove it from the inNodes vector
	std::vector<Node*>::iterator it = std::find(this->getOutNodes().begin(), this->getOutNodes().end(), outNode);

	if(it != this->getOutNodes().end()) {
		this->getOutNodes().erase(it);
		// Find the index of the edge related to the out_node and remove it from inEdges vector.
		int index = it - this->getOutNodes().begin();
		this->getOutEdges().erase(this->getOutEdges().begin()+index);
	}
	else {
		std::cout << "Output Missing: the output node was found in this INPUT node. Input: " << this->getName() << std::endl; //FIXME Adicionar Try Catch
	}
}

bool InputNode::isChoiceNode() const {
	return false;
}

bool InputNode::isInputNode() const {
	return true;
}

String InputNode::toDot() const {
	//return "\"" + getName() + "\"[label=\"" + getName() + "\\n" + getFunctionValue().to_hex() + "\", shape=circle, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#CC5500\", fillcolor=\"#FFCC99\", fontsize=15]\n";
	return "\"" + getName() + "\"[label=\"" + getName() + "\", shape=circle, height=1, width=1, penwidth=3, style=filled, fontcolor=\"#CC5500\", fillcolor=\"#FFCC99\", fontsize=35]\n";
}

}  // namespace SubjectGraph

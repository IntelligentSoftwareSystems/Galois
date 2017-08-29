/*
 * FunctionNode.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: mhbackes
 */

#include "FunctionNode.h"

namespace SubjectGraph {

FunctionNode::FunctionNode() : Node::Node() {
	this->value = nullptr;
}

FunctionNode::FunctionNode(const unsigned & id, const unsigned & typeId, word * value) : Node::Node(id, typeId) {
	this->value = value;
}

FunctionNode::FunctionNode(const FunctionNode & node) : Node::Node(node.getId(), node.getTypeId()) {
	this->value = node.getFunction(); // FIXME
}

FunctionNode& FunctionNode::operator=(const FunctionNode & rhs) {
	Node::operator=(rhs);
	value = rhs.value;
	return *this;
}

word * FunctionNode::getFunction() const {
	return value;
}

word * FunctionNode::getFunctionPtr() {
	return value;
}

bool FunctionNode::isAndNode() const {
	return false;
}

bool FunctionNode::isXorNode() const {
	return false;
}

bool FunctionNode::isMuxNode() const {
	return false;
}

bool FunctionNode::isFunctionNode() const {
	return true;
}

bool FunctionNode::isOutputNode() const {
	return false;
}

FunctionNode::~FunctionNode() {
}

} /* namespace subjectGraph */

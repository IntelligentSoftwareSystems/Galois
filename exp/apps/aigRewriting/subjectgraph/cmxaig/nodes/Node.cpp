#include "Node.h"

#include <sstream>

namespace SubjectGraph {

Node::Node() {
	id = 0;
	typeId = 0;
}

Node::Node(const unsigned & id, const unsigned & typeId) {
	this->id = id;
	this->typeId = typeId;
	std::stringstream name;
	name << "N" << typeId;
	//name << "Node_" << typeId;
	this->name = name.str();
}

Node::Node(const Node& node) {
	operator=(node);
}

Node::~Node() {
}

Node& Node::operator=(const Node & rhs) {
	this->id = rhs.id;
	this->typeId = rhs.typeId;
	this->name = rhs.name;
	this->inNodes = rhs.inNodes;
	this->inEdges = rhs.inEdges;
	this->outNodes = rhs.outNodes;
	this->outEdges = rhs.outEdges;
	return *this;
}

/****************** GETTERS AND SETTERS ******************/

unsigned Node::getId() const {
	return id;
}

unsigned Node::getTypeId() const {
	return typeId;
}

std::vector<Node*>& Node::getInNodes() {
	return inNodes;
}

void Node::setInNodes(const std::vector<Node*>& inNodes) {
	this->inNodes = inNodes;
}

EdgeValues& Node::getInEdges() {
	return inEdges;
}

void Node::setInEdges(const EdgeValues& inEdges) {
	this->inEdges = inEdges;
}

std::vector<Node*>& Node::getOutNodes() {
	return outNodes;
}

void Node::setOutNodes(const std::vector<Node*>& outNodes) {
	this->outNodes = outNodes;
}

EdgeValues& Node::getOutEdges() {
	return outEdges;
}

void Node::setOutEdges(const EdgeValues& outEdges) {
	this->outEdges = outEdges;
}

const String& Node::getName() const {
	return name;
}

void Node::setName(const String& name) {
	this->name = name;
}

}  // namespace SubjectGraph

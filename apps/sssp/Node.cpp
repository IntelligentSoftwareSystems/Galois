/*
 * Node.cpp
 *
 *  Created on: Oct 18, 2010
 *      Author: amshali
 */

#include "Node.h"

Node::~Node() {
	// TODO Auto-generated destructor stub
}

Node& Node::clone() {
	Node *n= new Node(this->id, this->dist);
	return *n;
}

void Node::restoreFrom(Node copy) {
	this->dist = copy.dist;
}


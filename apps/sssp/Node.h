/*
 * Node.h
 *
 *  Created on: Oct 18, 2010
 *      Author: amshali
 */

#ifndef NODE_H_
#define NODE_H_
#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;

class Node {
private:
	const int id;
	int dist;
public:
	Node(int _id, int _dist) : id(_id), dist(_dist) {};
	virtual ~Node();
	string toString() {string *s = new string(""); return *s;}; //TODO: complete toString
	Node& clone();
	void restoreFrom(Node copy);
};

#endif /* NODE_H_ */

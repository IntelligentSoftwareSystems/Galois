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

const int INFINITY = 2147483647;

class SNode {
private:
	int id;
	int dist;
	SNode* pred;
public:
	SNode(int _id) : id(_id) { dist = INFINITY; pred = NULL; };
	virtual ~SNode();
	const int get_id() { return id; };
	int get_dist() { return dist; };
	void set_dist(const int d) { dist = d; };
	void set_pred(SNode* p) { pred = p; };
	string toString() {string *s = new string(""); return *s;}; //TODO: complete toString
};

#endif /* NODE_H_ */

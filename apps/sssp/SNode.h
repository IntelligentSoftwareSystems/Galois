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

const int DIST_INFINITY = 2147483647;

class SNode {
private:
public:
	int id;
	int dist;

	SNode(int _id) : id(_id) { dist = DIST_INFINITY; };
	virtual ~SNode() {};
	const int get_id() { return id; };
	int get_dist() { return dist; };
	void set_dist(const int d) { dist = d; };
	string toString() {string *s = new string(""); return *s;}; //TODO: complete toString
};

#endif /* NODE_H_ */

/*
 * SEdge.h
 *
 *  Created on: Oct 18, 2010
 *      Author: reza
 */

#ifndef SEDGE_H_
#define SEDGE_H_

#include "SNode.h"

class SEdge {
private:
	int weight;
public:
	SEdge() {
		weight = 0;
	}
	;
	SEdge(const int w) {
		weight = w;
	}
	;
	SEdge(const SEdge & copy) :
		weight(copy.weight) {
	}
	;
	SEdge& operator=(const SEdge & copy) {
		weight = copy.weight;
		return *this;
	}
	virtual ~SEdge(){};

	int get_weight() const {
		return weight;
	}

	void set_weight(int weight) {
		this->weight = weight;
	}

};

#endif /* SEDGE_H_ */

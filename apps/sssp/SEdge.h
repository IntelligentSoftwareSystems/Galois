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
	SNode * source;
	SNode * destination;
	int weight;
public:
	SEdge(SNode &s, SNode &d, const int w) :
			source(&s), destination(&d), weight(w) {};
	SEdge(const SEdge & copy) : source(copy.source),
			destination(copy.destination), weight(copy.weight) {};
	SEdge& operator=(const SEdge & copy) {
		source = copy.source;
		destination = copy.destination;
		weight = copy.weight;
		return *this;
	}
	virtual ~SEdge();
    SNode & get_destination() const
    {
        return *destination;
    }

    SNode & get_source() const
    {
        return *source;
    }

    int get_weight() const
    {
        return weight;
    }

    void set_destination(SNode & destination)
    {
        this->destination = &destination;
    }

    void set_source(SNode & source)
    {
        this->source = &source;
    }

    void set_weight(int weight)
    {
        this->weight = weight;
    }

};

#endif /* SEDGE_H_ */

/*
 * SSSP.h
 *
 *  Created on: Oct 18, 2010
 *      Author: reza
 */

#ifndef SSSP_H_
#define SSSP_H_
#include <list>

#include "SNode.h"
#include "SEdge.h"
#include "Galois/Graphs/Graph.h"
typedef FirstGraph<SNode, SEdge> Graph;

class SSSP {
public:
	SSSP();
	virtual ~SSSP();

	void bellman_ford(const std::list<SNode> & nodes,
			const std::list<SEdge> & edges, SNode & source);
};

#endif /* SSSP_H_ */

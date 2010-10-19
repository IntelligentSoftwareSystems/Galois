/*
 * SSSP.h
 *
 *  Created on: Oct 18, 2010
 *      Author: amshali
 */

#ifndef SSSP_H_
#define SSSP_H_
#include "Node.h"
#include "Galois/Graphs/Graph.h"
typedef FirstGraph<Node,int>            Graph;


class SSSP {
public:
	void updateSourceAndSink(Graph &g, const int sourceId, const int sinkId);
	void initializeGraph(Graph &g, char* filename);
	SSSP();
	virtual ~SSSP();
};

#endif /* SSSP_H_ */

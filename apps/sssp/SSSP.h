/*
 * SSSP.h
 *
 *  Created on: Oct 18, 2010
 *      Author: reza
 */

#ifndef SSSP_H_
#define SSSP_H_

#include <list>
#include <vector>
#include <queue>

#include "SNode.h"
#include "SEdge.h"
#include "Galois/Graphs/Graph.h"
typedef FirstGraph<SNode, SEdge, true> Graph;
typedef FirstGraph<SNode, SEdge, true>::GraphNode GNode;

#include "ExecutorType.h"

#include "UpdateRequest.h"

class SSSP {
private:
	Graph* graph;
	GNode source;
	GNode sink;
	ExecutorType executorType;
	int numNodes;
	int numEdges;
	int maxWeight;
	int delta;

public:
	SSSP();
	virtual ~SSSP();

	void updateSourceAndSink(const int sourceId, const int sinkId);
	int getEdgeData(GNode src, GNode dst);
	void verify();
	void runBody(const GNode src);
};

#endif /* SSSP_H_ */

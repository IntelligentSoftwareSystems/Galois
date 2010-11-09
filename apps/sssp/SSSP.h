/*
 * SSSP.h
 *
 *  Created on: Oct 18, 2010
 *      Author: amin, reza
 */

#ifndef SSSP_H_
#define SSSP_H_

#include <list>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>

#include "Support/ThreadSafe/TSPQueue.h"

#include "Support/ThreadSafe/simple_lock.h"
#include "Support/ThreadSafe/TSQueue.h"

using namespace std;

#include "SNode.h"
#include "SEdge.h"

#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
typedef Galois::Graph::FirstGraph<SNode, SEdge, true> Graph;
typedef Galois::Graph::FirstGraph<SNode, SEdge, true>::GraphNode GNode;

#include "ExecutorType.h"

#include "UpdateRequest.h"


class SSSP {
private:
	GNode source;
	GNode sink;
	ExecutorType executorType;
	int numNodes;
	int numEdges;
	int maxWeight;

public:
	Graph* graph;
	int delta;
	SSSP() {
	}
	;
	virtual ~SSSP() {
	}
	;
	void initializeGraph(char *filename);
	void updateSourceAndSink(const int sourceId, const int sinkId);
	int getEdgeData(GNode src, GNode dst);
	bool verify();
	void runBody(const GNode src);
	void run(bool bfs, char *filename, int threadnum);
	void runBodyParallel(const GNode src);

	class UpdateRequestCompare {
	private:
		int computeIndx(UpdateRequest * u1) const {
			const int range = 30*1024;
      int bucket1 = std::min<int>(u1->w / 700, range - 1);
      int retval1;
      if (u1->light)
        retval1 = bucket1 * 2;
      else
        retval1 = bucket1 * 2 + 1;
      return retval1;
		}
	public:
		bool operator()(UpdateRequest * u1, UpdateRequest * u2) const {
			if (computeIndx(u1) > computeIndx(u2)) return true;
			return false;
		}
	};

};

#endif /* SSSP_H_ */

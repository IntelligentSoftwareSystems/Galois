/** GMetis -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Xin Sui <xinsui@cs.utexas.edu>
 */

#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "PMetis.h"
#include "defs.h"
#include <climits>
#include <vector>
#include <algorithm>
using namespace std;
static const int SMALL_NUM_ITER_PARTITION = 3;
static const int LARGE_NUM_ITER_PARTITION = 8;


void bisectionArray(GGraph* graph, GNode* nodes, int numNodes, int minWgtPart1, int maxWgtPart1, int* pwgts, int* visited, int* queue) {
  std::fill_n(visited, numNodes, 0);
	vector<int> indexes(numNodes);
	for(int i=0;i<numNodes;i++){
		indexes[i]=i;
	}
	std::random_shuffle(indexes.begin(), indexes.end());
	for(int i=0;i<numNodes;i++){
		int node = indexes[i];
		if(visited[node]) continue;
		queue[0] = node;
		int first = 0;
		int last = 1;
		bool drain = false;
		while(first!=last){
			node = queue[first++];
			visited[node] = 1;
			int nodeWeight = graph->getData(nodes[i]).getWeight();

			if (pwgts[0] > 0 && (pwgts[1] - nodeWeight) < minWgtPart1) {
				drain = true;
				continue;
			}

			graph->getData(nodes[i]).setPartition(0);
			pwgts[0] += nodeWeight;
			pwgts[1] -= nodeWeight;
			if (pwgts[1] <= maxWgtPart1) {
				return;
			}

			drain = false;

			for (GGraph::edge_iterator jj = graph->edge_begin(nodes[i], Galois::NONE), eejj = graph->edge_end(nodes[i], Galois::NONE); jj != eejj; ++jj) {
			  GNode neighbor = graph->getEdgeDst(jj);
			  int k = graph->getData(neighbor).getNodeId();//id is same as the position in nodes array
				if (visited[k] == 0) {
					queue[last++] = k;
					visited[k] = 1;
				}
			}
		}
		if(drain){
			return;
		}
	}
}

void bisection(GGraph* graph, GNode* nodes, int numNodes, int minWgtPart1, int maxWgtPart1, int* pwgts, int* visited, int* queue) {

  std::fill_n(visited, numNodes, 0);
	queue[0] = getRandom(numNodes);
	visited[queue[0]] = 1;
	int first = 0;
	int last = 1;
	int nleft = numNodes - 1;
	bool drain = false;
	for (;;) {
		if (first == last) {
			if (nleft == 0 || drain) {
				break;
			}

			int k = getRandom(nleft);
			int i = 0;
			for (; i < numNodes; i++) {
				if (visited[i] == 0) {
					if (k == 0) {
						break;
					} else {
						k--;
					}
				}
			}
			queue[0] = i;
			visited[i] = 1;
			first = 0;
			last = 1;
			nleft--;
		}

		int i = queue[first++];
		int nodeWeight = graph->getData(nodes[i]).getWeight();

		if (pwgts[0] > 0 && (pwgts[1] - nodeWeight) < minWgtPart1) {
			drain = true;
			continue;
		}

		graph->getData(nodes[i]).setPartition(0);
		pwgts[0] += nodeWeight;
		pwgts[1] -= nodeWeight;
		if (pwgts[1] <= maxWgtPart1) {
			break;
		}

		drain = false;

		for (GGraph::edge_iterator jj = graph->edge_begin(nodes[i], Galois::NONE), eejj = graph->edge_end(nodes[i], Galois::NONE); jj != eejj; ++jj) {
		  GNode neighbor = graph->getEdgeDst(jj);
			int k = graph->getData(neighbor).getNodeId();//id is same as the position in nodes array
			if (visited[k] == 0) {
				queue[last++] = k;
				visited[k] = 1;
				nleft--;
			}
		}
	}
}


void randomBisection(MetisGraph* metisGraph, int* tpwgts, int coarsenTo){
	GGraph* graph = metisGraph->getGraph();
	int maxWgtPart0 = (int) (UB_FACTOR * tpwgts[0]);
	int minWgtPart0 = (int) ((1.0 / UB_FACTOR) * tpwgts[0]);
	int numNodes = metisGraph->getNumNodes();
	int nbfs = (numNodes <= coarsenTo ? SMALL_NUM_ITER_PARTITION : LARGE_NUM_ITER_PARTITION);

	vector<int> indexes(numNodes);
	for (int i = 0; i < numNodes; i++) {
		indexes[i] = i;
	}
	GNode* nodes = new GNode[numNodes];
	for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
		GNode node = *ii;
		nodes[graph->getData(node).getNodeId()] = node;
	}

	int bestMinCut = INT_MAX;
	int* bestWhere = new int[numNodes];
	for (int inbfs=0; inbfs<nbfs; inbfs++) {
		std::random_shuffle(indexes.begin(), indexes.end());
		for (int i = 0; i < numNodes; i++) {
		  graph->getData(nodes[i]).setPartition(1);
		}
		int pwgts[2] ;
		pwgts[1] = tpwgts[0]+tpwgts[1];
		pwgts[0] = 0;

		if (nbfs != 1) {
			for (int ii=0; ii<numNodes; ii++) {
				int i = indexes[ii];
				int vwgt = graph->getData(nodes[i]).getWeight();
				if (pwgts[0]+vwgt < maxWgtPart0) {
				  graph->getData(nodes[i]).setPartition(0);
					pwgts[0] += vwgt;
					pwgts[1] -= vwgt;
					if (pwgts[0] > minWgtPart0)
						break;
				}
			}
		}
		metisGraph->computeTwoWayPartitionParams();
		balanceTwoWay(metisGraph, tpwgts);
		fmTwoWayEdgeRefine(metisGraph, tpwgts, 4);

		if (inbfs==0 || bestMinCut > metisGraph->getMinCut()) {
			bestMinCut = metisGraph->getMinCut();
			for (int i = 0; i < numNodes; i++) {
			  bestWhere[i] = graph->getData(nodes[i]).getPartition();
			}
			if (bestMinCut == 0)
				break;
		}
	}
	for (int i = 0; i < numNodes; i++) {
	  graph->getData(nodes[i]).setPartition(bestWhere[i]);
	  assert(graph->getData(nodes[i]).getPartition()>=0);
	}
	delete[] bestWhere;
	metisGraph->setMinCut(bestMinCut);
	delete[] nodes;
	//assert(metisGraph->verify());
}


void growBisection(MetisGraph* metisGraph, int* tpwgts, int coarsenTo) {

	GGraph* graph = metisGraph->getGraph();
	int numNodes = metisGraph->getNumNodes();
	GNode* nodes = new GNode[numNodes];

	for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
		GNode node = *ii;
		nodes[graph->getData(node).getNodeId()] = node;
	}

	int nbfs = (numNodes <= coarsenTo ? SMALL_NUM_ITER_PARTITION : LARGE_NUM_ITER_PARTITION);

	int maxWgtPart1 = (int) UB_FACTOR * tpwgts[1];
	int minWgtPart1 = (int) (1.0 / UB_FACTOR) * tpwgts[1];

	int bestMinCut = INT_MAX;
	int* bestWhere = new int[numNodes];
	int* visited = new int[numNodes];
	int* queue = new int[numNodes];

	for (; nbfs > 0; nbfs--) {

		int pwgts[2];
		pwgts[1] = tpwgts[0] + tpwgts[1];
		pwgts[0] = 0;

		for (int i = 0; i < numNodes; i++) {
		  graph->getData(nodes[i]).setPartition(1);
		}

		bisection(graph, nodes, numNodes, minWgtPart1, maxWgtPart1, pwgts, visited, queue);
		/* Check to see if we hit any bad limiting cases */
		if (pwgts[1] == 0) {
			int i = getRandom(numNodes);
			MetisNode nodeData = graph->getData(nodes[i]);
			nodeData.setPartition(1);
			pwgts[0] += nodeData.getWeight();
			pwgts[1] -= nodeData.getWeight();
		}

		metisGraph->computeTwoWayPartitionParams();
		balanceTwoWay(metisGraph, tpwgts);
		fmTwoWayEdgeRefine(metisGraph, tpwgts, 4);

		if (bestMinCut > metisGraph->getMinCut()) {
			bestMinCut = metisGraph->getMinCut();
			for (int i = 0; i < numNodes; i++) {
			  bestWhere[i] = graph->getData(nodes[i]).getPartition();
			}
		}
	}
	delete[] visited;
	delete[] queue;
	for (int i = 0; i < numNodes; i++) {
	  graph->getData(nodes[i], Galois::NONE).setPartition(bestWhere[i]);
	  assert(graph->getData(nodes[i],Galois::NONE).getPartition()>=0);
	}
	delete[] bestWhere;
	metisGraph->setMinCut(bestMinCut);
	delete[] nodes;
	//assert(metisGraph->verify());
}

void bisection(MetisGraph* metisGraph, int* tpwgts, int coarsenTo) {

	if(metisGraph->getNumEdges()==0){
		randomBisection(metisGraph, tpwgts, coarsenTo);
	} else{
		growBisection(metisGraph, tpwgts, coarsenTo);
	}
}

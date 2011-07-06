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
#include "PQueue.h"
#include <algorithm>
#include <stdlib.h>
using namespace std;

void moveNode(PQueue* parts[],MetisGraph* metisGraph, GNode higain, int to, int* moved, GNode* swaps, int nswaps) {
	GGraph* graph = metisGraph->getGraph();
	MetisNode& higainData = higain.getData();
	higainData.setPartition(to);
	moved[higainData.getNodeId()] = nswaps;
	swaps[nswaps] = higain;

	/* Update the id[i]/ed[i] values of the affected nodes */
	higainData.swapEDAndID();
	higainData.updateGain();

	if (higainData.getEdegree() == 0 && graph->neighborsSize(higain) >0) {
		metisGraph->unsetBoundaryNode(higain);
	}

	for (GGraph::neighbor_iterator jj = graph->neighbor_begin(higain, Galois::Graph::NONE), eejj = graph->neighbor_end(higain, Galois::Graph::NONE); jj != eejj; ++jj) {
		GNode neighbor = *jj;
		MetisNode& neighborData = neighbor.getData();
		int oldgain = neighborData.getGain();
		int edgeWeight = (int) graph->getEdgeData(higain, jj);
		int kwgt = (to == neighborData.getPartition() ? edgeWeight : -edgeWeight);
		neighborData.setEdegree(neighborData.getEdegree() - kwgt);
		neighborData.setIdegree(neighborData.getIdegree() + kwgt);
		neighborData.updateGain();

		/* Update its boundary information and queue position */
		if (neighborData.isBoundary()) { /* If k was a boundary node */
			if (neighborData.getEdegree() == 0) {
				/*
				 * Not a boundary node any more
				 */
				metisGraph->unsetBoundaryNode(neighbor);
				if (moved[neighborData.getNodeId()] == -1) {
					/*
					 * Remove it if in the queues
					 */
					parts[neighborData.getPartition()]->remove(neighbor, oldgain);
				}
			} else if (moved[neighborData.getNodeId()] == -1) {
				/* If it has not been moved, update its position in the queue */
				parts[neighborData.getPartition()]->update(neighbor, oldgain, neighborData.getGain());
			}
		} else if (neighborData.getEdegree() > 0) { /*
		 * It will now become a boundary node
		 */
			metisGraph->setBoundaryNode(neighbor);
			if (moved[neighborData.getNodeId()] == -1) {
				assert(neighborData.getPartition() < 2);
				parts[neighborData.getPartition()]->insert(neighbor, neighborData.getGain());
			}
		}
	}
}

void moveBackNode(MetisGraph* metisGraph, GNode higain) {
	GGraph* graph = metisGraph->getGraph();
	MetisNode& higainData = higain.getData();

	int to = (higainData.getPartition() + 1) % 2;
	higainData.setPartition(to);
	higainData.swapEDAndID();
	higainData.updateGain();

	if (higainData.getEdegree() == 0 && higainData.isBoundary() && graph->neighborsSize(higain)>0) {
		metisGraph->unsetBoundaryNode(higain);
	} else if (higainData.getEdegree() > 0 && !higainData.isBoundary()) {
		metisGraph->setBoundaryNode(higain);
	}

	metisGraph->incPartWeight((to + 1) % 2, -higainData.getWeight());
	metisGraph->incPartWeight(to, higainData.getWeight());

	for (GGraph::neighbor_iterator jj = graph->neighbor_begin(higain, Galois::Graph::NONE), eejj = graph->neighbor_end(higain, Galois::Graph::NONE); jj != eejj; ++jj) {
		GNode neighbor = *jj;
		MetisNode& neighborData = neighbor.getData();
		int edgeWeight = (int)metisGraph->getGraph()->getEdgeData(higain, jj);
		int kwgt = (to == neighborData.getPartition() ? edgeWeight : -edgeWeight);
		neighborData.setEdegree(neighborData.getEdegree() - kwgt);
		neighborData.setIdegree(neighborData.getIdegree() + kwgt);
		neighborData.updateGain();

		if (neighborData.isBoundary() && neighborData.getEdegree() == 0) {
			metisGraph->unsetBoundaryNode(neighbor);
		}
		if (!neighborData.isBoundary() && neighborData.getEdegree() > 0) {
			metisGraph->setBoundaryNode(neighbor);
		}
	}
}

void fmTwoWayEdgeRefine(MetisGraph* metisGraph, int* tpwgts, int npasses) {
	PQueue* parts[2];
	GGraph* graph = metisGraph->getGraph();
	int numNodes = graph->size();
	GNode* swaps = new GNode[numNodes];

	int* moved = new int[numNodes];

	parts[0] = new PQueue(numNodes, metisGraph->getMaxAdjSum());
	parts[1] = new PQueue(numNodes, metisGraph->getMaxAdjSum());

	int limit = min(max((int) (0.01 * numNodes), 15), 100);
	int totalWeight = metisGraph->getPartWeight(0) + metisGraph->getPartWeight(1);

	int avgvwgt = min(totalWeight, 2 * totalWeight / numNodes);
	arrayFill(moved, numNodes, -1);

	int origdiff = abs(tpwgts[0] - metisGraph->getPartWeight(0));
	for (int pass = 0; pass < npasses; pass++) {

		parts[0]->reset();
		parts[1]->reset();
		int newcut = metisGraph->getMinCut();
		int mincut = newcut;
		int initcut = newcut;

		GNodeSet* boundaryNodeSet= metisGraph->getBoundaryNodes();
		for (GNodeSet::iterator bndIter = boundaryNodeSet->begin(); bndIter != boundaryNodeSet->end(); ++bndIter) {
			GNode boundaryNode = *bndIter;
			MetisNode& boundaryNodeData = boundaryNode.getData();
			boundaryNodeData.updateGain();
//			assert(boundaryNodeData.getNodeId()<graph->size());
			parts[boundaryNodeData.getPartition()]->insert(boundaryNode, boundaryNodeData.getGain());

		}
		int mindiff = abs(tpwgts[0] - metisGraph->getPartWeight(0));
		int mincutorder = -1;
		int nswaps = 0;
		for (; nswaps < numNodes; nswaps++) {
			int from = 1;
			int to = 0;
			if (tpwgts[0] - metisGraph->getPartWeight(0) < tpwgts[1] - metisGraph->getPartWeight(1)) {
				from = 0;
				to = 1;
			}

			if (parts[from]->size() == 0) {
				break;
			}

			GNode higain = parts[from]->getMax();

			MetisNode& higainData = higain.getData();
//			cout<<higainData.getNodeId()<<" "<<higainData.getGain()<<endl;

			newcut -= higainData.getGain();

			metisGraph->incPartWeight(from, -higainData.getWeight());
			metisGraph->incPartWeight(to, higainData.getWeight());

			if ((newcut < mincut && abs(tpwgts[0] - metisGraph->getPartWeight(0)) <= origdiff + avgvwgt)
					|| (newcut == mincut && abs(tpwgts[0] - metisGraph->getPartWeight(0)) < mindiff)) {

				mincut = newcut;
				mindiff = abs(tpwgts[0] - metisGraph->getPartWeight(0));
				mincutorder = nswaps;
			} else if (nswaps - mincutorder > limit) { /* We hit the limit, undo last move */
				newcut += (higainData.getEdegree() - higainData.getIdegree());
				metisGraph->incPartWeight(from, higainData.getWeight());
				metisGraph->incPartWeight(to, -higainData.getWeight());
				break;
			}
			moveNode(parts, metisGraph, higain, to, moved, swaps, nswaps);
		}

		/* roll back computations */
		for (int i = 0; i < nswaps; i++) {
			moved[swaps[i].getData().getNodeId()] = -1; /* reset moved array */
		}
		nswaps--;
		for (; nswaps > mincutorder; nswaps--) {
			moveBackNode(metisGraph, swaps[nswaps]);
		}
		metisGraph->setMinCut(mincut);
		if (mincutorder == -1 || mincut == initcut) {
			break;
		}

	}
	delete[] swaps;
	delete[] moved;
	delete parts[0];
	delete parts[1];
}


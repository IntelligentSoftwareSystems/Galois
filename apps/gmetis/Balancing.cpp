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

#include "MetisGraph.h"
#include "PQueue.h"
#include "GMetisConfig.h"
#include "PMetis.h"
#include <boost/unordered_set.hpp>
#include <stdlib.h>
using namespace boost;
/**
 * a blancing algorithm for bisection
 * @param metisGraph the graph to balance
 * @param tpwgts the lowerbounds of weights for the two partitions
 */

void generalTwoWayBalance(MetisGraph* metisGraph, int* tpwgts) {
	GGraph* graph = metisGraph->getGraph();
	int numNodes = graph->size();
	int* moved = new int[numNodes];

	int mindiff = abs(tpwgts[0] - metisGraph->getPartWeight(0));
	int from = 0;
	int to = 1;
	if (metisGraph->getPartWeight(0) < tpwgts[0]) {
		from = 1;
		to = 0;
	}

	PQueue queue(numNodes, metisGraph->getMaxAdjSum(), graph);

	std::fill_n(moved, numNodes, -1);
	int fromConstant=from;
	/* Insert boundary nodes in the priority queues */
	for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
		GNode node = *ii;
		MetisNode& nodeData = graph->getData(node);
		int part = nodeData.getPartition();
		if (part == fromConstant && nodeData.getWeight() <= mindiff) {
			queue.insert(node, nodeData.getGain());
		}
	}


	int mincut = metisGraph->getMinCut();

	for (int nswaps = 0; nswaps < numNodes; nswaps++) {

		if (queue.size() == 0)
			break;

		GNode higain = queue.getMax();
		MetisNode& higainData = graph->getData(higain);
		if (metisGraph->getPartWeight(to) + higainData.getWeight() > tpwgts[to]) {
			break;
		}
		mincut -= (higainData.getEdegree() - higainData.getIdegree());
		metisGraph->incPartWeight(from, -higainData.getWeight());
		metisGraph->incPartWeight(to, higainData.getWeight());

		higainData.setPartition(to);
		moved[higainData.getNodeId()] = nswaps;

		/* Update the id[i]/ed[i] values of the affected nodes */
		higainData.swapEDAndID();

		if (higainData.getEdegree() == 0 && higainData.isBoundary() && graph->edge_begin(higain) != graph->edge_end(higain)) {
			metisGraph->unsetBoundaryNode(higain);
		}
		if (higainData.getEdegree() > 0 && !higainData.isBoundary()) {
			metisGraph->setBoundaryNode(higain);
		}

		for (GGraph::edge_iterator jj = graph->edge_begin(higain, Galois::NONE), eejj = graph->edge_end(higain, Galois::NONE); jj != eejj; ++jj) {
		  GNode neighbor = graph->getEdgeDst(jj);

			MetisNode& neighborData = graph->getData(neighbor);
			int oldgain = neighborData.getGain();
			int edgeWeight = (int)graph->getEdgeData(jj);
			int kwgt = (to == neighborData.getPartition() ? edgeWeight : -edgeWeight);
			neighborData.setEdegree(neighborData.getEdegree() - kwgt);
			neighborData.setIdegree(neighborData.getIdegree() + kwgt);
			neighborData.updateGain();
			/* Update the queue position */
			if (moved[neighborData.getNodeId()] == -1 && neighborData.getPartition() == fromConstant
					&& neighborData.getWeight() <= mindiff) {
				queue.update(neighbor, oldgain, neighborData.getGain());
			}
			/* Update its boundary information */
			if (neighborData.getEdegree() == 0 && neighborData.isBoundary()) {
				metisGraph->unsetBoundaryNode(neighbor);
			} else if (neighborData.getEdegree() > 0 && !neighborData.isBoundary()) {
				metisGraph->setBoundaryNode(neighbor);
			}
		}

	}
	delete[] moved;
	metisGraph->setMinCut(mincut);
}

void boundaryTwoWayBalance(MetisGraph* metisGraph, int* tpwgts) {

	GGraph* graph = metisGraph->getGraph();
	int numNodes = graph->size();
	int* moved = new int[numNodes];
	std::fill_n(moved, numNodes, -1);
	int mindiff = abs(tpwgts[0] - metisGraph->getPartWeight(0));
	int from = 0;
	int to = 1;
	if (metisGraph->getPartWeight(0) < tpwgts[0]) {
		from = 1;
		to = 0;
	}
	PQueue queue(numNodes, metisGraph->getMaxAdjSum(),graph);

	for(GNodeSet::iterator iter = metisGraph->getBoundaryNodes()->begin();iter != metisGraph->getBoundaryNodes()->end();++iter){
		GNode boundaryNode = *iter;
		MetisNode& boundaryNodeData = graph->getData(boundaryNode);
		boundaryNodeData.updateGain();
		if (boundaryNodeData.getPartition() == from && boundaryNodeData.getWeight() <= mindiff) {
			queue.insert(boundaryNode, boundaryNodeData.getGain());
		}
	}
	int mincut = metisGraph->getMinCut();
	for (int nswaps = 0; nswaps < numNodes; nswaps++) {
		if (queue.size() == 0)
			break;

		GNode higain = queue.getMax();
		MetisNode& higainData = graph->getData(higain);
		if (metisGraph->getPartWeight(to) + higainData.getWeight() > tpwgts[to]) {
			break;
		}
		mincut -= (higainData.getEdegree() - higainData.getIdegree());
		metisGraph->incPartWeight(from, -higainData.getWeight());
		metisGraph->incPartWeight(to, higainData.getWeight());

		higainData.setPartition(to);
		moved[higainData.getNodeId()] = nswaps;

		/* Update the id[i]/ed[i] values of the affected nodes */
		higainData.swapEDAndID();
		higainData.updateGain();
		if (higainData.getEdegree() == 0 && graph->edge_begin(higain) != graph->edge_end(higain)) {
			metisGraph->unsetBoundaryNode(higain);
		}

		int fromConstant=from;
		for (GGraph::edge_iterator jj = graph->edge_begin(higain, Galois::NONE), eejj = graph->edge_end(higain, Galois::NONE); jj != eejj; ++jj) {
		  GNode neighbor = graph->getEdgeDst(jj);
			MetisNode& neighborData = graph->getData(neighbor);
			int oldgain = neighborData.getGain();
			int edgeWeight = (int)graph->getEdgeData(jj);
			int kwgt = (to == neighborData.getPartition() ? edgeWeight : -edgeWeight);
			neighborData.setEdegree(neighborData.getEdegree() - kwgt);
			neighborData.setIdegree(neighborData.getIdegree() + kwgt);
			neighborData.updateGain();

			/* Update its boundary information and queue position */
			if (neighborData.isBoundary()) { /* If k was a boundary vertex */
				if (neighborData.getEdegree() == 0) { /* Not a boundary vertex any more */
					metisGraph->unsetBoundaryNode(neighbor);
					/* Remove it if in the queues */
					if (moved[neighborData.getNodeId()] == -1 && neighborData.getPartition() == fromConstant
							&& neighborData.getWeight() <= mindiff) {
						queue.remove(neighbor, oldgain);
					}
				} else if (moved[neighborData.getNodeId()] == -1 && neighborData.getPartition() == fromConstant
						&& neighborData.getWeight() <= mindiff) {
					/* If it has not been moved, update its position in the queue */
					queue.update(neighbor, oldgain, neighborData.getGain());
				}
			} else if (neighborData.getEdegree() > 0) { /* It will now become a boundary vertex */
				metisGraph->setBoundaryNode(neighbor);
				if (moved[neighborData.getNodeId()] == -1 && neighborData.getPartition() == fromConstant
						&& neighborData.getWeight() <= mindiff) {
					queue.insert(neighbor, neighborData.getGain());
				}
			}
		}
	}
	delete[] moved;
	metisGraph->setMinCut(mincut);
}

void balanceTwoWay(MetisGraph* metisGraph, int* tpwgts) {
	int pwgts0 = metisGraph->getPartWeight(0);
	int pwgts1 = metisGraph->getPartWeight(1);

	int mindiff = abs(tpwgts[0] - pwgts0);
	if (mindiff < 3 * (pwgts0 + pwgts1) / metisGraph->getNumNodes()) {
		return;
	}
	if (pwgts0 > tpwgts[0] && pwgts0 < (int) (UB_FACTOR * tpwgts[0])) {
		return;
	}
	if (pwgts1 > tpwgts[1] && pwgts1 < (int) (UB_FACTOR * tpwgts[1])) {
		return;
	}

	if (metisGraph->getNumOfBoundaryNodes() > 0) {
		boundaryTwoWayBalance(metisGraph, tpwgts);
	} else {
		generalTwoWayBalance(metisGraph, tpwgts);
	}
}




void greedyKWayEdgeBalance(MetisGraph* metisGraph, int nparts, float* tpwgts, float ubfactor,
		int npasses) {
	int* minwgts = new int[nparts];
	int* maxwgts = new int[nparts];
	int* itpwgts = new int[nparts];
	int tvwgt = 0;
	for (int i = 0; i < nparts; i++) {
		tvwgt += metisGraph->getPartWeight(i);
	}
	for (int i = 0; i < nparts; i++) {
		itpwgts[i] = (int) (tpwgts[i] * tvwgt);
		maxwgts[i] = (int) (tpwgts[i] * tvwgt * ubfactor);
		minwgts[i] = (int) (tpwgts[i] * tvwgt * (1.0 / ubfactor));
	}
	GGraph* graph = metisGraph->getGraph();

	PQueue queue(metisGraph->getNumNodes(), metisGraph->getMaxAdjSum(), graph);
	int* moved = new int[metisGraph->getNumNodes()];

	for (int pass = 0; pass < npasses; pass++) {

		int i = 0;
		for (; i < nparts; i++) {
			if (metisGraph->getPartWeight(i) > maxwgts[i]) {
				break;
			}
		}
		if (i == nparts)
			break;
		int graphSize = metisGraph->getNumNodes();
		std::fill_n(moved, graphSize, -1);
		queue.reset();

		GNodeSet* boundaryNodeSet= metisGraph->getBoundaryNodes();
		for (GNodeSet::iterator bndIter = boundaryNodeSet->begin(); bndIter != boundaryNodeSet->end(); ++bndIter) {
			GNode boundaryNode = *bndIter;
			MetisNode& boundaryNodeData = graph->getData(boundaryNode);
			boundaryNodeData.updateGain();
			queue.insert(boundaryNode, boundaryNodeData.getGain());

			moved[boundaryNodeData.getNodeId()] = 2;
		}

		while (true) {
			if (queue.size()==0)
				break;
			GNode higain = queue.getMax();
			MetisNode& higainData = graph->getData(higain);
			assert(higainData.getNodeId()<graphSize);
			moved[higainData.getNodeId()] = 1;
			int from = higainData.getPartition();
			if (metisGraph->getPartWeight(from) - higainData.getWeight() < minwgts[from])
				continue; /* This cannot be moved! */
			int k = 0;
			for (; k < higainData.getNDegrees(); k++) {
				int to = higainData.getPartIndex()[k];
				if (metisGraph->getPartWeight(to) + higainData.getWeight() <= maxwgts[to]
				                                                                      || itpwgts[from] * (metisGraph->getPartWeight(to) + higainData.getWeight()) <= itpwgts[to]* metisGraph->getPartWeight(from))
					break;
			}
			if (k == higainData.getNDegrees())
				continue; /* break out if you did not find a candidate */

			for (int j = k + 1; j < higainData.getNDegrees(); j++) {
				int to = higainData.getPartIndex()[j];
				if (itpwgts[higainData.getPartIndex()[k]] * metisGraph->getPartWeight(to) < itpwgts[to]
				                                                                               * metisGraph->getPartWeight(higainData.getPartIndex()[k]))
					k = j;
			}
			assert (k<nparts);
			int to = higainData.getPartIndex()[k];

			if (metisGraph->getPartWeight(from) < maxwgts[from] && metisGraph->getPartWeight(to) > minwgts[to]
			                                                                                               && higainData.getPartEd()[k] - higainData.getIdegree() < 0)
				continue;

			/*=====================================================================
			 * If we got here, we can now move the vertex from 'from' to 'to'
			 *======================================================================*/
			metisGraph->setMinCut(metisGraph->getMinCut() - (higainData.getPartEd()[k] - higainData.getIdegree()));

			/* Update where, weight, and ID/ED information of the vertex you moved */
			higainData.setPartition(to);
			metisGraph->incPartWeight(to, higainData.getWeight());
			metisGraph->incPartWeight(from, -higainData.getWeight());
			higainData.setEdegree(higainData.getEdegree() - higainData.getPartEd()[k] + higainData.getIdegree());
			int temp = higainData.getPartEd()[k];
			higainData.getPartEd()[k] = higainData.getIdegree();
			higainData.setIdegree(temp);

			if (higainData.getPartEd()[k] == 0) {
				higainData.setNDegrees(higainData.getNDegrees() - 1);
				higainData.getPartEd()[k] = higainData.getPartEd()[higainData.getNDegrees()];
				higainData.getPartIndex()[k] = higainData.getPartIndex()[higainData.getNDegrees()];
			} else {
				higainData.getPartIndex()[k] = from;
			}

			if (higainData.getEdegree() == 0) {
				metisGraph->unsetBoundaryNode(higain);
			}

			/* Update the degrees of adjacent vertices */
			for (GGraph::edge_iterator jj = graph->edge_begin(higain, Galois::NONE), eejj = graph->edge_end(higain, Galois::NONE); jj != eejj; ++jj) {
			  GNode neighbor = graph->getEdgeDst(jj);
			  MetisNode& neighborData = graph->getData(neighbor,Galois::NONE);
				assert(neighborData.getNodeId()<graphSize);
				int oldgain = neighborData.getGain();
				if (neighborData.getPartEd().size() == 0) {
					int numEdges = neighborData.getNumEdges();
					neighborData.initPartEdAndIndex(numEdges);
				}
				int edgeWeight = graph->getEdgeData(jj);
				if (neighborData.getPartition() == from) {
					neighborData.setEdegree(neighborData.getEdegree() + edgeWeight);
					neighborData.setIdegree(neighborData.getIdegree() - edgeWeight);
					if (neighborData.getEdegree() > 0 && !neighborData.isBoundary()){
						metisGraph->setBoundaryNode(neighbor);
					}
				} else if (neighborData.getPartition() == to) {
					neighborData.setEdegree(neighborData.getEdegree() - edgeWeight);
					neighborData.setIdegree(neighborData.getIdegree() + edgeWeight);
					if (neighborData.getEdegree() == 0 && neighborData.isBoundary()){
						metisGraph->unsetBoundaryNode(neighbor);
					}
				}

				/* Remove contribution from the .ed of 'from' */
				if (neighborData.getPartition() != from) {
					for (int k = 0; k < neighborData.getNDegrees(); k++) {
						if (neighborData.getPartIndex()[k] == from) {
							if (neighborData.getPartEd()[k] == edgeWeight) {
								neighborData.setNDegrees(neighborData.getNDegrees() - 1);
								neighborData.getPartEd()[k] = neighborData.getPartEd()[neighborData.getNDegrees()];
								neighborData.getPartIndex()[k] = neighborData.getPartIndex()[neighborData.getNDegrees()];
							} else {
								neighborData.getPartEd()[k] -= edgeWeight;
							}
							break;
						}
					}
				}

				/*
				 * add contribution to the .ed of 'to'
				 */

				if (neighborData.getPartition() != to) {
					int k = 0;
					for (k = 0; k < neighborData.getNDegrees(); k++) {
						if (neighborData.getPartIndex()[k] == to) {
							neighborData.getPartEd()[k] += edgeWeight;
							break;
						}
					}
					if (k == neighborData.getNDegrees()) {
						int nd = neighborData.getNDegrees();
						neighborData.getPartIndex()[nd] = to;
						neighborData.getPartEd()[nd++] = edgeWeight;
						neighborData.setNDegrees(nd);
					}
				}

				/* Update the queue */
				if (neighborData.getPartition() == from || neighborData.getPartition() == to) {
					neighborData.updateGain();
					if (moved[neighborData.getNodeId()] == 2) {
						if (neighborData.getEdegree() > 0) {
							queue.update(neighbor, oldgain, neighborData.getGain());
						} else {
							queue.remove(neighbor, oldgain);
							moved[neighborData.getNodeId()] = -1;
						}
					} else if (moved[neighborData.getNodeId()] == -1 && neighborData.getEdegree() > 0) {
						queue.insert(neighbor, neighborData.getGain());
						moved[neighborData.getNodeId()] = 2;
					}
				}
			}
		}
	}
//	metisGraph->verify();
	delete[] moved;
	delete[] itpwgts;
	delete[] maxwgts;
	delete[] minwgts;
}

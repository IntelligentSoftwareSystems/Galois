/*
 * PMetisRefining.cpp
 *
 *  Created on: Jun 16, 2011
 *      Author: xinsui
 */

#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "defs.h"

/**
 * project the partitioning information back the finer graph
 */
void projectTwoWayPartition(MetisGraph* metisGraph) {

	MetisGraph* finer = metisGraph->getFinerGraph();
	finer->setMinCut(metisGraph->getMinCut());
	finer->initPartWeight(2);
	finer->setPartWeight(0, metisGraph->getPartWeight(0));
	finer->setPartWeight(1, metisGraph->getPartWeight(1));
	GGraph* finerGraph = finer->getGraph();

	for (GGraph::active_iterator ii = finerGraph->active_begin(), ee = finerGraph->active_end(); ii != ee; ++ii) {
		GNode node = *ii;
		MetisNode& nodeData = node.getData();
		nodeData.setPartition(finer->getCoarseGraphMap(nodeData.getNodeId()).getData().getPartition());
		assert(nodeData.getPartition()>=0);
		nodeData.setEdegree(0);
		nodeData.setIdegree(0);
		finer->unsetBoundaryNode(node);
	}

	for (GGraph::active_iterator ii = finerGraph->active_begin(), ee = finerGraph->active_end(); ii != ee; ++ii) {
		GNode node = *ii;

		MetisNode& nodeData = node.getData();
		nodeData.setIdegree(nodeData.getAdjWgtSum());
		if (finerGraph->neighborsSize(node)!=0 && finer->getCoarseGraphMap(nodeData.getNodeId()).getData().isBoundary()) {
			for (GGraph::neighbor_iterator jj = finerGraph->neighbor_begin(node, Galois::Graph::NONE), eejj = finerGraph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
				GNode neighbor = *jj;
				MetisNode& neighborData = neighbor.getData();
				if (nodeData.getPartition() != neighborData.getPartition()) {
					nodeData.setEdegree(nodeData.getEdegree() + (int)finerGraph->getEdgeData(node, neighbor));
				}
			}
		}
		if (finerGraph->neighborsSize(node)!=0) {
			nodeData.setIdegree(nodeData.getIdegree() - nodeData.getEdegree());
		}
		if (finerGraph->neighborsSize(node)==0 || nodeData.getEdegree() > 0) {
			finer->setBoundaryNode(node);
		}
	}
}


void refineTwoWay(MetisGraph* metisGraph, MetisGraph* orgGraph, int* tpwgts) {

	metisGraph->computeTwoWayPartitionParams();
	while (metisGraph != orgGraph) {
		balanceTwoWay(metisGraph, tpwgts);
		fmTwoWayEdgeRefine(metisGraph, tpwgts, 8);
		projectTwoWayPartition(metisGraph);
		MetisGraph* coarseGraph = metisGraph;
		metisGraph = metisGraph->getFinerGraph();
		delete coarseGraph->getGraph();
		delete coarseGraph;
//		cout<<"pmetis verfiy:"<<endl;
//		metisGraph->verify();

	}
	balanceTwoWay(metisGraph, tpwgts);
	fmTwoWayEdgeRefine(metisGraph, tpwgts, 8);
}


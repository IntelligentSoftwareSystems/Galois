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
	finer->initBoundarySet();

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
			for (GGraph::neighbor_iterator jj = finerGraph->neighbor_begin(node, Galois::NONE), eejj = finerGraph->neighbor_end(node, Galois::NONE); jj != eejj; ++jj) {
				GNode neighbor = *jj;
				MetisNode& neighborData = neighbor.getData();
				if (nodeData.getPartition() != neighborData.getPartition()) {
					nodeData.setEdegree(nodeData.getEdegree() + (int)finerGraph->getEdgeData(node, jj));
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
		metisGraph->releaseCoarseGraphMap();
		delete coarseGraph->getGraph();
		delete coarseGraph;
//		cout<<"pmetis verfiy:"<<endl;
//		metisGraph->verify();

	}
	balanceTwoWay(metisGraph, tpwgts);
	fmTwoWayEdgeRefine(metisGraph, tpwgts, 8);
}


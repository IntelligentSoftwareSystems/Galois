/*
 * KMetisRefining.cpp
 *
 *  Created on: Jun 16, 2011
 *      Author: xinsui
 */

#include "RandomKWayRefiner.h"
#include "defs.h"
#include "GMetisConfig.h"
void projectNeighbors(GGraph* graph, GNode node, int* map, int& ndegrees, int& ed) {
	MetisNode& nodeData = node.getData(Galois::Graph::NONE);
	for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE), eejj = graph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
			GNode neighbor = *jj;
			MetisNode& neighborData = neighbor.getData(Galois::Graph::NONE);
			if (nodeData.getPartition() != neighborData.getPartition()) {
			int edgeWeight = graph->getEdgeData(node, neighbor, Galois::Graph::NONE);
			ed += edgeWeight;
			int index = map[neighborData.getPartition()];
			if (index == -1) {
				map[neighborData.getPartition()] = ndegrees;
				nodeData.getPartIndex()[ndegrees] = neighborData.getPartition();
				nodeData.getPartEd()[ndegrees] += edgeWeight;
				ndegrees++;
			} else {
				nodeData.getPartEd()[index] += edgeWeight;
			}
		}
	}
}
struct projectInfo {
	int nparts;
	GGraph* graph;
	MetisGraph* finer;
	projectInfo(int nparts, MetisGraph* finer){
		this->finer = finer;
		this->nparts = nparts;
		this->graph = finer->getGraph();
	}
	template<typename Context>
	void operator()(GNode node, Context& lwl) {
		MetisNode& nodeData = node.getData(Galois::Graph::NONE);
		int numEdges = graph->neighborsSize(node, Galois::Graph::NONE);
//		nodeData.partIndex = new int[numEdges];
//		nodeData.partEd = new int[numEdges];
		nodeData.initPartEdAndIndex(numEdges);
		nodeData.setIdegree(nodeData.getAdjWgtSum());
		if (finer->getCoarseGraphMap(nodeData.getNodeId()).getData(Galois::Graph::NONE).getEdegree() > 0) {
			int* map = new int[nparts];
			arrayFill(map,nparts,-1);
			int ed = 0;
			int ndegrees = 0;
			projectNeighbors(graph, node, map, ndegrees, ed);
			nodeData.setEdegree(ed);
			nodeData.setIdegree(nodeData.getIdegree() - nodeData.getEdegree());

//			if (nodeData.getEdegree() - nodeData.getIdegree() >= 0)
//				finer->setBoundaryNode(node);

			nodeData.setNDegrees(ndegrees);
			delete[] map;
		}
	}
};

void computeKWayPartInfo(int nparts, MetisGraph* finer,
		GGraph* coarseGraph, GGraph* graph){
	projectInfo pi(nparts, finer);
	Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<64> >(graph->active_begin(), graph->active_end(), pi);
}

void projectKWayPartition(MetisGraph* metisGraph, int nparts){
	MetisGraph* finer = metisGraph->getFinerGraph();
	GGraph* coarseGraph = metisGraph->getGraph();
	GGraph* graph = finer->getGraph();

	for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
		GNode node = *ii;
		MetisNode& nodeData = node.getData();
		nodeData.setPartition(finer->getCoarseGraphMap(nodeData.getNodeId()).getData().getPartition());
//		cout<<"nodeId:"<<nodeData.getNodeId()<<" "<<finer->getCoarseGraphMap(nodeData.getNodeId()).getData().getPartition()<<endl;

		assert(nodeData.getPartition()>=0);
	}
	computeKWayPartInfo(nparts, finer, coarseGraph, graph);
	for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
		GNode node = *ii;
		MetisNode& nodeData = node.getData(Galois::Graph::NONE);
		if (finer->getCoarseGraphMap(nodeData.getNodeId()).getData(Galois::Graph::NONE).getEdegree() > 0) {
			if (nodeData.getEdegree() - nodeData.getIdegree() >= 0)
				finer->setBoundaryNode(node);
		}
	}


	finer->initPartWeight(nparts);
	for (int i = 0; i < nparts; i++) {
		finer->setPartWeight(i, metisGraph->getPartWeight(i));
	}
	finer->setMinCut(metisGraph->getMinCut());
}

void refineKWay(MetisGraph* metisGraph, MetisGraph* orgGraph, float* tpwgts, float ubfactor, int nparts){
	metisGraph->computeKWayPartitionParams(nparts);
	int nlevels = 0;
	MetisGraph* metisGraphTemp = metisGraph;
	while (metisGraphTemp!=orgGraph) {
		metisGraphTemp = metisGraphTemp->getFinerGraph();
		nlevels++;
	}
	int i = 0;
	RandomKwayEdgeRefiner rkRefiner(tpwgts, nparts, ubfactor, 10, 1);
	metisGraph->verify();

	while (metisGraph != orgGraph) {
		if (2 * i >= nlevels && !metisGraph->isBalanced(tpwgts, (float) 1.04 * ubfactor)) {
			metisGraph->computeKWayBalanceBoundary();
			cout<<"refine once:"<<endl;
			greedyKWayEdgeBalance(metisGraph, nparts, tpwgts, ubfactor, 8);
			metisGraph->computeKWayBoundary();
		}

		rkRefiner.refine(metisGraph);
		projectKWayPartition(metisGraph, nparts);
		MetisGraph* coarseGraph = metisGraph;
		metisGraph = metisGraph->getFinerGraph();
		delete coarseGraph->getGraph();
		delete coarseGraph;
		i++;
//		metisGraph->verify();
	}
	if (2 * i >= nlevels && !metisGraph->isBalanced(tpwgts, (float) 1.04 * ubfactor)) {
		metisGraph->computeKWayBalanceBoundary();
		greedyKWayEdgeBalance(metisGraph, nparts, tpwgts, ubfactor, 8);
		metisGraph->computeKWayBoundary();
	}
	rkRefiner.refine(metisGraph);

	if (!metisGraph->isBalanced(tpwgts, ubfactor)) {
		metisGraph->computeKWayBalanceBoundary();
		greedyKWayEdgeBalance(metisGraph, nparts, tpwgts, ubfactor, 8);
		rkRefiner.refine(metisGraph);
	}
}


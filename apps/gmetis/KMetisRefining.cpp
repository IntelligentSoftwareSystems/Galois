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

#include "RandomKWayRefiner.h"
#include "defs.h"
#include "GMetisConfig.h"
void projectNeighbors(GGraph* graph, int nparts, GNode node, int& ndegrees, int& ed) {
	MetisNode& nodeData = node.getData(Galois::Graph::NONE);
	int map[nparts];
	arrayFill(map,nparts,-1);
	for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE), eejj = graph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
		GNode neighbor = *jj;
		MetisNode& neighborData = neighbor.getData(Galois::Graph::NONE);
		if (nodeData.getPartition() != neighborData.getPartition()) {
			int edgeWeight = graph->getEdgeData(node, jj, Galois::Graph::NONE);
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
//		int numEdges = graph->neighborsSize(node, Galois::Graph::NONE);
//		nodeData.partIndex = new int[numEdges];
//		nodeData.partEd = new int[numEdges];
//		nodeData.initPartEdAndIndex(numEdges);
		nodeData.setIdegree(nodeData.getAdjWgtSum());
		if (finer->getCoarseGraphMap(nodeData.getNodeId()).getData(Galois::Graph::NONE).getEdegree() > 0) {
//			int* map = new int[nparts];

			int ed = 0;
			int ndegrees = 0;
			projectNeighbors(graph,  nparts, node,ndegrees, ed);
			nodeData.setEdegree(ed);
			nodeData.setIdegree(nodeData.getIdegree() - nodeData.getEdegree());
//			if (nodeData.getEdegree() - nodeData.getIdegree() >= 0)
//				finer->setBoundaryNode(node);
			nodeData.setNDegrees(ndegrees);
//			delete[] map;
		}
	}
};

struct projectPartition {
	GGraph* graph;
	MetisGraph* finer;
	projectPartition(MetisGraph* finer){
		this->graph = finer->getGraph();
		this->finer = finer;
	}
	template<typename Context>
	void operator()(GNode node, Context& lwl) {
		MetisNode& nodeData = node.getData(Galois::Graph::NONE);
		nodeData.setPartition(finer->getCoarseGraphMap(nodeData.getNodeId()).getData(Galois::Graph::NONE).getPartition());
//		nodeData.initPartEdAndIndex(nodeData.getNumEdges());
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
	finer->initBoundarySet();
//	projectPartition pp(finer);
//	Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<64> >(graph->active_begin(), graph->active_end(), pp);

	for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
		GNode node = *ii;
		MetisNode& nodeData = node.getData();
		nodeData.setPartition(finer->getCoarseGraphMap(nodeData.getNodeId()).getData().getPartition());
		nodeData.initPartEdAndIndex(nodeData.getNumEdges());
		assert(nodeData.getPartition()>=0);
	}
	Galois::Timer p_Time;
	p_Time.start();
	computeKWayPartInfo(nparts, finer, coarseGraph, graph);
	p_Time.stop();
	cout<<"------------projectParallel:"<<p_Time.get()<<endl;
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

	while (metisGraph != orgGraph) {
		if (2 * i >= nlevels && !metisGraph->isBalanced(tpwgts, (float) 1.04 * ubfactor)) {
			metisGraph->computeKWayBalanceBoundary();
			greedyKWayEdgeBalance(metisGraph, nparts, tpwgts, ubfactor, 8);
			metisGraph->computeKWayBoundary();
		}

		rkRefiner.refine(metisGraph);
		Galois::Timer p_Time;
		p_Time.start();
		projectKWayPartition(metisGraph, nparts);
		p_Time.stop();
		cout<<"project time:"<<p_Time.get()<<endl;
		MetisGraph* coarseGraph = metisGraph;
		metisGraph = metisGraph->getFinerGraph();
		metisGraph->releaseCoarseGraphMap();
		delete coarseGraph->getGraph();
		delete coarseGraph;

		i++;
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


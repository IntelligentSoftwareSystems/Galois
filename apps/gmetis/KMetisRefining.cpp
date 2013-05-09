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
#include "RandomRefiner.h"
#include "defs.h"
#include "GMetisConfig.h"
void projectNeighbors(GGraph* graph, int nparts, GNode node, int& ndegrees, int& ed,int &id) {
	if(!variantMetis::noPartInfo){
	MetisNode& nodeData = graph->getData(node, Galois::MethodFlag::NONE);
	int map[nparts];
	for(int i=0;i<nparts;i++)
		map[i]=-1;

	for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::MethodFlag::NONE), eejj = graph->edge_end(node, Galois::MethodFlag::NONE); jj != eejj; ++jj) {
		GNode neighbor = graph->getEdgeDst(jj);
		MetisNode& neighborData = graph->getData(neighbor,Galois::MethodFlag::NONE);
		int edgeWeight = graph->getEdgeData(jj, Galois::MethodFlag::NONE);
		if (nodeData.getPartition() != neighborData.getPartition()) {
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
		}else {
			id+=edgeWeight;
		}
	}
	}
	else {
	MetisNode& nodeData = graph->getData(node, Galois::MethodFlag::NONE);

	for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::MethodFlag::NONE), eejj = graph->edge_end(node, Galois::MethodFlag::NONE); jj != eejj; ++jj) {
		GNode neighbor = graph->getEdgeDst(jj);
		MetisNode& neighborData = graph->getData(neighbor,Galois::MethodFlag::NONE);
		int edgeWeight = graph->getEdgeData(jj, Galois::MethodFlag::NONE);
		if(nodeData.getPartition()!=neighborData.getPartition()) {
			ed+=edgeWeight;
		}
		else {
			id+=edgeWeight;
		}
	}
	}


}
struct projectInfo {
	int nparts;

	MetisGraph* finerMetisGraph;
	MetisGraph *coarseMetisGraph;
	GGraph *coarseGGraph;
	GGraph *finerGGraph;
	projectInfo(int nparts, MetisGraph* finerMetisGraph,MetisGraph *coarseMetisGraph){
		this->nparts = nparts;
		this->finerMetisGraph = finerMetisGraph;
		this->coarseMetisGraph = coarseMetisGraph;
		this->coarseGGraph = coarseMetisGraph->getGraph();
		this->finerGGraph = finerMetisGraph->getGraph();

	}
	template<typename Context>
	void operator()(GNode node, Context& lwl) {

		MetisNode& nodeData = finerGGraph->getData(node,Galois::MethodFlag::NONE);
		nodeData.setIdegree(nodeData.getAdjWgtSum());
		GNode multiNode;
		if(variantMetis::localNodeData){
			multiNode = static_cast<GNode>(nodeData.multiNode);
		}else {
			multiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
		}
		MetisNode &multiNodeData = coarseGGraph->getData(multiNode,Galois::MethodFlag::NONE);

		if (multiNodeData.getEdegree() > 0)
		{
			int ed = 0;
			int ndegrees = 0;
			int id =0;
			projectNeighbors(finerGGraph,  nparts, node,ndegrees, ed,id);
			nodeData.setEdegree(ed);
			nodeData.setIdegree(id);
			if(!variantMetis::noPartInfo)
				nodeData.setNDegrees(ndegrees);
		}
	}
};

struct projectPartition {
	MetisGraph* finerMetisGraph;
	MetisGraph *coarseMetisGraph;
	GGraph* finerGGraph;
	GGraph* coarseGGraph;
	int nparts;
	projectPartition(MetisGraph* finer,MetisGraph *coarseGraph,int nparts){
		this->finerMetisGraph = finer;
		this->coarseMetisGraph = coarseGraph;
		this->coarseGGraph = coarseMetisGraph->getGraph();
		this->finerGGraph = finerMetisGraph->getGraph();
		this->nparts = nparts;

	}
	//template<typename Context>
	void operator()(GNode node) {
		MetisNode& nodeData = finerGGraph->getData(node,Galois::MethodFlag::NONE);
		GNode multiNode;
		if(variantMetis::localNodeData){
			multiNode = static_cast<GNode>(nodeData.multiNode);
		}else {
			multiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
		}
		MetisNode &multiNodeData = coarseGGraph->getData(multiNode,Galois::MethodFlag::NONE);
		nodeData.setPartition(multiNodeData.getPartition());
	}
};


void projectKWayPartition(MetisGraph* coarseMetisGraph, int nparts){

	MetisGraph* finerMetisGraph = coarseMetisGraph->getFinerGraph();
	//GGraph* coarseGGraph = coarseMetisGraph->getGraph();
	GGraph* finerGGraph = finerMetisGraph->getGraph();
	finerMetisGraph->initBoundarySet();
	Galois::Timer t;
	t.start();
	projectPartition pp(finerMetisGraph,coarseMetisGraph,nparts);
	//Galois::for_each_local<Galois::WorkList::dChunkedFIFO<64, GNode> >(*finerGGraph, pp,"Project Partition");
	Galois::do_all_local(*finerGGraph,pp,"project partition");
	t.stop();
	//cout<<"project partition "<<t.get()<<" ms"<<endl;
	projectInfo pi(nparts,finerMetisGraph,coarseMetisGraph);
	Galois::for_each_local<Galois::WorkList::ChunkedLIFO<64, GNode> >(*finerGGraph, pi,"Projecting Info");
	t.stop();
	//cout<<"projecting info "<<t.get()<<" ms "<<endl;
	t.start();
	finerMetisGraph->initPartWeight(nparts);

	for (int i = 0; i < nparts; i++) {
		finerMetisGraph->setPartWeight(i, coarseMetisGraph->getPartWeight(i));
	}

	for(GGraph::iterator ii = finerGGraph->begin(),ee=finerGGraph->end();ii!=ee;ii++) {
		GNode node = *ii;
		MetisNode &nodeData = finerGGraph->getData(node);
		if(nodeData.getEdegree()-nodeData.getIdegree()>0) {
			finerMetisGraph->setBoundaryNode(node);
		}
	}
	finerMetisGraph->setMinCut(coarseMetisGraph->getMinCut());
	t.stop();
	//cout<<"Waste time "<<t.get()<<" ms"<<endl;
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
			Galois::Timer t;
			t.start();
			metisGraph->computeKWayBalanceBoundary();
			greedyKWayEdgeBalance(metisGraph, nparts, tpwgts, ubfactor, 8);
			metisGraph->computeKWayBoundary();
			t.stop();
			cout<<"Balance and computing boundary "<<t.get()<<" ms "<<endl;
		}
		Galois::Timer t;
		t.start();
		rkRefiner.refine(metisGraph);
		t.stop();
		cout<<"Refine "<<t.get()<<" ms "<<endl;
		projectKWayPartition(metisGraph, nparts);

		MetisGraph* coarseGraph = metisGraph;
		metisGraph = metisGraph->getFinerGraph();

		if(!variantMetis::localNodeData)
		metisGraph->releaseCoarseGraphMap();

		delete coarseGraph->getGraph();
		delete coarseGraph;
		i++;
	}
	/*
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
	}*/
}


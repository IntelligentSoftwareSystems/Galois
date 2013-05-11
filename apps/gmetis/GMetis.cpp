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

#include <vector>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "PMetis.h"

#include "Galois/Graph/LCGraph.h"
#include "Galois/Statistic.h"
#include "GraphReader.h"
#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* name = "GMetis";
static const char* desc = "Partitions a graph into K parts and minimizing the graph cut";
static const char* url = "gMetis";

static cll::opt<bool> mtxInput("mtxinput", cll::desc("Use text mtx files instead binary based ones"), cll::init(false));
static cll::opt<bool> weighted("weighted", cll::desc("weighted"), cll::init(false));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> numPartitions(cll::Positional, cll::desc("<Number of partitions>"), cll::Required);

bool verifyCoarsening(MetisGraph *metisGraph) {

	if(metisGraph == NULL)
		return true;
	cout<<endl<<"##### Verifying coarsening #####"<<endl;
	int matchedCount=0;
	int unmatchedCount=0;
	GGraph *graph = metisGraph->getGraph();

	for(GGraph::iterator ii=graph->begin(),ee=graph->end();ii!=ee;ii++) {
		GNode node = *ii;
		MetisNode &nodeData = graph->getData(node);
		GNode matchNode;
#ifdef localNodeData
		if(!nodeData.isMatched())
			return false;
		matchNode = static_cast<GNode>(nodeData.getMatchNode());
#else
		if(!metisGraph->isMatched(nodeData.getNodeId())) {
			return false;
		}
		matchNode = metisGraph->getMatch(nodeData.getNodeId());
#endif

		if(matchNode == node) {
			unmatchedCount++;
		}
		else{
			matchedCount++;
			MetisNode &matchNodeData = graph->getData(matchNode);
			GNode mmatch;
#ifdef localNodeData
			if(!matchNodeData.isMatched())
				return false;
			mmatch = static_cast<GNode>(matchNodeData.getMatchNode());
#else
			if(!metisGraph->isMatched(matchNodeData.getNodeId()))
				return false;
			mmatch = metisGraph->getMatch(matchNodeData.getNodeId());
#endif

			if(node!=mmatch){
				cout<<"Node's matched node is not matched to this node";
				return false;
			}
		}
		int edges=0;
		for(GGraph::edge_iterator ii=graph->edge_begin(node),ee=graph->edge_end(node);ii!=ee;ii++) {
			edges++;
		}
		if(edges!=nodeData.getNumEdges()) {
			cout<<"Number of edges dont match";
			return false;
		}
	}
	bool ret = verifyCoarsening(metisGraph->getFinerGraph());
	cout<<matchedCount<<" "<<unmatchedCount<<endl;
	if(matchedCount+unmatchedCount != metisGraph->getNumNodes())
		return false;
	if(ret == false)
		return false;
	return true;

}

bool verifyRecursiveBisection(MetisGraph* metisGraph,int nparts) {

	GGraph *graph = metisGraph->getGraph();
	int partNodes[nparts];
	memset(partNodes,0,sizeof(partNodes));
	for(GGraph::iterator ii = graph->begin(),ee=graph->end();ii!=ee;ii++) {
		GNode node = *ii;
		MetisNode &nodeData = graph->getData(node);
		if(!(nodeData.getPartition()<nparts))
			return false;
		partNodes[nodeData.getPartition()]++;
		int edges=0;
		for(GGraph::edge_iterator ii=graph->edge_begin(node),ee=graph->edge_end(node);ii!=ee;ii++) {
			edges++;
		}
		if(nodeData.getNumEdges()!=edges) {
			return false;
		}
	}
	int sum=0;
	for(int i=0;i<nparts;i++) {
		if(partNodes[i]<=0)
			return false;
		sum+=partNodes[i];
	}


	if(sum != metisGraph->getNumNodes())
		return false;
	return true;
}

/**
 * KMetis Algorithm
 */
void partition(MetisGraph* metisGraph, int nparts) {
	int coarsenTo = (int) max(metisGraph->getNumNodes() / (40 * intlog2(nparts)), 20 * (nparts));
	int maxVertexWeight = (int) (1.5 * ((metisGraph->getNumNodes()) / (double) coarsenTo));
	Coarsener coarsener(false, coarsenTo, maxVertexWeight);
	Galois::StatTimer T;
	T.start();
	Galois::Timer t;
	t.start();
	MetisGraph* mcg = coarsener.coarsen(metisGraph);
	t.stop();
	cout<<"coarsening time: " << t.get() << " ms"<<endl;
	T.stop();

	if(testMetis::testCoarsening) {
		if(verifyCoarsening(mcg->getFinerGraph())) {
			cout<<"#### Coarsening is correct ####"<<endl;
		} else {
			cout<<"!!!! Coarsening is wrong !!!!"<<endl;
		}
	}


	float* totalPartitionWeights = new float[nparts];
	std::fill_n(totalPartitionWeights, nparts, 1 / (float) nparts);
	maxVertexWeight = (int) (1.5 * ((mcg->getNumNodes()) / COARSEN_FRACTION));
	PMetis pmetis(20, maxVertexWeight);
	Galois::Timer init_part_t;
	init_part_t.start();
	pmetis.mlevelRecursiveBisection(mcg, nparts, totalPartitionWeights, 0, 0);
	init_part_t.stop();
	cout << "initial partition time: "<< init_part_t.get()  << " ms"<<endl;

	if(testMetis::testInitialPartition) {
		cout<<endl<<"#### Verifying initial partition ####"<<endl;
		if(!verifyRecursiveBisection(mcg,nparts)) {
			cout<<endl<<"!!!! Initial partition is wrong !!!!"<<endl;
		}else {
			cout<<endl<<"#### Initial partition is right ####"<<endl;
		}
	}

	//return;
	Galois::Timer refine_t;
	std::fill_n(totalPartitionWeights, nparts, 1 / (float) nparts);
	refine_t.start();
	refineKWay(mcg, metisGraph, totalPartitionWeights, (float) 1.03, nparts);
	refine_t.stop();
	cout << "refine time: " << refine_t.get() << " ms"<<endl;
	delete[] totalPartitionWeights;
	T.stop();
}


void verify(MetisGraph* metisGraph) {
	if (!metisGraph->verify()) {
		cout<<"KMetis failed."<<endl;
	}else{
		cout<<"KMetis okay"<<endl;
	}
}

namespace testMetis {
bool testCoarsening = false;
bool testInitialPartition = false;;
}
namespace variantMetis {
bool mergeMatching = true;
bool noPartInfo = true;
}

struct parallelInitMorphGraph {
	GGraph &graph;
	parallelInitMorphGraph(GGraph &g):graph(g) {

	}
	void operator()(unsigned int tid, unsigned int num) {
		int id = tid;
		for(GGraph::iterator ii = graph.local_begin(),ee=graph.local_end();ii!=ee;ii++) {
			GNode node = *ii;
			MetisNode &nodeData = graph.getData(node);
			nodeData.setNodeId(id);
			nodeData.init();
			nodeData.setWeight(1);
			int count = std::distance(graph.edge_begin(node),graph.edge_end(node));
			nodeData.incNumEdges(count);
			int weight=0;
			for(GGraph::edge_iterator jj = graph.edge_begin(node),kk=graph.edge_end(node);jj!=kk;jj++) {
				graph.getEdgeData(jj)=1;
				weight+=1;
			}
			nodeData.addEdgeWeight(weight);
			id+=num;
		}
	}
};



int main(int argc, char** argv) {
	Galois::StatManager statManager;
	LonestarStart(argc, argv, name, desc, url);

	srand(-1);
	MetisGraph metisGraph;
	GGraph graph;
	metisGraph.setGraph(&graph);
	//bool directed = true;

	Galois::reportPageAlloc("MeminfoPre1");
	//You will need to allocate 2-4 the total memory of graph. So pre alloc pages.
	Galois::preAlloc(10000);
	Galois::reportPageAlloc("MeminfoPre1");
#ifndef LC_MORPH
	if(mtxInput){
		readMetisGraph(&metisGraph, filename.c_str());
	}else{
		readGraph(&metisGraph, filename.c_str(), weighted, false);
	}
#else
	graph.structureFromFile(filename);
#endif


	Galois::on_each(parallelInitMorphGraph(graph));

	metisGraph.setNumNodes(graph.size());
	metisGraph.setNumEdges(graph.sizeEdges());
	cout<<"Nodes "<<graph.size()<<"| Edges "<<graph.sizeEdges()<<endl;

	Galois::Timer t;
	t.start();
	partition(&metisGraph, numPartitions);
	t.stop();
	cout<<"Total Time "<<t.get()<<" ms "<<endl;
	Galois::reportPageAlloc("MeminfoPre3");
	verify(&metisGraph);
}

int getRandom(int num){
	//	int randNum = rand()%num;
	//	return (rand()>>3)%(num);
	//	return randNum;
	return ((int)(drand48()*((double)(num))));
}

// int gNodeToInt(GNode node){
// 	return graph->getData(node).getNodeId();
// }

int intlog2(int a){
	int i;
	for (i=1; a > 1; i++, a = a>>1);
	return i-1;
}

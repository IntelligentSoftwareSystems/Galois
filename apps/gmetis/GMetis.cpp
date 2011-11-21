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

#include "Galois/Graphs/FileGraph.h"
#include "Galois/Statistic.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* name = "GMetis";
static const char* desc = "Partion a graph into K parts and minimize the graph cut\n";
static const char* url = "gMetis";

static cll::opt<bool> mtxInput("mtxinput", cll::desc("Use text mtx files instead binary based ones"), cll::init(false));
static cll::opt<bool> weighted("weighted", cll::desc("weighted"), cll::init(false));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> numPartitions(cll::Positional, cll::desc("<Number of partitions>"), cll::Required);

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

	float* totalPartitionWeights = new float[nparts];
	arrayFill(totalPartitionWeights, nparts, 1 / (float) nparts);
	maxVertexWeight = (int) (1.5 * ((mcg->getNumNodes()) / COARSEN_FRACTION));
	PMetis pmetis(20, maxVertexWeight);
	Galois::Timer init_part_t;
	init_part_t.start();
	pmetis.mlevelRecursiveBisection(mcg, nparts, totalPartitionWeights, 0, 0);
	init_part_t.stop();
	cout << "initial partition time: "<< init_part_t.get()  << " ms"<<endl;
	Galois::Timer refine_t;

	arrayFill(totalPartitionWeights, nparts, 1 / (float) nparts);
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

typedef Galois::Graph::LC_FileGraph<int, unsigned int> InputGraph;
typedef Galois::Graph::LC_FileGraph<int, unsigned int>::GraphNode InputGNode;

void readMetisGraph(MetisGraph* metisGraph, const char* filename){
	std::ifstream file(filename);
	string line;
	std::getline(file, line);
	while(line.find('%')!=string::npos){
		std::getline(file, line);
	}

	int numNodes, numEdges;
	sscanf(line.c_str(), "%d %d", &numNodes, &numEdges);
	cout<<numNodes<<" "<<numEdges<<endl;
	GGraph* graph = metisGraph->getGraph();
	vector<GNode> nodes(numNodes);
	for (int i = 0; i < numNodes; i++) {
		GNode n = graph->createNode(MetisNode(i, 1));
		nodes[i] = n;
		graph->addNode(n);
	}
	int countEdges = 0;
	for (int i = 0; i < numNodes; i++) {
		std::getline(file, line);
		char const * items = line.c_str();
		char* remaining;
		GNode n1 = nodes[i];

		while (true) {
			int index = strtol(items, &remaining,10) - 1;
			if(index < 0) break;
			items = remaining;
			GNode n2 = nodes[index];
			if(n1==n2){
				continue;
			}
			graph->addEdge(n1, n2, 1);
			n1.getData().addEdgeWeight(1);
			n1.getData().incNumEdges();
			countEdges++;
		}
	}

	assert(countEdges == numEdges*2);
	metisGraph->setNumEdges(numEdges);
	metisGraph->setNumNodes(numNodes);
	cout<<"finshied reading graph " << metisGraph->getNumNodes() << " " << metisGraph->getNumEdges()<<endl;
}


void readGraph(MetisGraph* metisGraph, const char* filename, bool weighted = false, bool directed = true){
	InputGraph inputGraph;
	inputGraph.structureFromFile(filename);
	inputGraph.emptyNodeData();
	cout<<"start to transfer data to GGraph"<<endl;
	int id = 0;
	for (InputGraph::active_iterator ii = inputGraph.active_begin(), ee = inputGraph.active_end(); ii != ee; ++ii) {
		InputGNode node = *ii;
		inputGraph.getData(node)=id++;
	}

	GGraph* graph = metisGraph->getGraph();
	vector<GNode> gnodes(inputGraph.size());
	id = 0;
	for(uint64_t i=0;i<inputGraph.size();i++){
		GNode node = graph->createNode(MetisNode(id, 1));
		graph->addNode(node);
		gnodes[id++] = node;
	}

	int numEdges = 0;
	for (InputGraph::active_iterator ii = inputGraph.active_begin(), ee = inputGraph.active_end(); ii != ee; ++ii) {
		InputGNode inNode = *ii;

		int nodeId = inputGraph.getData(inNode);
		GNode node = gnodes[nodeId];

		MetisNode& nodeData = node.getData();

		for (InputGraph::neighbor_iterator jj = inputGraph.neighbor_begin(inNode), eejj = inputGraph.neighbor_end(inNode); jj != eejj; ++jj) {
			InputGNode inNeighbor = *jj;
			if(inNode == inNeighbor) continue;
			int neighId = inputGraph.getData(inNeighbor);
			int weight = 1;
			if(weighted){
				weight = inputGraph.getEdgeData(inNode, inNeighbor);
			}
			if(!directed){
				graph->addEdge(node, gnodes[neighId], weight);//
				nodeData.incNumEdges();
				nodeData.addEdgeWeight(weight);//inputGraph.getEdgeData(inNode, inNeighbor));
				numEdges++;
			}else{
				graph->addEdge(node, gnodes[neighId], weight);//
				graph->addEdge(gnodes[neighId], node, weight);//
			}
		}

	}

	if(directed){
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = node.getData();
			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node), eejj = graph->neighbor_end(node); jj != eejj; ++jj) {
				GNode neighbor = *jj;
				nodeData.incNumEdges();
				nodeData.addEdgeWeight(graph->getEdgeData(node, neighbor));
				assert(graph->getEdgeData(node, neighbor) == graph->getEdgeData(neighbor, node));
				numEdges++;
			}
		}
	}
	cout<<"numNodes: "<<inputGraph.size()<<"|numEdges: "<<numEdges/2<<endl;
	metisGraph->setNumEdges(numEdges/2);
	metisGraph->setNumNodes(gnodes.size());
	cout<<"end of transfer data to GGraph"<<endl;
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

	srand(-1);
	MetisGraph metisGraph;
	GGraph graph;
	metisGraph.setGraph(&graph);
	bool directed = true;
	if(mtxInput){
	  readMetisGraph(&metisGraph, filename.c_str());
	}else{
	  readGraph(&metisGraph, filename.c_str(), weighted, directed);
	}
	partition(&metisGraph, numPartitions);
	verify(&metisGraph);
}

int getRandom(int num){
//	int randNum = rand()%num;
//	return (rand()>>3)%(num);
	//	return randNum;
	return ((int)(drand48()*((double)(num))));
}

int gNodeToInt(GNode node){
	return node.getData().getNodeId();
}

void mergeP::operator()(PerCPUValue& a, PerCPUValue& b){
	a.mincutInc+=b.mincutInc;
	a.changedBndNodes.insert(b.changedBndNodes.begin(), b.changedBndNodes.end());
}
int intlog2(int a){
	int i;
	for (i=1; a > 1; i++, a = a>>1);
	return i-1;
}

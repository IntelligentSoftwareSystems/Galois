/*
 * GMetis.cpp
 *
 *  Created on: Jun 13, 2011
 *      Author: xinsui
 */

#include "GMetisConfig.h"
#include "MetisGraph.h"
#include <vector>
#include <iostream>
#include <string.h>
#include "Galois/Launcher.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include "PMetis.h"
#include "Galois/Graphs/FileGraph.h"

static const char* name = "GMetis";
static const char* description = "Partion a graph into K parts and minimize the graph cut\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/gmetis.html";
static const char* help = "<input file> numPartitions";

/**
 * KMetis Algorithm
 */
void partition(MetisGraph* metisGraph, int nparts) {
	GGraph* graph= metisGraph->getGraph();

	int coarsenTo = (int) max(graph->size() / (40 * log(nparts)), (double)20 * (nparts));
	int maxVertexWeight = (int) (1.5 * ((graph->size()) / (double) coarsenTo));
	Coarsener coarsener(false, coarsenTo, maxVertexWeight);
	cout<<"start coarsening"<<endl;
	Galois::Launcher::startTiming();
	MetisGraph* mcg = coarsener.coarsen(metisGraph);
//	Galois::Launcher::stopTiming();
//	cout<<"coarsening time: " << Galois::Launcher::elapsedTime() << " ms";

	float* totalPartitionWeights = new float[nparts];
	arrayFill(totalPartitionWeights, nparts, 1 / (float) nparts);
//	Galois::Launcher::startTiming();
	maxVertexWeight = (int) (1.5 * ((mcg->getGraph()->size()) / Coarsener::COARSEN_FRACTION));
	PMetis pmetis(20, maxVertexWeight);
	cout<<"initial partion"<<endl;
	pmetis.mlevelRecursiveBisection(mcg, nparts, totalPartitionWeights, 0, 0);
//	Galois::Launcher::stopTiming();
	cout << "initial partition time: "  << " ms";
	//	mcg->setNParts(nparts);
//	Galois::Launcher::startTiming();
	arrayFill(totalPartitionWeights, nparts, 1 / (float) nparts);
	refineKWay(mcg, metisGraph, totalPartitionWeights, (float) 1.03, nparts);
//	cout << "refine time: " << Galois::Launcher::elapsedTime() << " ms"<<endl;
	delete[] totalPartitionWeights;
	Galois::Launcher::stopTiming();
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

//void readGraph(MetisGraph* metisGraph, char* filename){
//	std::ifstream file(filename);
//	string line;
//	std::getline(file, line);
//	while(line.find('%')!=string::npos){
//		std::getline(file, line);
//	}
//	int numNodes, numEdges;
//	sscanf(line.c_str(), "%d %d", &numNodes, &numEdges);
//	GGraph* graph = metisGraph->getGraph();
//	vector<GNode> nodes(numNodes);
//	for (int i = 0; i < nodeNum; i++) {
//		GNode n = graph.createNode(new MetisNode(i, 1));
//		nodes[i] = n;
//		graph->addNode(n);
//	}
//	for (int i = 0; i < nodeNum; i++) {
//		std::getline(file, line);
//
//		GNode n1 = nodes.get(i);
//
//		for (int j = 0; j < segs.length; j++) {
//			GNode n2 = nodes.get(Integer.valueOf(segs[j]) - 1);
//			graph->addEdge(n1, n2, 1);
//			n1.getData().addEdgeWeight(1);
//			n1.getData().incNumEdges();
//			numEdges++;
//		}
//	}
//	MetisGraph metisGraph = new MetisGraph();
//	metisGraph.setNumEdges(numEdges / 2);
//	metisGraph.setGraph(graph);
//	System.out.println("finshied reading graph " + graph.size() + " " + metisGraph.getNumEdges());
//}


void readGraph(MetisGraph* metisGraph, const char* filename){
	InputGraph inputGraph;
	inputGraph.structureFromFile(filename);
	inputGraph.emptyNodeData();
	cout<<"start to transfer data to GGraph"<<endl;
	int id = 0;
	//	vector<InputGNode> inNodes(inputGraph.size());
	for (InputGraph::active_iterator ii = inputGraph.active_begin(), ee = inputGraph.active_end(); ii != ee; ++ii) {
		InputGNode node = *ii;
		inputGraph.getData(node)=id++;
	}

	GGraph* graph = metisGraph->getGraph();
	vector<GNode> gnodes(inputGraph.size());
	id = 0;
	for(int i=0;i<inputGraph.size();i++){
		GNode node = graph->createNode(MetisNode(id, 1));
		graph->addNode(node);
		gnodes[id++] = node;

	}
	cout<<"finish create node "<<id<<endl;
	int numEdges = 0;
	for (InputGraph::active_iterator ii = inputGraph.active_begin(), ee = inputGraph.active_end(); ii != ee; ++ii) {
		InputGNode inNode = *ii;

		int nodeId = inputGraph.getData(inNode);
		GNode node = gnodes[nodeId];

		MetisNode& nodeData = node.getData();

		for (InputGraph::neighbor_iterator jj = inputGraph.neighbor_begin(inNode), eejj = inputGraph.neighbor_end(inNode); jj != eejj; ++jj) {
			InputGNode inNeighbor = *jj;
			int neighId = inputGraph.getData(inNeighbor);
			graph->addEdge(node, gnodes[neighId], 1);//inputGraph.getEdgeData(inNode, inNeighbor));
			nodeData.incNumEdges();
			nodeData.addEdgeWeight(1);//inputGraph.getEdgeData(inNode, inNeighbor));
			numEdges++;
		}
	}

	metisGraph->setNumEdges(numEdges/2);
	metisGraph->setNumNodes(gnodes.size());
	cout<<"end of transfer data to GGraph"<<endl;
}

int main(int argc, const char** argv) {
	srand(3);
	std::vector<const char*> args = parse_command_line(argc, argv, help);
	if (args.size() != 2) {
		std::cout << "incorrect number of arguments, use -help for usage information\n";
		return 1;
	}
	printBanner(std::cout, name, description, url);
	MetisGraph metisGraph;
	GGraph graph;
	metisGraph.setGraph(&graph);
	readGraph(&metisGraph, argv[1]);
	//	Galois::setMaxThreads(4);
	partition(&metisGraph, atoi(argv[2]));
	verify(&metisGraph);
}

int getRandom(int num){
	int randNum = rand()%num;
	return randNum;
}

int gNodeToInt(GNode node){
	return node.getData().getNodeId();
}

void merge(PerCPUValue& a, PerCPUValue& b){
	a.mincutInc+=b.mincutInc;
	a.changedBndNodes.insert(b.changedBndNodes.begin(), b.changedBndNodes.end());
}

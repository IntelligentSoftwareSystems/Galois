/*
 * DelaunayTriangulation.cpp
 *
 *  Created on: Jan 25, 2011
 *      Author: xinsui
 */
#include <vector>
#include <iostream>
#include <string.h>
#include "Element.h"
#include "DataManager.h"
#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Delaunay Triangulation";
static const char* description = "Produces a Delaunay triangulation from a given a set of points\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/delaunaytriangulation.html";
static const char* help = "<input file>";


typedef Galois::Graph::FirstGraph<DTElement,int,true>            Graph;
typedef Galois::Graph::FirstGraph<DTElement,int,true>::GraphNode GNode;
typedef std::vector<GNode, Galois::PerIterMem::ItAllocTy::rebind<GNode>::other> GNodeVector;
typedef std::vector<GNode, Galois::PerIterMem::ItAllocTy::rebind<GNode>::other>::iterator GNodeVectorIter;
//typedef std::vector<GNode> GNodeVector;
//typedef std::vector<GNode>::iterator GNodeVectorIter;

	
#include "Cavity.h"

Graph* mesh;
int threads = 1;

struct process {
	template<typename Context>
	void operator()(GNode item, Context& lwl) {
	  assert(!item.isNull());
		DTElement& data = item.getData(Galois::Graph::ALL); //lock
		if (data.isProcessed())
			return;
	
		DTCavity cav(mesh, item, data.getTuples().back(),  &lwl);
		cav.build();
		
		GNodeVector newNodes(lwl.PerIterationAllocator);
		cav.update(&newNodes);
		for(GNodeVectorIter iter=newNodes.begin();iter!=newNodes.end();iter++){
		   GNode node = *iter;
		   
		   if (!node.getData(Galois::Graph::NONE).getTuples().empty()) {
				lwl.push(node);
	           }
		}
	}
};

template<typename WLTY>
void triangulate(WLTY& wl) {
	//GaloisRuntime::WorkList::LIFO<GNode> wl2;
    //GaloisRuntime::WorkList::FIFO<GNode> wl2;
	GaloisRuntime::WorkList::ChunkedFIFO<GNode, 64, false> wl2;
	wl2.fill_initial(wl.begin(), wl.end());
	Galois::for_each(wl2, process());
	
	//Galois::for_each(wl.begin(), wl.end(), process());
}

using namespace std;

DTTuple DataManager::t1;
DTTuple DataManager::t2;
DTTuple DataManager::t3;
double DataManager::MIN_X;
double DataManager::MIN_Y; 
double DataManager::MAX_X; 
double DataManager::MAX_Y;

int main(int argc, const char** argv) {

	std::vector<const char*> args = parse_command_line(argc, argv, help);

	if (args.size() != 1) {
		std::cout << "incorrect number of arguments, use -help for usage information\n";
		return 1;
	}
	printBanner(std::cout, name, description, url);

	mesh = new Graph();
	vector<DTTuple> tuples;
	DataManager::readTuplesFromFile(args[0], &tuples);

	DTElement large_triangle(DataManager::t1, DataManager::t2, DataManager::t3);
	GNode large_node = mesh->createNode(large_triangle);
	
	mesh->addNode(large_node, Galois::Graph::NONE, 0);

        DTElement border_ele1(DataManager::t1, DataManager::t2);
        DTElement border_ele2(DataManager::t2, DataManager::t3);
        DTElement border_ele3(DataManager::t3, DataManager::t1);
    
	GNode border_node1 = mesh->createNode(border_ele1);
	GNode border_node2 = mesh->createNode(border_ele2);
	GNode border_node3 = mesh->createNode(border_ele3);
	mesh->addNode(border_node1, Galois::Graph::NONE, 0);
	mesh->addNode(border_node2, Galois::Graph::NONE, 0);
	mesh->addNode(border_node3, Galois::Graph::NONE, 0);

	mesh->addEdge(large_node, border_node1, 0);
	mesh->addEdge(large_node, border_node2, 1);
	mesh->addEdge(large_node, border_node3, 2);

	mesh->addEdge(border_node1, large_node, 0);
	mesh->addEdge(border_node2, large_node, 0);
	mesh->addEdge(border_node3, large_node, 0);
	// END --- Create the main initial triangle
	
	large_node.getData().getTuples().swap(tuples);
	
	std::vector<GNode> wl;
	wl.push_back(large_node);
	cout << "configuration: " << large_node.getData().getTuples().size() << " total points\n"
			<< "number of threads: " << threads << "\n"
			<< "\n";

	Galois::setMaxThreads(numThreads);
	Galois::Launcher::startTiming();
	triangulate(wl);
	Galois::Launcher::stopTiming();
	cout << " mesh size:" << mesh->size() <<"\n";
	cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";

	return 0;
}

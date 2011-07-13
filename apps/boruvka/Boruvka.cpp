/** Boruvka application -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Rashid Kaleem <rashid@cs.utexas.edu>
 */


#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include "Galois/Graphs/Serialize.h"
#include "Galois/Graphs/FileGraph.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <set>
 
#define BORUVKA_DEBUG 0 


static const char* name = "Boruvka MST";
static const char* description = "Computes the Minimal Spanning Tree using Boruvka\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/boruvka.html";
static const char* help = "<input file> ";
static unsigned int nodeID = 0;
struct Node {
	unsigned int id;
	Node(int _id = -1 ): id(_id){}
	std::string toString() {
		std::ostringstream s;
		s << '[' << id << "]" ;
		return s.str();
	}
};

typedef Galois::Graph::FirstGraph<Node, int, false> Graph;
typedef Graph::GraphNode GNode;
//The graph.
Graph graph;
std::vector<GNode> nodes;

void printGraph() {
	int numEdges = 0;
	for (Graph::active_iterator src = graph.active_begin(), esrc = graph.active_end();src != esrc; ++src) {
		Node& sdata = graph.getData(*src, Galois::NONE);
		if(graph.containsNode(*src))
			for (Graph::neighbor_iterator dst = graph.neighbor_begin(*src, Galois::NONE), edst = graph.neighbor_end(*src, Galois::NONE);dst != edst; ++dst) {
				int w = graph.getEdgeData(*src, *dst, Galois::NONE);
				int x = graph.getEdgeData(*dst, *src, Galois::NONE);
				Node& ddata = graph.getData(*dst, Galois::NONE);
				std::cout<<"1) "<<sdata.toString()<<" => "<< ddata.toString() << " [ "<<w << " ] "<<x << std::endl;
				numEdges++;
			}
	}
	std::cout<<"Num edges "<<numEdges << std::endl;
}
GaloisRuntime::PerCPU<unsigned int> MSTWeight;
struct process {
	template<typename ContextTy>
		void __attribute__((noinline)) operator()(GNode& src, ContextTy& lwl) {
			if(graph.containsNode(src)==false)
				return;
			graph.getData(src);
			//Graph::neighbor_iterator minNeighbor;
			GNode * minNeighbor;
#if BORUVKA_DEBUG
			std::cout<<"Processing "<<graph.getData(src).toString()<<std::endl;
#endif
			int minEdgeWeight=INT_MAX;
			int numNeighbors = 0;
			//Acquire locks on neighborhood.
			for (Graph::neighbor_iterator dst = graph.neighbor_begin(src, Galois::ALL), edst = graph.neighbor_end(src, Galois::ALL);dst != edst; ++dst) {
				graph.getData(*dst);
				graph.getEdgeData(src,*dst, Galois::ALL);
				graph.getEdgeData(*dst,src, Galois::ALL);
			}
			for (Graph::neighbor_iterator dst = graph.neighbor_begin(src, Galois::NONE), edst = graph.neighbor_end(src, Galois::NONE);dst != edst; ++dst) {
				numNeighbors++;
				int minNbrId = graph.getData(*dst).id;
				
				int w = graph.getEdgeData(src, *dst, Galois::NONE);
				if(w<minEdgeWeight){
					minNeighbor = &nodes[minNbrId];
					minEdgeWeight = w;
				}
			}
			//If there are no outgoing neighbors.
			if(numNeighbors==0  || minEdgeWeight == INT_MAX){
				graph.removeNode(src, Galois::NONE);
				return;
			}
			graph.getData(*minNeighbor);
#if BORUVKA_DEBUG
			std::cout << " Min edge from "<<graph.getData(src).toString() << " to "<<graph.getData(*minNeighbor).toString()<<" " <<minEdgeWeight << " "<<std::endl ;
#endif
			//Acquire locks on neighborhood of min neighbor.
			for (Graph::neighbor_iterator dst = graph.neighbor_begin(*minNeighbor, Galois::ALL), edst = graph.neighbor_end(*minNeighbor, Galois::ALL);dst != edst; ++dst) {
				graph.getData(*dst);
				graph.getEdgeData(*minNeighbor,*dst, Galois::ALL);
				graph.getEdgeData(*dst,*minNeighbor, Galois::ALL);
			}

			//update MST weight.
			MSTWeight.get()+=minEdgeWeight;
			//std::set<GNode> toRemove;
			std::set<int> toRemove;
			typedef std::pair<int,int> EdgeData;
			std::set<EdgeData> toAdd;
			//std::set<int> toAdd;

			for(Graph::neighbor_iterator mdst = graph.neighbor_begin(*minNeighbor, Galois::NONE), medst = graph.neighbor_end(*minNeighbor, Galois::NONE); mdst!=medst;++mdst){
				graph.getData(*mdst);
				int edgeWeight = graph.getEdgeData(*minNeighbor, *mdst, Galois::NONE);
				if(*mdst!=src){
					GNode dstNode = (*mdst);
					if(src.hasNeighbor(dstNode) || dstNode.hasNeighbor(src)){
						if(graph.getEdgeData(src,*mdst, Galois::NONE)<edgeWeight)
							edgeWeight = graph.getEdgeData(src,*mdst, Galois::NONE);
						graph.getEdgeData(src,*mdst,Galois::NONE)=edgeWeight;
					}
					else
					{
						EdgeData e (graph.getData(*mdst).id, edgeWeight);
						toAdd.insert(e);
					}
				}
				toRemove.insert(graph.getData(*mdst).id);
			}

			for(std::set<int>::iterator it = toRemove.begin(), endIt = toRemove.end();it!=endIt; it++){
				graph.removeEdge(*minNeighbor, nodes[*it], Galois::NONE);
			}
			for(std::set<EdgeData>::iterator it = toAdd.begin(), endIt = toAdd.end();it!=endIt; it++){
				graph.addEdge(src, nodes[it->first], it->second, Galois::NONE);
			}
			graph.removeNode(*minNeighbor, Galois::NONE);
			lwl.push(src);
		}
};
//End body of for-each.
void runBodyParallel() {
	using namespace GaloisRuntime::WorkList;
	std::vector<GNode> all(graph.active_begin(), graph.active_end());
#if BORUVKA_DEBUG
	std::cout<<"Begining to process with worklist size :: "<<all.size()<<std::endl;
	std::cout<<"Graph size "<<graph.size()<<std::endl;
#endif
	for(int i=0;i<MSTWeight.size();i++)
		MSTWeight.get(i)=0;
	Galois::for_each<ChunkedFIFO<32> >(all.begin(), all.end(), process());
	unsigned int res = 0;
	for(int i=0;i<MSTWeight.size();i++){
#if BORUVKA_DEBUG
		std::cout<<"MST +=" << MSTWeight.get(i)<<std::endl;
#endif
		res+=MSTWeight.get(i);
	}
	std::cout<<"MST Weight is "<< res << std::endl;
}

static void makeGraph(const char* input) {
	//Create local computation graph.
  typedef Galois::Graph::LC_FileGraph<Node, int> InGraph;
	typedef InGraph::GraphNode InGNode;
	InGraph in_graph;
	//Read graph from file.
	in_graph.structureFromFile(input);
	std::cout << "Read " << in_graph.size() << " nodes\n";
	//A node and a int is an element.
	typedef std::pair<InGNode, int> Element;
	//A vector of element is 'Elements'
	typedef std::vector<Element> Elements;
	//A vector of 'Elements' is a 'Map'
	typedef std::vector<Elements> Map;
	//'in_edges' is a vector of vector of pairs of nodes and int. 
	Map edges(in_graph.size());
	// 
	int numEdges = 0;
	for (InGraph::active_iterator src = in_graph.active_begin(), esrc = in_graph.active_end();src != esrc; ++src) {
		for (InGraph::neighbor_iterator dst = in_graph.neighbor_begin(*src, Galois::NONE), edst = in_graph.neighbor_end(*src, Galois::NONE);dst != edst; ++dst) {
			int w = in_graph.getEdgeData(*src, *dst, Galois::NONE);
			Element e(*src, w);
			edges[*dst].push_back(e);
			numEdges++;
		}
	}
#if BORUVKA_DEBUG
	std::cout<<"Number of edges "<<numEdges<<std::endl;
#endif
	unsigned int id = 0;

	//std::vector<GNode> nodes(in_graph.size());
	nodes.resize(in_graph.size());
	for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
		Node n(nodeID);
		GNode node = graph.createNode(n);
		graph.addNode(node);
		nodes[nodeID] = node;
		nodeID++;
	}

	id = 0;
	numEdges=0;
	int numDups = 0;
	for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
		GNode src = nodes[id];
		for (Elements::iterator j = i->begin(), ej = i->end(); j != ej; ++j) {
			if(src.hasNeighbor(nodes[j->first])){
				numDups++;
				int w = (graph.getEdgeData(src,nodes[j->first],Galois::NONE));	
				if(j->second<w)
					graph.getEdgeData(src,nodes[j->first],Galois::NONE)=j->second;
			}
			else{
				graph.addEdge(src, nodes[j->first], j->second);
			}
			numEdges++;
		}
		id++;
	}
#if BORUVKA_DEBUG
	std::cout<<"Final num edges "<<numEdges<< " Dups "<<numDups<<std::endl;
#endif
}


int main(int argc, const char **argv) {
	std::vector<const char*> args = parse_command_line(argc, argv, help);
	if (args.size() < 1) {
		std::cout << "not enough arguments, use -help for usage information\n";
		return 1;
	}
	printBanner(std::cout, name, description, url);
	const char* inputfile = args[0];
	makeGraph(inputfile);
#if BORUVKA_DEBUG
	printGraph();
#endif
	Galois::StatTimer T;
	T.start();
	runBodyParallel();
	T.stop();
	return 0;
}

// vim:sw=2:sts=2:ts=8

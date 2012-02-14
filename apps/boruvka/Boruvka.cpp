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
#include "Galois/Graphs/LCGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <map>
#include <set>

#define BORUVKA_DEBUG 0

using namespace std;
namespace cll = llvm::cl;

static const char* name = "Boruvka MST";
static const char* desc = "Computes the Minimal Spanning Tree using Boruvka\n";
static const char* url = "boruvkas_algorithm";

static cll::opt<std::string> inputfile(cll::Positional, cll::desc(
		"<input file>"), cll::Required);

static unsigned int nodeID = 0;
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
struct Node {
	unsigned int id;
	Node(int _id = -1) :
		id(_id) {
	}
	std::string toString() {
		std::ostringstream s;
		s << '[' << id << "]";
		return s.str();
	}
};
std::ostream& operator<<(std::ostream& s, Node& n) {
	s << "Node[" << n.id << "]";
	return s;
}
bool operator<(const Node&n1, const Node &n2) {
	return n1.id < n2.id;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
typedef Galois::Graph::FirstGraph<Node, int, false> Graph;
typedef Graph::GraphNode GNode;
//The graph.
Graph graph;
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void printGraph() {
	int numEdges = 0;
	for (Graph::active_iterator src = graph.active_begin(), esrc =
			graph.active_end(); src != esrc; ++src) {
		Node& sdata = graph.getData(*src, Galois::NONE);
		if (graph.containsNode(*src))
			for (Graph::neighbor_iterator dst = graph.neighbor_begin(*src,
					Galois::NONE), edst =
					graph.neighbor_end(*src, Galois::NONE); dst != edst; ++dst) {
				int w = graph.getEdgeData(*src, *dst, Galois::NONE);
				int x = graph.getEdgeData(*dst, *src, Galois::NONE);
				Node& ddata = graph.getData(*dst, Galois::NONE);
				std::cout << "1) " << sdata.toString() << " => "
						<< ddata.toString() << " [ " << w << " ] " << x
						<< std::endl;
				numEdges++;
			}
	}
	std::cout << "Num edges " << numEdges << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
GaloisRuntime::PerCPU<long> MSTWeight;
struct process {
	template<typename ContextTy>
	void __attribute__((noinline)) operator()(GNode& src, ContextTy& lwl) {
		if (graph.containsNode(src) == false)
			return;
		graph.getData(src);
		GNode minNeighbor;
#if BORUVKA_DEBUG
		std::cout<<"Processing "<<graph.getData(src).toString()<<std::endl;
#endif
		int minEdgeWeight = std::numeric_limits<int>::max();
		int numNeighbors = 0;
		//Acquire locks on neighborhood.
		for (Graph::neighbor_iterator dst = graph.neighbor_begin(src,
				Galois::ALL), edst = graph.neighbor_end(src, Galois::ALL); dst
				!= edst; ++dst) {
			graph.getData(*dst);
			graph.getEdgeData(src, *dst, Galois::ALL);
			graph.getEdgeData(*dst, src, Galois::ALL);
		}
		//Find minimum neighbor
		for (Graph::neighbor_iterator dst = graph.neighbor_begin(src,Galois::NONE), edst = graph.neighbor_end(src, Galois::NONE); dst!= edst; ++dst) {
			numNeighbors++;
			int w = graph.getEdgeData(src, *dst, Galois::NONE);
			assert (graph.getEdgeData(src, *dst, Galois::NONE) == graph.getEdgeData(*dst, src, Galois::NONE));
			if (w < minEdgeWeight) {
				minNeighbor = *dst;
				minEdgeWeight = w;
			}
		}
		//If there are no outgoing neighbors.
		if (numNeighbors == 0 || minEdgeWeight== std::numeric_limits<int>::max()) {
			graph.removeNode(src, Galois::NONE);
			//XXX remove the return stmt to have a single point of exit.
			return;
		}
#if BORUVKA_DEBUG
		std::cout << " Min edge from "<<graph.getData(src).toString() << " to "<<graph.getDataminNeighbor.toString()<<" " <<minEdgeWeight << " "<<std::endl;
#endif
		//Acquire locks on neighborhood of min neighbor.
		for (Graph::neighbor_iterator dst = graph.neighbor_begin(minNeighbor,Galois::ALL), edst = graph.neighbor_end(minNeighbor,Galois::ALL); dst != edst; ++dst) {
			graph.getData(*dst);
			graph.getEdgeData(minNeighbor, *dst, Galois::ALL);
			graph.getEdgeData(*dst, minNeighbor, Galois::ALL);
		}
		//update MST weight.
		MSTWeight.get() += minEdgeWeight;
		typedef std::set<GNode, std::less<GNode>,Galois::PerIterAllocTy::rebind<GNode>::other> intsetTy;
		intsetTy toRemove(std::less<GNode>(), Galois::PerIterAllocTy::rebind<GNode>::other(lwl.getPerIterAlloc()));
		typedef std::pair<GNode, int> EdgeData;
		typedef std::set<EdgeData, std::less<EdgeData>,Galois::PerIterAllocTy::rebind<EdgeData>::other> edsetTy;
		edsetTy toAdd(std::less<EdgeData>(), Galois::PerIterAllocTy::rebind<EdgeData>::other(lwl.getPerIterAlloc()));

		for (Graph::neighbor_iterator mdst = graph.neighbor_begin(minNeighbor,Galois::NONE), medst = graph.neighbor_end(minNeighbor,Galois::NONE); mdst != medst; ++mdst) {
			graph.getData(*mdst);
			int edgeWeight = graph.getEdgeData(minNeighbor, *mdst,Galois::NONE);
			if (*mdst != src) {
				GNode dstNode = (*mdst);
				if (src.hasNeighbor(dstNode) || dstNode.hasNeighbor(src)) {
					if (graph.getEdgeData(src, dstNode, Galois::NONE)< edgeWeight)
						edgeWeight= graph.getEdgeData(src, dstNode, Galois::NONE);
					graph.getEdgeData(src, dstNode, Galois::NONE) = edgeWeight;
				} else {
					EdgeData e(dstNode, edgeWeight);
					toAdd.insert(e);
				}
			}
			toRemove.insert(*mdst);
		}
		graph.removeNode(minNeighbor, Galois::NONE);
		for (edsetTy::iterator it = toAdd.begin(), endIt = toAdd.end(); it!= endIt; it++) {
			graph.addEdge(src, it->first, it->second, Galois::NONE);
		}
		lwl.push(src);
	}
};
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
struct Indexer: public std::unary_function<const GNode&, unsigned int> {
	unsigned operator()(const GNode& n) {
		return std::distance(graph.neighbor_begin(n, Galois::NONE),
				graph.neighbor_end(n, Galois::NONE));
	}
	static unsigned foo(const GNode& n) {
		return std::distance(graph.neighbor_begin(n, Galois::NONE),
				graph.neighbor_end(n, Galois::NONE));
	}
};
struct seq_less: public std::binary_function<const GNode&, const GNode&, bool> {
	bool operator()(const GNode& lhs, const GNode& rhs) const {
		if (Indexer::foo(lhs) < Indexer::foo(rhs))
			return true;
		if (Indexer::foo(lhs) > Indexer::foo(rhs))
			return false;
		return lhs < rhs;
	}
};
struct seq_gt: public std::binary_function<const GNode&, const GNode&, bool> {
	bool operator()(const GNode& lhs, const GNode& rhs) const {
		if (Indexer::foo(lhs) > Indexer::foo(rhs))
			return true;
		if (Indexer::foo(lhs) < Indexer::foo(rhs))
			return false;
		return lhs > rhs;
	}
};
//End body of for-each.
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void runBodyParallel() {
	using namespace GaloisRuntime::WorkList;
	typedef dChunkedFIFO<64> dChunk;
	typedef ChunkedFIFO<64> Chunk;
	typedef OrderedByIntegerMetric<Indexer, dChunk> OBIM;

#if BORUVKA_DEBUG
	std::cout<<"Graph size "<<graph.size()<<std::endl;
#endif

	for (unsigned i = 0; i < MSTWeight.size(); i++)
		MSTWeight.get(i) = 0;

	Galois::StatTimer T;

	T.start();
	//Exp::StartWorklistExperiment<OBIM, dChunk, Chunk, Indexer, seq_less, seq_gt>()(std::cout, graph.active_begin(), graph.active_end(), process());
	Galois::for_each<dChunk>(graph.active_begin(), graph.active_end(), process());
	T.stop();

	//TODO: use a reduction variable here
	long res = 0;
	for (unsigned i = 0; i < MSTWeight.size(); i++) {
#if BORUVKA_DEBUG
		std::cout<<"MST +=" << MSTWeight.get(i)<<std::endl;
#endif
		res += MSTWeight.get(i);
	}
	std::cout << "MST Weight is " << res << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
static void makeGraph(const char* input) {
	std::vector<GNode> nodes;
	//Create local computation graph.
	typedef Galois::Graph::LC_CRS_Graph<Node, int> InGraph;
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
	for (InGraph::active_iterator src = in_graph.active_begin(), esrc =
			in_graph.active_end(); src != esrc; ++src) {
		for (InGraph::edge_iterator dst = in_graph.edge_begin(*src,
				Galois::NONE), edst = in_graph.edge_end(*src, Galois::NONE); dst
				!= edst; ++dst) {
			if (*src == *dst) {
#if BORUVKA_DEBUG
				std::cout<<"ERR:: Self loop at "<<*src<<std::endl;
#endif
				continue;
			}
			int w = in_graph.getEdgeData(dst);
			Element e(*src, w);
			edges[in_graph.getEdgeDst(dst)].push_back(e);
			numEdges++;
		}
	}
#if BORUVKA_DEBUG
	std::cout<<"Number of edges "<<numEdges<<std::endl;
#endif
	unsigned int id = 0;
	nodes.resize(in_graph.size());
	for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
		Node n(nodeID);
		GNode node = graph.createNode(n);
		graph.addNode(node);
		nodes[nodeID] = node;
		nodeID++;
	}

	id = 0;
	numEdges = 0;
	int numDups = 0;
	for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
		GNode src = nodes[id];
		for (Elements::iterator j = i->begin(), ej = i->end(); j != ej; ++j) {
			if (src.hasNeighbor(nodes[j->first])) {
				numDups++;
				int w = (graph.getEdgeData(src, nodes[j->first], Galois::NONE));
				if (j->second < w)
					graph.getEdgeData(src, nodes[j->first], Galois::NONE)
							= j->second;
			} else {
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
//////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
	LonestarStart(argc, argv, std::cout, name, desc, url);
	makeGraph(inputfile.c_str());
#if BORUVKA_DEBUG
	printGraph();
#endif
	runBodyParallel();
	return 0;
}

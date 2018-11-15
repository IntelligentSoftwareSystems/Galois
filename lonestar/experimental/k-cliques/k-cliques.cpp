/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "galois/graphs/Graph.h"
#include "galois/runtime/Profile.h"

#include <boost/iterator/transform_iterator.hpp>

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

//#define DEBUG 0

const char* name = "K-Cliques";
const char* desc = "Counts the number of K-Cliques in a graph for any given K";
const char* url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);

static cll::opt<unsigned int>
    clique_size("k", // not uint64_t due to a bug in llvm cl
             cll::desc("Clique Size"),
             cll::init(3));


typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;

//typedef galois::graphs::MorphGraph<int,int,true,false,false,true> DAG;
typedef galois::graphs::MorphGraph<int,int,true> DAG;

typedef Graph::GraphNode GNode;
typedef DAG::GraphNode DAGNode;



struct WorkItem {
		std::vector<DAGNode> clique; //set of graph nodes that form a clique
		std::vector<DAGNode> neighbors; //set of neighbors of the clique
};


/**
 * Compares the given two graph nodes n1 and n2.
 * It returns true if
 * 		(1) degree(n1) < degree(n2) (or)
 * 		(2) (degree(n1) = degree(n2)) and (node id of n1 < node id of n2)
 *
 * \param g is the input graph g
 * \param n1 is the first input node
 * \param n2 is the second input node
 * \return True if n1 < n2 else False.
 */

bool lessThan(Graph &g, const GNode& n1, const GNode& n2) {
	  int n1_degree=std::distance(g.edge_begin(n1), g.edge_end(n1));
	  int n2_degree=std::distance(g.edge_begin(n2), g.edge_end(n2));

	  if(n1_degree < n2_degree) {
		  return true;
	  }
	  else if ((n1_degree == n2_degree ) && (g.getData(n1) < g.getData(n2))) {
		  return true;
	  }
  	  else {
	  	  return false;
  	  }
}

/**
 * Extracts neighbors of a given node of directed acyclic graph (DAG)
 *
 * \param n is the node of a DAG
 * \param neighbors is the output vector that contains the neighbors of node n
 * \param dag is the input DAG
 */

void getNeighbors(DAGNode &n, std::vector<DAGNode> &neighbors, DAG &dag) {
	int i=0;
	for (auto jj : dag.edges(n)) {
		DAGNode dst  = dag.getEdgeDst(jj);
		neighbors[i]=dst;
		i++;
	}
}

/**
 * Gives the intersection two neighbor vector. It requires the vectors be sorted according to the node ids.
 *
 * \param jointNeighbors is the output neighbor vector
 * \param srcNeighbors is the first input neighbor vector
 * \param dstNeighbors is the second input neighbor vector
 */

size_t intersect(std::vector<DAGNode> &jointNeighbors, std::vector<DAGNode> &srcNeighbors, std::vector<DAGNode> &dstNeighbors) {
  std::vector<DAGNode>::iterator srcItr= srcNeighbors.begin();
  std::vector<DAGNode>::iterator dstItr= dstNeighbors.begin();
  size_t size=0;

  while (srcItr != srcNeighbors.end() && dstItr != dstNeighbors.end()) {
    if (*srcItr < *dstItr) {
    	++srcItr;
    } else if (*dstItr < *srcItr) {
    	++dstItr;
    } else {
    	jointNeighbors[size] = (*srcItr);
    	size++;
		++srcItr;
    	++dstItr;
    }
  }
  return size;
}

/**
 * Prints edges of a DAG
 *
 * \param dag is the input directed acyclic graph
 */

void printDAG(DAG &dag) {
	std::cout << "DAG: " << "\n";
	for (DAG::iterator ii = dag.begin();  ii != dag.end(); ++ii) {
		DAGNode src = *ii;
		for (DAG::edge_iterator jj = dag.edge_begin(src); jj != dag.edge_end(src); ++jj) {
			 DAGNode dst = dag.getEdgeDst(jj);
			 std::cout << "(" << dag.getData(src) << "," << dag.getData(dst) <<")\n";
		}
	}
}

/**
 * Prints the details of a work item.
 *
 * \param w is the input work item
 * \param dag is a directed acyclic graph
 */

void print_workitem(WorkItem &w, DAG &dag) {
	std::cout << "\n-------------------------------------------------------- \n";
	std::cout << "Clique: ";
	for (std::vector<DAGNode>::iterator v=w.clique.begin(); v != w.clique.end(); ++v) {
		std::cout << dag.getData(*v) << " ";
	}

	std::cout << "Neighborhood:";
	for (std::vector<DAGNode>::iterator nitr=w.neighbors.begin(); nitr != w.neighbors.end(); ++nitr) {
	   std::cout << dag.getData(*nitr) << " ";
	}
	std::cout << "\n-------------------------------------------------------- \n";
}

/**
 * Prints the vertices of a clique
 *
 * \param w is the input work item which contains a clique
 * \param dag is a directed acyclic graph
 */

void print_clique(WorkItem &w, DAG &dag) {

	std::cout << "Clique: ";
	for (std::vector<DAGNode>::iterator v=w.clique.begin(); v != w.clique.end(); ++v) {
		std::cout << dag.getData(*v) << " ";
	}
	std::cout << "\n\n";
}

/**
 * Prints the vertices that are present in the neighbors
 *
 * \param neighbors is the input vector whose vertices need to be printed
 * \param dag is a directed acyclic graph
 * \param size the number of vertices in the neighbors to be printed
 */

void print_neighbors(std::vector<DAGNode> neighbors, DAG &dag, int size) {

	int i=0;
	std::cout << "Neighbors ";
	for (std::vector<DAGNode>::iterator nitr=neighbors.begin(); nitr != neighbors.end() && (i<size); ++nitr, ++i) {
		std::cout << dag.getData(*nitr) << " ";
	}
	std::cout << "\n";
}

/**
 * This procedure computes the number of k-cliques of a given graph.
 *
 * \param k is the clique size
 * \param dag is a directed acyclic graph, which is obtained using an input graph.
 */

void compute_kcliques(unsigned int k, DAG& dag) {

	galois::InsertBag<WorkItem> items; // worklist that monitors the cliques and their neighborhood
	galois::GAccumulator<size_t> num_kcliques; // maintains the number k cliques

	//Initialize the worklist. Each work item contains a single vertex, and all its neighbors.

	galois::StatTimer TWorkInit("Work list initialization");
	galois::StatTimer Tsort("Sorting");
	galois::StatTimer TCliqueResize("Clique Resize");
	galois::StatTimer TNeighborResize("Neighbor Resize");
	galois::StatTimer TProcessWorkItem("Processing work item");
	galois::StatTimer TProcessEdge("Processing Edges");
	galois::StatTimer TIntersect("Neighbor Intersection");
	galois::StatTimer TGetNeighbor("Get Neighbors");

	TWorkInit.start();
	galois::do_all(galois::iterate(dag),
				 [&](DAGNode n) {
					WorkItem w;

					TCliqueResize.start();
					w.clique.resize(1);
					TCliqueResize.stop();

					w.clique[0]=n;
					int neigh_size=std::distance(dag.edge_begin(n), dag.edge_end(n));

					TNeighborResize.start();
					w.neighbors.resize(neigh_size);
					TNeighborResize.stop();
					getNeighbors(n, w.neighbors, dag);

					Tsort.start();
					std::sort (w.neighbors.begin(), w.neighbors.end());
					Tsort.stop();
					items.push(w);

					#ifdef DEBUG
						print_workitem(w,dag);
					#endif
				 },
				 galois::loopname("Initialize")
				);
	TWorkInit.stop();


	//Process a work item w containing a clique C and neighbood N.
	auto processClique = [&](WorkItem& w, auto& ctx) {
		TProcessWorkItem.start();

		if(w.clique.size() == k) {
			//If the clique size is k, increment number of k-cliques by 1. No new work item is generated here.
			num_kcliques += 1;

			#ifdef DEBUG
				print_clique(w,dag);
			#endif
		}
		else if(w.clique.size() == k-1) {
			/**
			 * Clique size is k-1, so form a new clique of size k by adding a vertex from the neighborhood.
			 * So, for a neighborhood for size N, new N K-cliques are formed.
			 */

			#ifdef DEBUG
				std::cout <<"Processing Work Item:";
				print_workitem(w,dag);
			#endif

			for (std::vector<DAGNode>::iterator neighitr=w.neighbors.begin(); neighitr != w.neighbors.end(); ++neighitr) {
				WorkItem nextItem;

				TCliqueResize.start();
				nextItem.clique.resize(w.clique.size()+1);
				TCliqueResize.stop();

				int i=0;
				for (std::vector<DAGNode>::iterator v=w.clique.begin(); v != w.clique.end(); ++v) {
					nextItem.clique[i] = (*v);
					i++;
				}
				nextItem.clique[i] = *neighitr;

				#ifdef DEBUG
					print_clique(nextItem,dag);
				#endif

				num_kcliques += 1;
			}
		}
		else {
			/**
			 * For each edge u->v present in N, construct a new work W with (C', N') and insert it into the work list
			 * where C' = C U {u,v} and
			 * N' = intersection{N, neighbors of u, neighbors of v)
			 */

			for (std::vector<DAGNode>::iterator src=w.neighbors.begin(); src != w.neighbors.end(); ++src) {
				for (std::vector<DAGNode>::iterator dst=w.neighbors.begin(); dst != w.neighbors.end(); ++dst) {
					if((dag.getData(*src) != dag.getData(*dst)) &&
							(dag.findEdge(*src,*dst) != dag.edge_end(*src, galois::MethodFlag::UNPROTECTED))) {

						TProcessEdge.start();
						#ifdef DEBUG
							std::cout <<"Processing Work Item:";
							print_workitem(w,dag);
						#endif

						WorkItem nextItem;

						TCliqueResize.start();
						nextItem.clique.resize(w.clique.size()+2);
						TCliqueResize.stop();

						int i=0;


						for (std::vector<DAGNode>::iterator itr=w.clique.begin(); itr != w.clique.end(); ++itr) {
							nextItem.clique[i] = (*itr);
							i++;
						}
						nextItem.clique[i]=*src;
						nextItem.clique[i+1]=*dst;


						TGetNeighbor.start();
						std::vector<DAGNode> srcNeighbors;
						std::vector<DAGNode> dstNeighbors;
						std::vector<DAGNode> jointNeighbors;
						std::vector<DAGNode> tmpNeighbors;


						size_t srcDegree = std::distance(dag.edge_begin(*src), dag.edge_end(*src));
						size_t dstDegree = std::distance(dag.edge_begin(*dst), dag.edge_end(*dst));

						TGetNeighbor.stop();

						TNeighborResize.start();
						srcNeighbors.resize(srcDegree);
						dstNeighbors.resize(dstDegree);
						tmpNeighbors.resize(std::min(srcDegree, w.neighbors.size()));
						TNeighborResize.stop();

						TGetNeighbor.start();
						getNeighbors(*src, srcNeighbors, dag);
						TGetNeighbor.stop();

						Tsort.start();
						std::sort (srcNeighbors.begin(), srcNeighbors.end());
						Tsort.stop();


						getNeighbors(*dst, dstNeighbors, dag);


						Tsort.start();
						std::sort (dstNeighbors.begin(), dstNeighbors.end());
						Tsort.stop();

						TIntersect.start();
						size_t tmp_neighborsize=intersect(tmpNeighbors, srcNeighbors, w.neighbors);
						TIntersect.stop();

						nextItem.neighbors.resize(tmp_neighborsize);


						TIntersect.start();
						size_t nextItem_neighborsize = intersect(nextItem.neighbors, dstNeighbors, tmpNeighbors);
						TIntersect.stop();

						TNeighborResize.start();
						nextItem.neighbors.resize(nextItem_neighborsize);
						TNeighborResize.stop();


						ctx.push(nextItem);


						#ifdef DEBUG
							std::cout<< "Processing an edge with Source : " << dag.getData(*src) << " Destination : " << dag.getData(*dst) << "\n";
							std::cout<< "Source Neighborhood: ";
							print_neighbors(srcNeighbors, dag, srcNeighbors.size());
							std::cout<< "Temporary Neighborhood: Intersection of Source Neighborhood and WorkItem Neighborhood: ";
							print_neighbors(tmpNeighbors, dag, tmp_neighborsize);
							std::cout<< "Destination Neighborhood: ";
							print_neighbors(dstNeighbors, dag, dstNeighbors.size());
							std::cout<< "NextItem Neighborhood: Intersection of Temporary Neighborhood and Destination Neighborhood: ";
							print_neighbors(nextItem.neighbors, dag, nextItem_neighborsize);
						#endif

						TProcessEdge.stop();

					}
		   		}
		   	}

		}
		TProcessWorkItem.stop();
	};


	// parallel loop that processes work list until it is empty.
	galois::for_each(
		 galois::iterate(items), // initial range using initializer list
		 processClique,                   // operator
		 galois::loopname("process clique"),
		 galois::steal());

	std::cout << "Number of K-Cliques: " << num_kcliques.reduce() << "\n";

}

/*
 * For a given graph (G), it constructs a directed acyclic graph (DAG) corresponding to it.
 * The graph DAG represents a total order among the vertices of the graph G.
 * Procedure:
 * V (DAG) = V(G)
 * An edge u->v is present in DAG if
 *  (1) (u,v) is in E(G) and
 *  (2) (degree(u) < degree(v)) or (degree(u) = degree(v) and u < v)
 *
 * \param graph is the input graph
 * \param dag is the output directed acyclic graph
 */
void constructDAG(Graph& graph, DAG& dag){

	std::vector<DAGNode> nodes;
	nodes.resize(graph.size());

	// Creating the DAG vertices
	galois::do_all(galois::iterate(graph),
		  [&graph, &dag, &nodes](GNode N) {
				int index = graph.getData(N);
				nodes[index] = dag.createNode(index);
				dag.addNode(nodes[index]);
		  }	  // operator as lambda expression
	);

	// Adding the edges depending on the degree of source and destination.
	galois::do_all(galois::iterate(graph),
		  [&graph, &dag, &nodes](GNode N) {
			  for (Graph::edge_iterator edge :
				graph.out_edges(N, galois::MethodFlag::UNPROTECTED)) {
					  GNode dst = graph.getEdgeDst(edge);
					  if (lessThan(graph, N, dst )) {
						  DAGNode srcNode = nodes[graph.getData(N)];
						  DAGNode dstNode = nodes[graph.getData(dst)];
						  dag.addEdge(srcNode, dstNode);
					  }
				  }
		  }  // operator as lambda expression
	);

	#ifdef DEBUG
		printDAG(dag);
	#endif
}

/**
 * Read a graph from the given input file (inputFileName)
 *
 * \param graph is the output graph
 */
void readGraph(Graph& graph) {

	galois::graphs::readGraph(graph, inputFilename);
	size_t index = 0;
	for (GNode n : graph) {
		graph.getData(n) = index++;
	}
}

/**
 * This is an implementation for computing number of k-cliques in a given graph.
 * It uses the edge parallel algorithm described in Danisch et al. WWW 2018 (https://dl.acm.org/citation.cfm?id=3186125).
 * It is implemented using Galois framework.
 */

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);

	Graph graph;
	DAG dag;

	// Read the input Graph
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	readGraph(graph);
	Tinitial.stop();

	// Constructing a DAG from the original graph
	galois::StatTimer Tdag("DAG Construction Time");
	Tdag.start();
	constructDAG(graph, dag);
	Tdag.stop();

	galois::preAlloc(numThreads + 16 * (graph.size() + graph.sizeEdges()) /
									galois::runtime::pagePoolSize());
	galois::reportPageAlloc("MeminfoPre");

	galois::StatTimer T;
	T.start();
	// the main procedure that computes k-cliques
	compute_kcliques(clique_size, dag);

	T.stop();

	galois::reportPageAlloc("MeminfoPost");
	return 0;
}



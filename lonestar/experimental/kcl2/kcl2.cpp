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

// This is a implementation of the WWW'18 paper:
// Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Profile.h"
#include <boost/iterator/transform_iterator.hpp>
#define CHUNK_SIZE 256

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (inputs do NOT need to be symmetrized)";
const char* url  = 0;

enum Algo {
	nodeiterator,
	edgeiterator,
};

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: unsymmetrized graph>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), cll::values(
	clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator"),
	clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"), clEnumValEnd), cll::init(Algo::nodeiterator));
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-clique (default value 3)"), cll::init(3));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Mining/util.h"
#include "Lonestar/subgraph.h"
int core;
typedef std::vector<unsigned> UintVec;
typedef galois::substrate::PerThreadStorage<Subgraph> LocalSubgraph;
typedef galois::substrate::PerThreadStorage<UintVec> LocalVector;

// construct the subgraph induced by vertex u's neighbors
void mksub(Graph &g, GNode u, Subgraph &sg, UintVec &new_id, UintVec &old_id, unsigned k) {
	if (old_id.empty()) {
		new_id.resize(g.size());
		old_id.resize(core);
		for (unsigned i = 0; i < g.size(); i ++) new_id[i] = (unsigned)-1;
	}
	for (unsigned i = 0; i < sg.n[k-1]; i ++) sg.lab[i] = 0;
	unsigned j = 0;
	for (auto e : g.edges(u)) {
		GNode v = g.getEdgeDst(e);
		new_id[v] = j;
		old_id[j] = v;
		sg.lab[j] = k-1;
		sg.vertices[k-1][j] = j;
		sg.d[k-1][j] = 0;//new degrees
		j ++;
	}
	sg.n[k-1] = j; // number of neighbors of u. Since u is in level k, u's neighbors are in level k-1
	//reodering adjacency list and computing new degrees
	for (unsigned i = 0; i < sg.n[k-1]; i ++) {
		unsigned v = old_id[i]; // for each neighbor v of u
		for (auto e : g.edges(v)) {
			GNode w = g.getEdgeDst(e); // w is the neighbor's neighbor
			j = new_id[w];
			if (j != (unsigned)-1) // if w is also a neighbor of u
				sg.adj[sg.core * i + sg.d[k-1][i]++] = j;
		}
	}
	for (auto e : g.edges(u)) {
		GNode v = g.getEdgeDst(e);
		new_id[v] = (unsigned)-1;
	}
}

// each task extends from a vertex, l is the level starting from k-1 and decreases until l=2
void kclique_thread(unsigned l, Subgraph &sg, galois::GAccumulator<long long> &num) {
	if (l == 2) {
		for(unsigned i = 0; i < sg.n[2]; i++) { //list all edges
			unsigned u = sg.vertices[2][i];
			unsigned begin = u * sg.core;
			unsigned end = begin + sg.d[2][u];
			for (unsigned j = begin; j < end; j ++) {
				num += 1; //listing here!!!
			}
		}
		return;
	}
	// compute the subgraphs induced on the neighbors of each node in current level,
	// and then recurse on such a subgraph
	for(unsigned i = 0; i < sg.n[l]; i ++) {
		// for each vertex u in level l
		// a new induced subgraph G[∆G(u)] is built
		unsigned u = sg.vertices[l][i];
		sg.n[l-1] = 0;
		unsigned begin = u * sg.core;
		unsigned end = begin + sg.d[l][u];
		// extend one vertex v which is a neighbor of u
		for (unsigned j = begin; j < end; j ++) {
			// for each out-neighbor v of node u in G, set its label to l − 1
			// if the label was equal to l. We thus have that if a label of a
			// node v is equal to l − 1 it means that node v is in the new subgraph
			unsigned v = sg.adj[j];
			// update info of v
			// relabeling vertices and forming U'.
			if (sg.lab[v] == l) {
				sg.lab[v] = l-1;
				sg.vertices[l-1][sg.n[l-1]++] = v;
				sg.d[l-1][v] = 0;//new degrees
			}
		}
		// for each out-neighbor v of u
		// reodering adjacency list and computing new degrees
		for (unsigned j = 0; j < sg.n[l-1]; j ++) {
			unsigned v = sg.vertices[l-1][j];
			begin = v * sg.core;
			end = begin + sg.d[l][v];
			// move all the out-neighbors of v with label equal to l − 1 
			// in the first part of ∆(v) (by swapping nodes),
			// and compute the out-degree of node v in the new subgraph
			// updating d(v). The first d(v) nodes in ∆(v) are
			// thus the out-neighbors of v in the new subgraph.
			for (unsigned k = begin; k < end; k ++) {
				unsigned w = sg.adj[k];
				if (sg.lab[w] == l-1) {
					sg.d[l-1][v] ++;
				}
				else {
					sg.adj[k--] = sg.adj[--end];
					sg.adj[end] = w;
				}
			}
		}
		kclique_thread(l-1, sg, num);
		for (unsigned j = 0; j < sg.n[l-1]; j ++) {//restoring labels
			unsigned v = sg.vertices[l-1][j];
			sg.lab[v] = l;
		}
	}
}

void KclSolver(Graph& graph) {
	galois::GAccumulator<long long> total_num;
	total_num.reset();
	LocalSubgraph lsub;
	LocalVector lold, lnew;
	for (unsigned i = 0; i < lsub.size(); i++)
		lsub.getLocal(i)->allocate(core, k);
	galois::for_each(
		galois::iterate(graph.begin(), graph.end()),
		[&](const GNode& u, auto& ctx) {
			Subgraph *sg = lsub.getLocal();
			UintVec *new_id = lnew.getLocal();
			UintVec *old_id = lold.getLocal();
			//Subgraph sg(core, k);
			mksub(graph, u, *sg, *new_id, *old_id, k);
			kclique_thread(k-1, *sg, total_num);
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("KclSolver")
	);
	galois::gPrint("\n\ttotal_num_cliques = ", total_num.reduce(), "\n\n");
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	core = read_graph(graph, filetype, filename, true);
	Tinitial.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");

	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	switch (algo) {
		case nodeiterator:
			KclSolver(graph);
			break;
		case edgeiterator:
			std::cerr << "Not supported currently\n";
			break;
		default:
			std::cerr << "Unknown algo: " << algo << "\n";
	}
	Tcomp.stop();
	return 0;
}

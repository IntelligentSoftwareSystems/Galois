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
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#define USE_SIMPLE
#define DEBUG 0

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph";
const char* url  = 0;

enum Algo {
	nodeiterator,
	edgeiterator,
};

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), cll::values(
	clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator"),
	clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"), clEnumValEnd),
	cll::init(Algo::nodeiterator));
static cll::opt<unsigned> k("k",
	cll::desc("max number of vertices in k-clique (default value 3)"), cll::init(3));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Mining/miner.h"
#include "Lonestar/mgraph.h"
#include "Mining/util.h"
#include "Lonestar/subgraph.h"
#define CHUNK_SIZE 256
int core;
typedef std::vector<unsigned> UintVec;
typedef galois::substrate::PerThreadStorage<Subgraph> LocalSubgraph;
typedef galois::substrate::PerThreadStorage<UintVec> LocalVector;

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
	sg.n[k-1] = j;
	for (unsigned i = 0; i < sg.n[k-1]; i ++) {//reodering adjacency list and computing new degrees
		unsigned v = old_id[i];
		for (auto e : g.edges(v)) {
			GNode w = g.getEdgeDst(e);
			j = new_id[w];
			if (j != (unsigned)-1)
				sg.adj[sg.core * i + sg.d[k-1][i]++] = j;
		}
	}
	for (auto e : g.edges(u)) {
		GNode v = g.getEdgeDst(e);
		new_id[v] = (unsigned)-1;
	}
}

void kclique_thread(unsigned l, Subgraph &sg, galois::GAccumulator<long long> &num) {
	if (l == 2) {
		for(unsigned i = 0; i < sg.n[2]; i++) { //list all edges
			unsigned u = sg.vertices[2][i];
			unsigned end = u * sg.core + sg.d[2][u];
			for (unsigned j = u * sg.core; j < end; j ++) {
				num += 1; //listing here!!!
			}
		}
		return;
	}
	printf("TODO\n");
	for(unsigned i = 0; i < sg.n[l]; i ++) {
		unsigned u = sg.vertices[l][i];
		//printf("%u %u\n",i,u);
		sg.n[l-1] = 0;
		unsigned end = u*sg.core+sg.d[l][u];
		for (unsigned j = u*sg.core; j < end; j ++) {//relabeling vertices and forming U'.
			unsigned v = sg.adj[j];
			if (sg.lab[v] == l) {
				sg.lab[v] = l-1;
				sg.vertices[l-1][sg.n[l-1]++] = v;
				sg.d[l-1][v] = 0;//new degrees
			}
		}
		for (unsigned j = 0; j < sg.n[l-1]; j ++) {//reodering adjacency list and computing new degrees
			unsigned v = sg.vertices[l-1][j];
			end = sg.core * v + sg.d[l][v];
			for (unsigned k = sg.core * v; k < end; k ++) {
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
	MGraph mgraph(true);
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	if (filetype == "txt") {
		printf("Reading .lg file: %s\n", filename.c_str());
		mgraph.read_txt(filename.c_str());
		genGraph(mgraph, graph);
	} else if (filetype == "adj") {
		printf("Reading .adj file: %s\n", filename.c_str());
		mgraph.read_adj(filename.c_str());
		genGraph(mgraph, graph);
	} else if (filetype == "mtx") {
		printf("Reading .mtx file: %s\n", filename.c_str());
		mgraph.read_mtx(filename.c_str(), true); //symmetrize
		genGraph(mgraph, graph);
	} else if (filetype == "gr") {
		printf("Currently .gr file not supported\n");
		exit(1);
		printf("Reading .gr file: %s\n", filename.c_str());
		galois::graphs::readGraph(graph, filename);
		for (GNode n : graph) graph.getData(n) = 1;
	} else { printf("Unkown file format\n"); exit(1); }
	core = mgraph.get_core();
	//print_graph(graph);
	Tinitial.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	switch (algo) {
		case nodeiterator:
			KclSolver(graph);
			break;
		case edgeiterator:
			break;
		default:
			std::cerr << "Unknown algo: " << algo << "\n";
	}
	Tcomp.stop();
	return 0;
}

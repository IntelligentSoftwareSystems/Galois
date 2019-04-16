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
#include "galois/runtime/Profile.h"
#include <boost/iterator/transform_iterator.hpp>
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#define USE_SIMPLE
#define DEBUG 0
#define ENABLE_LABEL 0

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
#define CHUNK_SIZE 256

// insert edges into the worklist (task queue)
void initialization(Graph& graph, BaseEmbeddingQueue &queue) {
	printf("\n=============================== Init ===============================\n");
	galois::do_all(
		galois::iterate(graph.begin(), graph.end()),
		[&](const GNode& src) {
			for (auto e : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e);
				if (src < dst) {
					BaseEmbedding new_emb;
					new_emb.push_back(src);
					new_emb.push_back(dst);
					queue.push_back(new_emb);
				}
			}
		},
		galois::chunk_size<512>(), galois::steal(), galois::loopname("Initialization")
	);
}

void KclSolver(Graph& graph, Miner &miner) {
	std::cout << "=============================== Start ===============================\n";
	BaseEmbeddingQueue queue, queue2;
	initialization(graph, queue);
	printout_embeddings(0, miner, queue, DEBUG);
	unsigned level = 1;
	while (level < k-1) {
		std::cout << "\n============================== Level " << level << " ==============================\n";
		std::cout << "\n------------------------------------- Step 1: Joining -------------------------------------\n";
		queue2.clear();
		galois::for_each(
			galois::iterate(queue),
			[&](const BaseEmbedding& emb, auto& ctx) {
				miner.extend_vertex(emb, queue2);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("ExtendVertex")
		);
		miner.update_base_embedding_size(); // increase the embedding size since one more edge added
		printout_embeddings(level, miner, queue2, DEBUG);

		std::cout << "\n----------------------------------- Step 2: Aggregation -----------------------------------\n";
		queue.clear();
#if 0
		miner.aggregate_clique(queue2, queue); // sequential implementaion
#else
		// Parallel aggregation
		LocalSimpleMap lmap;
		SimpleMap map;
		galois::for_each(
			galois::iterate(queue2),
			[&](BaseEmbedding& emb, auto& ctx) {
				miner.aggregate_clique_each(emb, *(lmap.getLocal()), queue);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Aggregation")
		);
		for (unsigned i = 0; i < lmap.size(); i++) {
			SimpleMap sm = *lmap.getLocal(i);
			for (auto element : sm) {
				auto it = map.find(element.first);
				if (it != map.end()) {
					if(it->second + element.second == it->first.size() - 1) {
						queue.push_back(it->first);
						map.erase(it);
					} else map[element.first] += element.second;
				} else
					map[element.first] = element.second;
			}
		}
#endif
		printout_embeddings(level, miner, queue, DEBUG);
		level ++;
		//std::cout << "Number of " << level+1 << "-cliques is: " << std::distance(queue.begin(), queue.end()) << "\n";
	}
	std::cout << "\n=============================== Done ===============================\n\n";
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	MGraph mgraph;
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
		printf("Reading .gr file: %s\n", filename.c_str());
		galois::graphs::readGraph(graph, filename);
		for (GNode n : graph) graph.getData(n) = 1;
	} else { printf("Unkown file format\n"); exit(1); }
	//print_graph(graph);
	unsigned sizeof_embedding = 2 * sizeof(SimpleElement);
	Miner miner(false, sizeof_embedding, &graph);
	Tinitial.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	galois::reportPageAlloc("MeminfoPre");
	galois::StatTimer T;
	T.start();
	switch (algo) {
		case nodeiterator:
			KclSolver(graph, miner);
			break;
		case edgeiterator:
			break;
		default:
			std::cerr << "Unknown algo: " << algo << "\n";
	}
	T.stop();
	galois::reportPageAlloc("MeminfoPost");
	return 0;
}

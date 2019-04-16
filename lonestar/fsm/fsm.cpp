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

#define ENABLE_LABEL 1

const char* name = "FSM";
const char* desc = "Frequent subgraph mining";
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
	cll::desc("max number of vertices in k-motif (default value 0)"), cll::init(0));
static cll::opt<unsigned> minsup("minsup",
	cll::desc("minimum suuport (default value 0)"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;
int total_num = 0;

#include "Mining/miner.h"
#include "Lonestar/mgraph.h"
#include "Mining/util.h"
#define CHUNK_SIZE 256

void init(Graph& graph, EmbeddingQueue &queue) {
	//print_graph(graph);
	printf("\n=============================== Init ===============================\n\n");
	galois::do_all(
		galois::iterate(graph.begin(), graph.end()),
		[&](const GNode& src) {
			auto& src_label = graph.getData(src);
			//Graph::edge_iterator first = graph.edge_begin(src);
			//Graph::edge_iterator last = graph.edge_end(src);
			//for (auto e = first; e != last; ++ e) {
			for (auto e : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e);
				if (src < dst) {
					auto& dst_label = graph.getData(dst);
					Embedding new_emb;
					new_emb.push_back(Element_In_Tuple(src, 0, src_label));
					new_emb.push_back(Element_In_Tuple(dst, 0, dst_label));
					queue.push_back(new_emb);
				}
			}
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
		galois::loopname("Initialization")
	);
	//}
}

void aggregator(Miner& miner, EmbeddingQueue& queue, CgMap& cg_map) {
	// Parallel quick aggregation
	QpMap qp_map; // quick pattern map
	LocalQpMap qp_localmap; // quick pattern local map for each thread
	galois::for_each(
		galois::iterate(queue),
		[&](const Embedding& emb, auto& ctx) {
			miner.quick_aggregate_each(emb, *(qp_localmap.getLocal()));
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
		galois::loopname("QuickAggregation")
	);
	for (unsigned i = 0; i < qp_localmap.size(); i++) {
		QpMap qp_lmap = *qp_localmap.getLocal(i);
		for (auto element : qp_lmap) {
			if (qp_map.find(element.first) != qp_map.end())
				qp_map[element.first] += element.second;
			else
				qp_map[element.first] = element.second;
		}
	}

	// Parallel canonical aggregation
	LocalCgMap cg_localmap; // canonical pattern local map for each thread
	galois::do_all(
		galois::iterate(qp_map),
		[&](std::pair<Quick_Pattern, int> qp) {
			Quick_Pattern subgraph = qp.first;
			int count = qp.second;
			miner.canonical_aggregate_each(subgraph, count, *(cg_localmap.getLocal()));
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
		galois::loopname("CanonicalAggregation")
	);
	for (unsigned i = 0; i < cg_localmap.size(); i++) {
		CgMap cg_lmap = *cg_localmap.getLocal(i);
		for (auto element : cg_lmap) {
			if (cg_map.find(element.first) != cg_map.end())
				cg_map[element.first] += element.second;
			else
				cg_map[element.first] = element.second;
		}
	}
	for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
		if (it->second >= minsup) total_num ++;
}

void filter(Miner& miner, EmbeddingQueue& in_queue, EmbeddingQueue& out_queue, CgMap cg_map) {
	//miner.filter(in_queue, out_queue, cg_map);
	///*
	std::cout << "num_patterns: " << cg_map.size() << " num_embeddings: " 
		<< std::distance(in_queue.begin(), in_queue.end()) << "\n";
	galois::for_each(
		galois::iterate(in_queue),
		[&](Embedding& emb, auto& ctx) {
			miner.filter_each(emb, out_queue, cg_map);
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
		galois::loopname("Filter")
	);
	//*/
}

#define DEBUG 0
#define SHOW_OUTPUT 0
void FsmSolver(Graph &graph, Miner &miner) {
	std::cout << "=============================== Start ===============================\n";
	EmbeddingQueue queue, filtered_queue;
	init(graph, queue);
	if(DEBUG) printout_embeddings(0, miner, queue);

	std::cout << "\n----------------------------------- Aggregating -----------------------------------\n";
	CgMap cg_map; // canonical graph map
	aggregator(miner, queue, cg_map);
	//std::cout << "num_patterns: " << cg_map.size() << " num_embeddings: " 
	//	<< std::distance(queue.begin(), queue.end()) << "\n";
	if(SHOW_OUTPUT) miner.printout_agg(cg_map);

	std::cout << "\n------------------------------------ Filtering ------------------------------------\n";
	filter(miner, queue, filtered_queue, cg_map);
	printout_embeddings(0, miner, filtered_queue);
	unsigned level = 1;

	while (level < k) {
		std::cout << "\n============================== Level " << level << " ==============================\n";
		std::cout << "\n------------------------------------- Joining -------------------------------------\n";
		queue.clear();
		galois::for_each(
			galois::iterate(filtered_queue),
			[&](const Embedding& embedding, auto& ctx) {
				miner.extend_edge(k, embedding, queue);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Join")
		);
		miner.update_embedding_size();
		printout_embeddings(level, miner, queue);

		std::cout << "\n----------------------------------- Aggregating -----------------------------------\n";
		cg_map.clear();
		aggregator(miner, queue, cg_map);
		//std::cout << "num_patterns: " << cg_map.size() << " num_embeddings: " 
		//	<< std::distance(queue.begin(), queue.end()) << "\n";
		if(SHOW_OUTPUT) miner.printout_agg(cg_map);

		std::cout << "\n------------------------------------ Filtering ------------------------------------\n";
		filtered_queue.clear();
		filter(miner, queue, filtered_queue, cg_map);
		printout_embeddings(level, miner, filtered_queue);
		level ++;
	}
	std::cout << "\n=============================== Done ===============================\n\n";
	std::cout << "Number of frequent subgraphs (minsup=" << minsup << "): " << total_num << "\n";
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
	} else if (filetype == "gr") {
		printf("Reading .gr file: %s\n", filename.c_str());
		galois::graphs::readGraph(graph, filename);
		for (GNode n : graph) {
			graph.getData(n) = rand() % 10 + 1;
			for (auto e : graph.edges(n)) {
				graph.getEdgeData(e) = 1;
			}
		}
	} else { printf("Unkown file format\n"); exit(1); }
	Tinitial.stop();
	std::cout << "k = " << k << std::endl;
	std::cout << "minsup = " << minsup << std::endl;
	std::cout << "num_threads = " << numThreads << std::endl;
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	//print_graph(graph);
	unsigned sizeof_emb = 2 * sizeof(Element_In_Tuple);
	Miner miner(true, sizeof_emb, &graph);
	miner.set_threshold(minsup);
	galois::reportPageAlloc("MeminfoPre");
	galois::StatTimer T;
	T.start();
	FsmSolver(graph, miner);
	T.stop();
	galois::reportPageAlloc("MeminfoPost");
	return 0;
}

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

#include "Mining/miner.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining";
const char* url  = 0;

enum Algo {
	nodeiterator,
	edgeiterator,
};

namespace cll = llvm::cl;
static cll::opt<std::string>
inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), cll::values(
	clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator"),
	clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"), clEnumValEnd),
	cll::init(Algo::nodeiterator));
static cll::opt<unsigned> k("k",
	cll::desc("max number of vertices in k-motif (default value 0)"), cll::init(0));
static cll::opt<unsigned> minsup("minsup",
	cll::desc("minimum suuport (default value 0)"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

void init(Graph& graph, EmbeddingQueue &queue) {
	printf("\n=============================== Init ===============================\n\n");
	galois::do_all(
		galois::iterate(graph),
		[&](const GNode& src) {
			// for each vertex
			auto& src_label = graph.getData(src);
			Graph::edge_iterator first = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
			Graph::edge_iterator last = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
			// foe each edge of this vertex
			for (auto e = first; e != last; ++ e) {
				GNode dst = graph.getEdgeDst(e);
				auto& dst_label = graph.getData(dst);
				Embedding new_tuple;
				new_tuple.push_back(Element_In_Tuple(src, 0, src_label));
				new_tuple.push_back(Element_In_Tuple(dst, 0, dst_label));
				if(!Pattern::is_automorphism_init(new_tuple)) {
					queue.push_back(new_tuple);
				}
			}
		},
		galois::chunk_size<512>(), galois::steal(),
		galois::loopname("Initialization")
	);
}

void construct_edgelist(Graph& graph, std::vector<LabeledEdge> &edgelist) {
	for (Graph::iterator it = graph.begin(); it != graph.end(); it ++) {
//	galois::do_all(
//		galois::iterate(graph),
//		[&](const GNode& src) {
			// for each vertex
			GNode src = *it;
			auto& src_label = graph.getData(src);
			Graph::edge_iterator first = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
			Graph::edge_iterator last = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
			// foe each edge of this vertex
			for (auto e = first; e != last; ++ e) {
				GNode dst = graph.getEdgeDst(e);
				auto& dst_label = graph.getData(dst);
				LabeledEdge edge(src, dst, src_label, dst_label);
				edgelist.push_back(edge);
			}
		}
//	);
	assert(edgelist.size() == graph.sizeEdges());
}

void printout_tuples(int level, Miner& miner, EmbeddingQueue& queue) {
	int num_tuples = std::distance(queue.begin(), queue.end());
	unsigned tuple_size = miner.get_sizeof_tuple();
	std::cout << "Number of tuples in level " << level << ": " << num_tuples << " (tuple_size=" << tuple_size << ")" << std::endl;
	for (EmbeddingQueue::iterator it = queue.begin(); it != queue.end(); it ++)
		miner.printout_tuple(level, *it);
}

void aggregator(Miner& miner, EmbeddingQueue& queue, CgMap& cg_map) {
	// Parallel quick aggregation
	LocalQpMap quick_pattern_localmap;
	QpMap quick_patterns_map;
	galois::for_each(
		galois::iterate(queue),
		[&](const Embedding& emb, auto& ctx) {
			miner.quick_aggregate_each(emb, *(quick_pattern_localmap.getLocal()));
		},
		galois::chunk_size<128>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<128>>(),
		galois::loopname("QuickAggregation")
	);
	for (unsigned i = 0; i < quick_pattern_localmap.size(); i++) {
		QpMap qp_map = *quick_pattern_localmap.getLocal(i);
		for (auto element : qp_map) {
			if (quick_patterns_map.find(element.first) != quick_patterns_map.end())
				quick_patterns_map[element.first] += element.second;
			else
				quick_patterns_map[element.first] = element.second;
		}
	}

	// Parallel canonical aggregation
	LocalCgMap cg_localmap; // canonical graph local map for each thread
	galois::do_all(
		galois::iterate(quick_patterns_map),
		[&](std::pair<Quick_Pattern, int> qp) {
			Quick_Pattern subgraph = qp.first;
			int count = qp.second;
			miner.canonical_aggregate_each(subgraph, count, *(cg_localmap.getLocal()));
		},
		galois::chunk_size<128>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<128>>(),
		galois::loopname("CanonicalAggregation")
	);
	for (unsigned i = 0; i < cg_localmap.size(); i++) {
		CgMap cgm = *cg_localmap.getLocal(i);
		for (auto element : cgm) {
			if (cg_map.find(element.first) != cg_map.end())
				cg_map[element.first] += element.second;
			else
				cg_map[element.first] = element.second;
		}
	}
}

void filter(Miner& miner, EmbeddingQueue& queue, EmbeddingQueue& filtered_queue, int minsup, CgMap cg_map) {
	galois::for_each(
		galois::iterate(queue),
		[&](const Embedding& emb, auto& ctx) {
			miner.filter_each(emb, filtered_queue, minsup, cg_map);
		},
		galois::chunk_size<128>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<128>>(),
		galois::loopname("Filter")
	);
}

#define DEBUG 0
#define SHOW_OUTPUT 0
void FsmSolver(Graph& graph) {
	std::cout << "k = " << k << std::endl;
	std::cout << "minsup = " << minsup << std::endl;
	std::vector<LabeledEdge> edge_list;
	construct_edgelist(graph, edge_list);
	std::cout << "=============================== Start ===============================\n";
	EmbeddingQueue queue, filtered_queue;
	init(graph, queue);
	unsigned sizeof_tuple = 2 * sizeof(Element_In_Tuple);
	Miner miner(true, sizeof_tuple, graph.size(), graph.sizeEdges(), edge_list);
	if(DEBUG) printout_tuples(0, miner, queue);

	std::cout << "\n----------------------------------- Aggregating -----------------------------------\n";
	CgMap cg_map; // canonical graph map
	aggregator(miner, queue, cg_map);
	if(SHOW_OUTPUT) miner.printout_agg(cg_map);

	std::cout << "\n------------------------------------ Filtering ------------------------------------\n";
	filter(miner, queue, filtered_queue, minsup, cg_map);
	if(DEBUG) printout_tuples(0, miner, filtered_queue);
	unsigned level = 1;

	while (level < k) {
		std::cout << "\n============================== Level " << level << " ==============================\n";
		std::cout << "\n------------------------------------- Joining -------------------------------------\n";
		queue.clear();
		galois::for_each(
			galois::iterate(filtered_queue),
			[&](const Embedding& embedding, auto& ctx) {
				miner.join_each(k, embedding, queue);
			},
			galois::chunk_size<128>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<128>>(),
			galois::loopname("Join")
		);
		miner.update_tuple_size();
		if(DEBUG) printout_tuples(level, miner, queue);

		std::cout << "\n----------------------------------- Aggregating -----------------------------------\n";
		cg_map.clear();
		aggregator(miner, queue, cg_map);
		if(SHOW_OUTPUT) miner.printout_agg(cg_map);

		std::cout << "\n------------------------------------ Filtering ------------------------------------\n";
		filtered_queue.clear();
		filter(miner, queue, filtered_queue, minsup, cg_map);
		if(DEBUG) printout_tuples(level, miner, filtered_queue);
		level ++;
	}
	std::cout << "\n=============================== Done ===============================\n\n";
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinitial("GraphReadingTime");
	//galois::gPrint("Start readGraph\n");
	Tinitial.start();
	//galois::gPrint("Start loading", inputFilename, "\n");
	galois::graphs::readGraph(graph, inputFilename);
	for (GNode n : graph)
		//graph.getData(n) = 1;
		graph.getData(n) = rand() % 10 + 1;
	Tinitial.stop();
	//galois::gPrint("Done readGraph\n");
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	galois::reportPageAlloc("MeminfoPre");
	galois::StatTimer T;
	T.start();
	switch (algo) {
		case nodeiterator:
			FsmSolver(graph);
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

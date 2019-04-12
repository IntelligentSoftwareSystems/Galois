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
#define DEBUG 0
#define ENABLE_LABEL 0

const char* name = "Motif";
const char* desc = "Counts the motifs in a graph";
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
	clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"), clEnumValEnd), cll::init(Algo::nodeiterator));
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-motif(default value 0)"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Mining/miner.h"
#include "Lonestar/mgraph.h"
#include "Mining/util.h"
#define CHUNK_SIZE 256
// insert edges into the worklist (task queue)
void initialization(Graph& graph, EmbeddingQueue &queue) {
	printf("\n=============================== Init ===============================\n\n");
	galois::do_all(
		//galois::iterate(graph),
		galois::iterate(graph.begin(), graph.end()),
		[&](const GNode& src) {
			// for each vertex
			//auto& src_label = graph.getData(src);
			//Graph::edge_iterator first = graph.edge_begin(src);
			//Graph::edge_iterator last = graph.edge_end(src);
			// for each edge of this vertex
			//for (auto e = first; e != last; ++ e) {
			for (auto e : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e);
				//auto& dst_label = graph.getData(dst);
				if(src < dst) {
					// create a new embedding
					Embedding new_emb;
					new_emb.push_back(Element_In_Tuple(src));
					new_emb.push_back(Element_In_Tuple(dst));
					queue.push_back(new_emb);
				}
			}
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
		galois::loopname("Initialization")
	);
}

void MotifSolver(Graph& graph, Miner &miner) {
	std::cout << "=============================== Start ===============================\n";
	// task queues. double buffering
	EmbeddingQueue queue, queue2;
	// initialize the task queue
	initialization(graph, queue);
	if(DEBUG) printout_embeddings(0, miner, queue);
	unsigned level = 1;
	int queue_size = std::distance(queue.begin(), queue.end());
	std::cout << "Queue size = " << queue_size << std::endl;

	// a level-by-level approach for Apriori search space (breadth first seach)
	while (level < k) {
		std::cout << "\n============================== Level " << level << " ==============================\n";
		std::cout << "\n------------------------------------- Step 1: Joining -------------------------------------\n";
		// for each embedding in the task queue, do the edge-extension operation
		galois::for_each(
			galois::iterate(queue),
			[&](const Embedding& emb, auto& ctx) {
				miner.extend_edge(k, emb, queue2);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Join")
		);
		miner.update_embedding_size(); // increase the embedding size since one more edge added
		queue.swap(queue2);
		queue2.clear();
		if(DEBUG) printout_embeddings(level, miner, queue);

		std::cout << "\n----------------------------------- Step 2: Aggregation -----------------------------------\n";
		// Sub-step 1: aggregate on quick patterns: gather embeddings into different quick patterns
		//miner.aggregate(queue); // sequential implementaion

		// Parallel aggregation
		LocalQpMap quick_pattern_localmap;
		QpMap quick_patterns_map;
		galois::for_each(
			galois::iterate(queue),
			[&](const Embedding& emb, auto& ctx) {
				miner.quick_aggregate_each(emb, *(quick_pattern_localmap.getLocal()));
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
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

		// Sub-step 2: aggregate on canonical patterns: gather quick patterns into different canonical patterns
		//miner.canonical_aggregate(quick_patterns_map);
		/*
		// sequential implementation
		for (auto qp = quick_patterns_map.begin(); qp != quick_patterns_map.end(); ++ qp) {
			Quick_Pattern subgraph = qp->first;
			int count = qp->second;
			miner.canonical_aggregate_each(subgraph, count, cg_map);
		}
		//*/

		// Parallel aggregation
		LocalCgMap cg_localmap; // canonical graph local map for each thread
		CgMap cg_map; // canonical graph map
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
		miner.printout_agg(cg_map);
		std::cout << "num_patterns: " << cg_map.size() << " num_quick_patterns: " << quick_patterns_map.size()
			<< " num_embeddings: " << std::distance(queue.begin(), queue.end()) << "\n";
		level ++;
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
		std::ifstream in;
		in.open(filename.c_str(), std::ios::in);
		mgraph.read_txt(in);
		in.close();
		genGraph(mgraph, graph);
	} else if (filetype == "adj") {
		printf("Reading .adj file: %s\n", filename.c_str());
		std::ifstream in;
		in.open(filename.c_str(), std::ios::in);
		mgraph.read_adj(in);
		in.close();
		genGraph(mgraph, graph);
	} else if (filetype == "gr") {
		printf("Reading .gr file: %s\n", filename.c_str());
		galois::graphs::readGraph(graph, filename);
		for (GNode n : graph) graph.getData(n) = 1;
	} else { printf("Unkown file format\n"); exit(1); }
	//print_graph(graph);
	std::vector<LabeledEdge> edge_list;
	construct_edgelist(graph, edge_list);
	// the initial size of a embedding is 2 (nodes) for single-edge embeddings
	unsigned sizeof_embedding = 2 * sizeof(Element_In_Tuple);
	// a miner defines the operators (join and aggregation)
	Miner miner(false, sizeof_embedding, &graph, edge_list);
	Tinitial.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	galois::reportPageAlloc("MeminfoPre");
	galois::StatTimer T;
	T.start();
	switch (algo) {
		case nodeiterator:
			MotifSolver(graph, miner);
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

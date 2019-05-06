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
#define DEBUG 0

const char* name = "Motif Counting";
const char* desc = "Counts the motifs in a graph using BFS traversal";
const char* url  = 0;
namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
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
		// for each vertex
		galois::iterate(graph.begin(), graph.end()),
		[&](const GNode& src) {
			// for each edge of this vertex
			for (auto e : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e);
				if(src < dst) {
					// create a new embedding
					Embedding new_emb;
					new_emb.push_back(ElementType(src));
					new_emb.push_back(ElementType(dst));
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
				miner.extend_edge(k, emb, queue2); // edge extension
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Join")
		);
		miner.update_embedding_size(); // increase the embedding size since one more edge added
		queue.swap(queue2);
		queue2.clear();
		printout_embeddings(level, miner, queue);

		std::cout << "\n----------------------------------- Step 2: Aggregation -----------------------------------\n";
		// Sub-step 1: aggregate on quick patterns: gather embeddings into different quick patterns
		QpMapFreq qp_map; // quick patterns map for counting the frequency
		//miner.quick_aggregate(queue, qp_map); // sequential implementaion
		// Parallel quick pattern aggregation
		LocalQpMapFreq qp_localmap; // quick patterns local map for each thread
		galois::for_each(
			galois::iterate(queue),
			[&](Embedding& emb, auto& ctx) {
				miner.quick_aggregate_each(emb, *(qp_localmap.getLocal())); // quick pattern aggregation
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("QuickAggregation")
		);
		// merging results sequentially
		for (unsigned i = 0; i < qp_localmap.size(); i++) {
			QpMapFreq qp_lmap = *qp_localmap.getLocal(i);
			for (auto element : qp_lmap) {
				if (qp_map.find(element.first) != qp_map.end())
					qp_map[element.first] += element.second;
				else
					qp_map[element.first] = element.second;
			}
		}

		// Sub-step 2: aggregate on canonical patterns: gather quick patterns into different canonical patterns
		CgMapFreq cg_map; // canonical graph map for couting the frequency
		//miner.canonical_aggregate(qp_map, cg_map);
		// Parallel canonical pattern aggregation
		LocalCgMapFreq cg_localmap; // canonical graph local map for each thread
		galois::do_all(
			galois::iterate(qp_map),
			[&](std::pair<QuickPattern, Frequency> qp) {
				miner.canonical_aggregate_each(qp.first, qp.second, *(cg_localmap.getLocal())); // canonical pattern aggregation
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
			//galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("CanonicalAggregation")
		);
		// merging results sequentially
		for (unsigned i = 0; i < cg_localmap.size(); i++) {
			CgMapFreq cg_lmap = *cg_localmap.getLocal(i);
			for (auto element : cg_lmap) {
				if (cg_map.find(element.first) != cg_map.end())
					cg_map[element.first] += element.second;
				else
					cg_map[element.first] = element.second;
			}
		}
		miner.printout_agg(cg_map);
		std::cout << "num_patterns: " << cg_map.size() << " num_quick_patterns: " << qp_map.size()
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
	galois::StatTimer Tinit("GraphReadingTime");
	Tinit.start();
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
		for (GNode n : graph) graph.getData(n) = 1;
	} else { printf("Unkown file format\n"); exit(1); }
	//print_graph(graph);
	// the initial size of a embedding is 2 (nodes) for single-edge embeddings
	unsigned sizeof_embedding = 2 * sizeof(ElementType);
	galois::gPrint("Embedding Element size: ", sizeof(ElementType), "\n");
	// a miner defines the operators (join and aggregation)
	Miner miner(&graph, sizeof_embedding);
	Tinit.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	MotifSolver(graph, miner);
	Tcomp.stop();
	return 0;
}

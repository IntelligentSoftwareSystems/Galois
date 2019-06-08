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

const char* name = "Motif Counting";
const char* desc = "Counts the edge-induced motifs in a graph using BFS traversal";
const char* url  = 0;
namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-motif(default value 0)"), cll::init(0));
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Mining/element.h"
typedef StructuralElement ElementType;
#include "Mining/miner.h"
#include "Mining/util.h"
#define CHUNK_SIZE 256
typedef EdgeEmbedding EmbeddingT;
typedef EdgeEmbeddingQueue EmbeddingQueueT;

// insert edges into the worklist (task queue)
void initialization(Graph& graph, EmbeddingQueueT &queue) {
	printf("\n=============================== Init ================================\n\n");
	galois::do_all(
		// for each vertex
		galois::iterate(graph.begin(), graph.end()),
		[&](const GNode& src) {
			// for each edge of this vertex
			for (auto e : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e);
				if(src < dst) {
					// create a new embedding
					EmbeddingT new_emb;
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
	EmbeddingQueueT queue, queue2; // task queues. double buffering
	initialization(graph, queue); // initialize the task queue
	if(show) queue.printout_embeddings(0);
	unsigned level = 1;
	int queue_size = std::distance(queue.begin(), queue.end());
	unsigned max_num_edges = k * (k - 1) / 2; // maximum number of edges in k-motif (i.e. k-clique)

	// a level-by-level approach for Apriori search space (breadth first seach)
	//while (level < k) { // to get the same output as RStream (which is not complete)
	while (queue_size > 0 && level < max_num_edges) { // to get the complete (correct) k-motif output
		std::cout << "\n============================== Level " << level << " ==============================\n";
		std::cout << "\n------------------------- Step 1: Expanding -------------------------\n";
		// for each embedding in the task queue, do the edge-extension operation
		galois::for_each(
			galois::iterate(queue),
			[&](const EmbeddingT& emb, auto& ctx) {
				miner.extend_edge(k, emb, queue2); // edge extension
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Expanding")
		);
		queue.swap(queue2);
		queue2.clear();
		if(show) queue.printout_embeddings(level);

		std::cout << "\n------------------------ Step 2: Aggregation ------------------------\n";
		// Sub-step 1: aggregate on quick patterns: gather embeddings into different quick patterns
		QpMapFreq qp_map; // quick patterns map for counting the frequency
		//miner.quick_aggregate(queue, qp_map); // sequential implementaion
		// Parallel quick pattern aggregation
		LocalQpMapFreq qp_localmap; // quick patterns local map for each thread
		galois::for_each(
			galois::iterate(queue),
			[&](EmbeddingT& emb, auto& ctx) {
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
			[&](std::pair<QPattern, Frequency> qp) {
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
		queue_size = std::distance(queue.begin(), queue.end());
		if(show) std::cout << "num_patterns: " << cg_map.size() << " num_quick_patterns: " << qp_map.size()
					<< " num_embeddings: " << queue_size << "\n";
		level ++;
	}
	std::cout << "\n=============================== Done ===============================\n\n";
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinit("GraphReadingTime");
	Tinit.start();
	read_graph(graph, filetype, filename);
	Tinit.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");

	// a miner defines the operators (expanding and aggregation)
	Miner miner(&graph);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	MotifSolver(graph, miner);
	Tcomp.stop();
	return 0;
}

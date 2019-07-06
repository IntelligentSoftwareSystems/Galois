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
const char* desc = "Counts the edge-induced motifs in a graph using BFS expansion";
const char* url  = 0;
namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-motif(default value 3)"), cll::init(3));
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#define CHUNK_SIZE 256
#include "Mining/element.h"
typedef StructuralElement ElementType;
#include "Mining/embedding.h"
typedef EdgeEmbedding EmbeddingT;
typedef EdgeEmbeddingQueue EmbeddingQueueT;
#include "Mining/edge_miner.h"
#include "Mining/util.h"

void MotifSolver(EdgeMiner &miner) {
	if (show) std::cout << "=============================== Start ===============================\n";
	EmbeddingQueueT in_queue, out_queue; // in&out worklist. double buffering
	miner.init(in_queue); // initialize the worklist
	if(show) in_queue.printout_embeddings(0);
	unsigned level = 1;

	// a level-by-level approach for Apriori search space (breadth first seach)
	while (level < k) { // to get the same output as RStream (which is not complete)
		if (show) std::cout << "\n============================== Level " << level << " ==============================\n";
		miner.expand_edge(k, in_queue, out_queue); // edge expansion
		in_queue.swap(out_queue);
		out_queue.clear();
		if (show) in_queue.printout_embeddings(level);

		if (show) std::cout << "\n------------------------ Step 2: Aggregation ------------------------\n";
		// Sub-step 1: aggregate on quick patterns: gather embeddings into different quick patterns
		QpMapFreq qp_map; // quick patterns map for counting the frequency
		LocalQpMapFreq qp_localmap; // quick patterns local map for each thread
		galois::do_all(galois::iterate(in_queue),
			[&](const EmbeddingT& emb) {
				miner.quick_aggregate_each(emb, *(qp_localmap.getLocal())); // quick pattern aggregation
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("QuickAggregation")
		);
		miner.merge_qp_map(0, qp_localmap, qp_map);

		// Sub-step 2: aggregate on canonical patterns: gather quick patterns into different canonical patterns
		CgMapFreq cg_map; // canonical graph map for couting the frequency
		LocalCgMapFreq cg_localmap; // canonical graph local map for each thread
		galois::do_all(galois::iterate(qp_map),
			[&](std::pair<QPattern, Frequency> element) {
				miner.canonical_aggregate_each(element, *(cg_localmap.getLocal())); // canonical pattern aggregation
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("CanonicalAggregation")
		);
		miner.merge_cg_map(0, cg_localmap, cg_map);
		miner.printout_agg(cg_map);
		//queue_size = std::distance(in_queue.begin(), in_queue.end());
		//if (show) std::cout << "num_patterns: " << cg_map.size() << " num_quick_patterns: " << qp_map.size() << " num_embeddings: " << queue_size << "\n";
		level ++;
	}
	if (show) std::cout << "\n=============================== Done ===============================\n\n";
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

	EdgeMiner miner(&graph);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	MotifSolver(miner);
	Tcomp.stop();
	return 0;
}

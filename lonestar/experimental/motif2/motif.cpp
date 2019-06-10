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
const char* desc = "Counts the vertex-induced motifs in a graph using BFS traversal";
const char* url  = 0;
namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-motif(default value 0)"), cll::init(0));
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#define USE_SIMPLE
#define CHUNK_SIZE 256
#include "Mining/element.h"
typedef SimpleElement ElementType;
#include "Mining/embedding.h"
typedef VertexEmbedding EmbeddingT;
typedef VertexEmbeddingQueue EmbeddingQueueT;
#include "Mining/miner.h"
#include "Mining/util.h"

void MotifSolver(Miner &miner) {
	if (show) std::cout << "=============================== Start ===============================\n";
	EmbeddingQueueT queue, queue2; // task queues. double buffering
	miner.init(queue); // initialize the task queue
	int num_patterns[3] = {2, 6, 21};
	std::vector<UintAccu> accumulators(num_patterns[k-3]);
	for (int i = 0; i < num_patterns[k-3]; i++) accumulators[i].reset();
	if (show) queue.printout_embeddings(0);
	unsigned level = 1;
	while (level < k-2) {
		if (show) std::cout << "\n============================== Level " << level << " ==============================\n";
		if (show) std::cout << "\n------------------------- Step 1: Expanding -------------------------\n";
		// for each embedding in the task queue, do vertex-extension
		galois::for_each(galois::iterate(queue),
			[&](const EmbeddingT& emb, auto& ctx) {
				miner.extend_vertex_motif(emb, queue2); // vertex extension
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Expanding")
		);
		queue.swap(queue2);
		queue2.clear();
		if (show) queue.printout_embeddings(level);
		level ++;
	}
	if (show) std::cout << "\n------------------------ Step 2: Aggregation ------------------------\n";
	galois::for_each(galois::iterate(queue),
		[&](const EmbeddingT& emb, auto& ctx) {
			miner.aggregate_motif_each(emb, accumulators);
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(), galois::loopname("Reduce")
	);
	miner.printout_motifs(accumulators);
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
	assert(k > 2);
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");

	Miner miner(&graph);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	MotifSolver(miner);
	Tcomp.stop();
	return 0;
}

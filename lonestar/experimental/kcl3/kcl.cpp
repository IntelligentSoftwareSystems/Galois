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
#define USE_SIMPLE

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using BFS expansion";
const char* url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-clique (default value 3)"), cll::init(3));
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#define CHUNK_SIZE 256
#include "Mining/element.h"
typedef SimpleElement ElementType;
#include "Mining/embedding.h"
typedef BaseEmbedding EmbeddingT;
typedef BaseEmbeddingQueue EmbeddingQueueT;
#include "Mining/vertex_miner.h"
#include "Mining/util.h"

void KclSolver(VertexMiner &miner) {
	galois::GAccumulator<unsigned> total_num;
	total_num.reset();
	if(show) std::cout << "\n=============================== Start ===============================\n";
	EmbeddingQueueT queue, queue2;
	miner.init(queue); // insert single-edge (two-vertex) embeddings into the queue
	unsigned level = 1;
	while (1) {
		if(show) std::cout << "\n============================== Level " << level << " ==============================\n";
		if(show) queue.printout_embeddings(0);
		galois::do_all(galois::iterate(queue),
			[&](const EmbeddingT& emb) {
				miner.extend_vertex(emb, queue2, total_num, (level < k-2)); // expand one more vertex
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Expanding")
		);
		if (level == k-2) break; // if embedding size = k, done
		queue.swap(queue2);
		queue2.clean();
		level ++;
	}
	if(show) std::cout << "\n=============================== Done ================================\n";
	galois::gPrint("\n\ttotal_num_cliques = ", total_num.reduce(), "\n\n");
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	read_graph(graph, filetype, filename);
	Tinitial.stop();
	assert(k > 2);
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");

	VertexMiner miner(&graph);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	KclSolver(miner);
	Tcomp.stop();
	return 0;
}

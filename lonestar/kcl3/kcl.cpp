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
const char* desc = "Counts the K-Cliques in a graph using BFS traversal";
const char* url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-clique (default value 3)"), cll::init(3));
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Mining/miner.h"
#include "Mining/util.h"
#define CHUNK_SIZE 256

// insert edges into the worklist (task queue)
void initialization(Graph& graph, BaseEmbeddingQueue &queue) {
	if(show) printf("\n=============================== Init ===============================\n");
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
	if(show) std::cout << "\n=============================== Start ===============================\n";
	BaseEmbeddingQueue queue, queue2;
	initialization(graph, queue);
	if(show) printout_embeddings(0, miner, queue);
	unsigned level = 1;
	while (level < k-1) {
		if(show) std::cout << "\n============================== Level " << level << " ==============================\n";
		galois::for_each(
			galois::iterate(queue),
			[&](const BaseEmbedding& emb, auto& ctx) {
				miner.extend_vertex_clique(emb, queue2);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("ExtendVertex")
		);
		miner.update_base_embedding_size(); // increase the embedding size since one more edge added
		if(show) printout_embeddings(level, miner, queue2);
		queue.swap(queue2);
		queue2.clear();
		level ++;
	}
	if(show) std::cout << "\n=============================== Done ===============================\n";
	galois::gPrint("\n\ttotal_num_cliques = ", std::distance(queue.begin(), queue.end()), "\n\n");
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	read_graph(graph, filetype, filename);
	Tinitial.stop();

	unsigned sizeof_embedding = 2 * sizeof(SimpleElement);
	Miner miner(&graph, sizeof_embedding);
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");

	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	KclSolver(graph, miner);
	Tcomp.stop();
	return 0;
}

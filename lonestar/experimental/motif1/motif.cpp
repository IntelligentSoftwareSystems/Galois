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
const char* desc = "Counts the vertex-induced motifs in a graph using BFS extension";
const char* url  = 0;
namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-motif(default value 3)"), cll::init(3));
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#define USE_PID
#define USE_MAP
#define USE_WEDGE
#define USE_SIMPLE
#define USE_CUSTOM
#define USE_EMB_LIST
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "Mining/vertex_miner.h"
#include "Mining/util.h"
int num_patterns[3] = {2, 6, 21};
///*
class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np) : VertexMiner(g, size, np) {}
	~AppMiner() {}
	void print_output() { printout_motifs(); }
};

#include "Mining/engine.h"
//*/
/*
void solver(VertexMiner &miner, EmbeddingList &emb_list) {
	unsigned level = 1;
	while (1) {
		if(show) emb_list.printout_embeddings(level);
		miner.extend_vertex(level, emb_list);
		if (level == k-2) break; // if embedding size = k, done
		level ++;
	}
	if (k >= 5) {
		miner.merge_qp_map();
		miner.canonical_aggregate();
		miner.merge_cg_map();
	}
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

	assert(k > 2);
	ResourceManager rm;
	int npatterns = num_patterns[k-3];
	std::cout << k << "-motif has " << npatterns << " patterns in total\n";
	VertexMiner miner(&graph, k, npatterns);
	EmbeddingList emb_list;
	emb_list.init(graph, k);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	solver(miner, emb_list);
	Tcomp.stop();
	miner.printout_motifs();
	std::cout << "\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}
//*/

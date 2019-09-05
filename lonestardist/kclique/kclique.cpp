// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "kclique_cuda.h"
#include "galois/Timer.h"
#include "DistBenchStart.h"
#include "galois/DistGalois.h"
#include "llvm/Support/CommandLine.h"
#include <boost/iterator/transform_iterator.hpp>

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using BFS extension";
const char* url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
//static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-clique (default value 3)"), cll::init(3));
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
//typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
//typedef Graph::GraphNode GNode;

#define USE_DAG
#define USE_SIMPLE
#define USE_BASE_TYPES

#include "util.h"

#ifndef __GALOIS_HET_CUDA__
#define CHUNK_SIZE 256
typedef MGraph Graph;
#include "cpu_mining/vertex_miner.h"

void KclSolverCPU(VertexMiner &miner, EmbeddingList &emb_list) {
	unsigned level = 1;
	while (1) {
		miner.extend_vertex_clique(level, emb_list);
		if (level == k-2) break; 
		level ++;
	}
}
#endif

int main(int argc, char** argv) {
	galois::DistMemSys G;
	DistBenchStart(argc, argv, name, desc, url);
	//auto& net = galois::runtime::getSystemNetworkInterface();

	bool need_dag = false;
	#ifdef USE_DAG
	printf("Orientation enabled, using DAG\n");
	need_dag = true;
	#endif
	MGraph graph(need_dag);
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	read_graph(graph, filetype, inputFile);
	Tinitial.stop();
	assert(k > 2);
	std::cout << "num_vertices " << graph.num_vertices() 
		<< " num_edges " << graph.num_edges() << "\n";
	
	AccType total = 0;
	//ResourceManager rm;
#ifdef __GALOIS_HET_CUDA__
	KclInitGPU(graph, k);
#else
	VertexMiner miner(&graph, k);
	EmbeddingList emb_list;
	emb_list.init(graph, k, need_dag);
#endif
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
#ifdef __GALOIS_HET_CUDA__
	KclSolverGPU(k, total);
#else
	KclSolverCPU(miner, emb_list);
	total = miner.get_total_count();
#endif
	Tcomp.stop();
	std::cout << "\n\ttotal_num_cliques = " << total << "\n\n";
	//std::cout << "\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

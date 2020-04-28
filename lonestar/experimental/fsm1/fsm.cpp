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

#define USE_PID
#define USE_GSTL
#define USE_DOMAIN
#define ENABLE_LABEL
#define EDGE_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"
const char* name = "FSM";
const char* desc = "Frequent subgraph mining in a graph using BFS extension";
const char* url  = 0;
int total_num = 0;

void FsmSolver(EdgeMiner &miner, EmbeddingList& emb_list) {
	unsigned level = 1;
	if (debug) emb_list.printout_embeddings(1);
	int num_freq_patterns = miner.init_aggregator();
	total_num += num_freq_patterns;
	if (num_freq_patterns == 0) {
		std::cout << "No frequent pattern found\n\n";
		return;
	}
	std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns << "\n";
	miner.init_filter(emb_list);
	if (show) emb_list.printout_embeddings(level);

	while (1) {
		miner.extend_edge(level, emb_list);
		level ++;
		if (show) emb_list.printout_embeddings(level, debug);
		miner.quick_aggregate(level, emb_list);
		miner.merge_qp_map(level+1);
		miner.canonical_aggregate();
		miner.merge_cg_map(level+1);

		num_freq_patterns = miner.support_count();
		if (show) std::cout << "num_frequent_patterns: " << num_freq_patterns << "\n";
		if (debug) miner.printout_agg();

		total_num += num_freq_patterns;
		if (num_freq_patterns == 0) break;
		if (level == k) break;
		miner.filter(level, emb_list);
		if (show) emb_list.printout_embeddings(level, debug);
	}
	std::cout << "\n\tNumber of frequent patterns (minsup=" << minsup << "): " << total_num << "\n\n";
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinit("GraphReading");
	Tinit.start();
	read_graph(graph, filetype, filename);
	Tinit.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	//std::cout << "max_size = " << k << std::endl;
	//std::cout << "min_support = " << minsup << std::endl;
	//std::cout << "num_threads = " << numThreads << std::endl;

	assert(k > 1);
	ResourceManager rm;
	EdgeMiner miner(&graph);
	miner.set_threshold(minsup);
	EmbeddingList emb_list;
	emb_list.init(graph, k+1);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	FsmSolver(miner, emb_list);
	Tcomp.stop();
	std::cout << "\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

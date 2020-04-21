#define USE_DOMAIN
#define ENABLE_LABEL
#define EDGE_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining in a graph using BFS extension";
const char* url  = 0;
int total_num = 0;

void FsmSolver(EdgeMiner &miner) {
	unsigned level = 1;
	EmbeddingQueueType queue, filtered_queue;
	int num_freq_patterns = miner.init_aggregator();
	total_num += num_freq_patterns;
	if(num_freq_patterns == 0) {
		std::cout << "No frequent pattern found\n\n";
		return;
	}
	std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns << "\n";
	if (debug) miner.printout_agg();
	miner.init_filter(filtered_queue);

	// a level-by-level approach for Apriori search space (breadth first seach)
	while (1) {
		level ++;
		queue.clear();
		miner.extend_edge(filtered_queue, queue);
		if (show) queue.printout_embeddings(level, debug);

		// Sub-step 1: aggregate on quick patterns: gather embeddings into different quick patterns
		miner.quick_aggregate(queue);
		miner.merge_qp_map(level+1);
		// Sub-step 2: aggregate on canonical patterns: gather quick patterns into different canonical patterns
		miner.canonical_aggregate();
		miner.merge_cg_map(level+1);

		num_freq_patterns = miner.support_count();
		total_num += num_freq_patterns;
		if (debug) miner.printout_agg();
		if (num_freq_patterns == 0) break;
		if (level == k) break;

		filtered_queue.clear();
		miner.filter(queue, filtered_queue);
		if (show) filtered_queue.printout_embeddings(level, debug);
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
	std::cout << "num_vertices " << graph.size() << " num_edges " << graph.sizeEdges() << "\n";

	EdgeMiner miner(&graph);
	miner.set_max_size(k);
	miner.set_threshold(minsup);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	FsmSolver(miner);
	Tcomp.stop();
	return 0;
}

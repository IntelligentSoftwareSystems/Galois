#define USE_DAG
#define USE_DFS
#define ALGO_EDGE
#define USE_OPT
#define SHRINK // enables graph shrinking
#define USE_SIMPLE
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the WWW'18 paper:
// Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (using DAG and shrinking graph)";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g) : VertexMiner(g) {}
	~AppMiner() {}
	void init(unsigned max_degree, bool use_dag) {
		assert(k > 2);
		set_max_size(k);
		set_max_degree(max_degree);
		set_num_patterns(1);
		set_directed(use_dag);
		init_edgelist();
	}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}

	void edge_process_opt() {
		for (int i = 0; i < numThreads; i++) {
			emb_lists.getLocal(i)->allocate(graph, max_size, core);
		}
		std::cout << "DFS edge processing using advanced optimization\n";
		assert(max_size > 3);
		//galois::for_each(galois::iterate(edge_list), [&](const Edge &edge, auto &ctx) {
		//galois::do_all(galois::iterate(edge_list.begin(), edge_list.end()), [&](const Edge &edge) {
		galois::do_all(galois::iterate(edge_list), [&](const Edge &edge) {
			EmbeddingList *emb_list = emb_lists.getLocal();
			emb_list->init_edge(edge);
			dfs_extend(2, *emb_list);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("DfsEdgeSolver"));
		//}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(), galois::loopname("DfsEdgeSolver"));
	}
};

#include "DfsMining/engine.h"

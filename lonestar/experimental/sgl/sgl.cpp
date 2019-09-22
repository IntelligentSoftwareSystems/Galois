
//#define USE_DAG
#define USE_SIMPLE
#define ENABLE_STEAL
//#define USE_EMB_LIST
#define CHUNK_SIZE 256
#define USE_BASE_TYPES
#define USE_QUERY_GRAPH
#include "pangolin.h"

//#define CLR_BLACK 0
//#define CLR_GRAY 1
//#define CLR_WHITE 2
const char* name = "Sgl";
const char* desc = "Listing occurances of a given pattern in a graph using BFS extension";
const char* url  = 0;

class AppMiner : public VertexMiner {
protected:
	Graph *query_graph;

public:
	AppMiner(Graph *g, unsigned size, int np) : VertexMiner(g, size, np) {}
	AppMiner(Graph* dgraph, Graph* qgraph, unsigned size, int np) : VertexMiner(dgraph, size, np) {
		query_graph = qgraph;
		matching_order.resize(max_size);
		matching_order_map.resize(max_size);
		automorph_group_id.resize(max_size);
		//read_presets();
		for (size_t i = 0; i < max_size; ++i) {
			matching_order[i] = i;
			matching_order_map[i] = i;
			automorph_group_id[i] = 0;
		}
		for(int tid = 0; tid < numThreads; ++tid){
			Status *status = mt_status.getLocal();
			status->init(max_size);
		}
	}
	~AppMiner() {}
	// toExtend (only extend the last vertex in the embedding: fast)
	bool toExtend(unsigned n, const BaseEmbedding &emb, unsigned pos) {
		return true;//pos == n-1;
	}
	// toAdd (only add vertex that is connected to all the vertices in the embedding)
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		VertexId src = emb.get_vertex(pos);
		//std::cout << "\t n = " << n << ", src = " << src << ", dst = " << dst << ", pos = " << pos << "\n";
		//std::cout << ", deg(d) = " << get_degree(graph, dst) << ", deg(q) = " << get_degree(query_graph, pos+1);
		if (get_degree(graph, dst) < get_degree(query_graph, n)) return false;
		if (is_vertexInduced_automorphism<BaseEmbedding>(n, emb, pos, src, dst)) return false;
		//std::cout << "\tthis is canonical";
		// check the connectivity with previous vertices in the embedding
		for (auto e : query_graph->edges(n)) {
			VertexId query_prime = query_graph->getEdgeDst(e);
			//if (src == 2 && dst == 6) std::cout << "\t query_prime = " << query_prime;
			if (query_prime < n) {
				VertexId data_prime = emb.get_vertex(query_prime);
				//if (src == 2 && dst == 6) std::cout << "\t data_prime = " << data_prime;
				if (!is_connected(dst, data_prime)) return false;
			}
		}
		//std::cout << "\t extending with vertex " << dst << "\n";
		return true;
	}
	void print_output() {
		std::cout << "\n\ttotal_num_subgraphs = " << get_total_count() << "\n";
	}
};

unsigned get_degree(Graph &g, VertexId vid) {
	return std::distance(g.edge_begin(vid), g.edge_end(vid));
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph data_graph, query_graph;
	bool need_dag = false;
	#ifdef USE_DAG
	galois::gPrint("Orientation enabled, using DAG\n");
	need_dag = true;
	#endif
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	read_graph(data_graph, filetype, filename, false, need_dag);
	read_graph(query_graph, filetype, query_graph_filename, false, need_dag);
	Tinitial.stop();
	assert(k > 2);
	std::cout << "Data_graph: num_vertices " << data_graph.size() << " num_edges " << data_graph.sizeEdges() << "\n";
	std::cout << "Query_graph: num_vertices " << query_graph.size() << " num_edges " << query_graph.sizeEdges() << "\n";

	ResourceManager rm;
	AppMiner miner(&data_graph, &query_graph, query_graph.size(), 1);
	EmbeddingQueueType queue, queue2;
	for (size_t i = 0; i < data_graph.size(); ++i) {
		if(get_degree(data_graph, i) < get_degree(query_graph, 0)) continue;
		EmbeddingType emb;
		emb.push_back(i);
		queue.push_back(emb);
	}
	unsigned level = 1;
	while(1) {
		if (show) queue.printout_embeddings(level);
		miner.extend_vertex(queue, queue2);
		if (level == query_graph.size()-1) break; // if embedding size = k, done
		queue.swap(queue2);
		queue2.clear();
		level ++;
	}
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
}



#define USE_DFS
#define USE_SIMPLE
//#define USE_EMB_LIST
#define CHUNK_SIZE 256
#define USE_BASE_TYPES
#define USE_QUERY_GRAPH
#include "pangolin.h"

const char* name = "Sgl";
const char* desc = "Listing occurances of a given pattern in a graph using DFS extension";
const char* url  = 0;

class AppMiner : public VertexMiner {
protected:
	Graph *query_graph;

public:
	AppMiner(Graph *g, unsigned size, int np) : VertexMiner(g, size, np) {}
	AppMiner(Graph* dgraph, Graph* qgraph, unsigned size, int np) : VertexMiner(dgraph, size, np, false) {
		query_graph = qgraph;
		matching_order.resize(max_size);
		matching_order_map.resize(max_size);
		automorph_group_id.resize(max_size);
		read_presets();
	}
	~AppMiner() {}
	// toExtend (only extend the last vertex in the embedding: fast)
	bool toExtend(unsigned n, const BaseEmbedding &emb, unsigned pos) {
		return true;
		//return pos == n-1;
	}
	// toAdd (only add vertex that is connected to all the vertices in the embedding)
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		VertexId next_qnode = get_query_vertex(n); // using matching order to get query vertex id
		// the first vertex should always has the smallest id (if it is not special)
		if (!fv && dst <= emb.get_vertex(0)) return false;
		// if the degree is smaller than that of its corresponding query vertex
		if (get_degree(graph, dst) < get_degree(query_graph, next_qnode)) return false;
		// if this vertex already exists in the embedding
		for (unsigned i = 0; i < n; ++i) if (dst == emb.get_vertex(i)) return false;
		// check the connectivity with previous vertices in the embedding
		for (auto e : query_graph->edges(next_qnode)) {
			VertexId q_dst = query_graph->getEdgeDst(e);
			unsigned q_order = matching_order_map[q_dst];
			if (q_order < n && q_order != pos) {
				VertexId d_vertex = emb.get_vertex(q_order);
				//if (debug && n == 3 && pos == 1 && emb.get_vertex(pos) == 3 && dst == 5) std:: cout << "\t\t d_vedrtex = " << d_vertex << "\n";
				if (!is_connected(dst, d_vertex)) return false;
			}
		}
		// if it is not the canonical automorphism
		for (unsigned i = 0; i < n; ++i) {
			VertexId q_neighbor = get_query_vertex(i);
			if (automorph_group_id[q_neighbor] == automorph_group_id[next_qnode]) {
				VertexId dnode = emb.get_vertex(i);
				if ((q_neighbor < next_qnode && dnode > dst) ||
					(q_neighbor > next_qnode && dnode < dst))
					return false;
			}
		}
		if (debug) std::cout << "\t extending with vertex " << dst << "\n";
		return true;
	}
	void vertex_process_naive() {
		VertexId curr_qnode = get_query_vertex(0);
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const auto& src) {
				EmbeddingList *emb_list = emb_lists.getLocal();
				if (get_degree(graph, src) < get_degree(query_graph, curr_qnode)) return;
				emb_list->init_vertex(src);
				dfs_extend_base(1, 0, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("DfsVertexNaiveSolver")
		);
	}

	void edge_process_naive() {
		VertexId qnode0 = get_query_vertex(0);
		VertexId qnode1 = get_query_vertex(1);
		//galois::do_all(galois::iterate((size_t)0, edge_list.size()),
			//[&](const size_t& pos) {
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const auto& src) {
				if (get_degree(graph, src) < get_degree(query_graph, qnode0)) return;
				EmbeddingList *emb_list = emb_lists.getLocal();
				emb_list->init();
				emb_list->set_size(1, 0);
				for (auto e : graph->edges(src)) {
					auto dst = graph->getEdgeDst(e);
					if (fv || src < dst) { 
						//Edge edge = edge_list.get_edge(pos);
						//Edge edge(src, dst);
						//emb_list->init(edge);
						if (get_degree(graph, dst) < get_degree(query_graph, qnode1)) continue;
						auto start = emb_list->size(1);
						emb_list->set_vid(1, start, dst);
						emb_list->set_idx(1, start, src);
						emb_list->set_size(1, start+1);
						dfs_extend_base(1, start, *emb_list);
						//dfs_extend_base(1, *emb_list);
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("DfsEdgeNaiveSolver")
		);
	}
	inline void dfs_extend_base(unsigned level, unsigned pos, EmbeddingList &emb_list) {
		unsigned n = level + 1;
		BaseEmbedding emb(n);
		emb_list.get_embedding<BaseEmbedding>(level, pos, emb);
		if (debug) std::cout << "current embedding: " << emb << "\n";
		// get next query vertex
		auto next_qnode = get_query_vertex(n); // using matching order to get query vertex id
		if (n == max_size-1) {
			for (auto q_edge : query_graph->edges(next_qnode)) {
				VertexId q_dst = query_graph->getEdgeDst(q_edge);
				unsigned q_order = matching_order_map[q_dst]; // using query vertex id to get its matching order
				if (q_order < n) {
					auto d_vertex = emb.get_vertex(q_order);
					for (auto d_edge : graph->edges(d_vertex)) {
						auto d_dst = graph->getEdgeDst(d_edge);
						if (toAdd(n, emb, d_dst, q_order))
							total_num += 1; // if size = max_size, no need to add to the queue, just accumulate
					}
				}
			}
			return;
		}
		emb_list.set_size(level+1, 0);
		for (auto q_edge : query_graph->edges(next_qnode)) {
			VertexId q_dst = query_graph->getEdgeDst(q_edge);
			unsigned q_order = matching_order_map[q_dst]; // using query vertex id to get its matching order
			if (q_order < n) {
				auto d_vertex = emb.get_vertex(q_order);
				for (auto d_edge : graph->edges(d_vertex)) {
					auto d_dst = graph->getEdgeDst(d_edge);
					if (toAdd(n, emb, d_dst, q_order)) {
						auto start = emb_list.size(level+1);
						emb_list.set_vid(level+1, start, d_dst);
						emb_list.set_idx(level+1, start, pos);
						emb_list.set_size(level+1, start+1);
						dfs_extend_base(level+1, start, emb_list);
					}
					break;
				}
			}
		}
	}
	VertexId get_query_vertex(unsigned id) { return matching_order[id]; }
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
	EmbeddingList emb_list;

	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	miner.edge_process_naive();
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
}


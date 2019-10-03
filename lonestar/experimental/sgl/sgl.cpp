
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
		read_presets();
		/*
		for (size_t i = 0; i < max_size; ++i) {
			matching_order[i] = i;
			matching_order_map[i] = i;
			automorph_group_id[i] = 0;
		}
		//*/
		for(int tid = 0; tid < numThreads; ++tid){
			Status *status = mt_status.getLocal();
			status->init(max_size);
		}
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
		if (debug) {
			VertexId src = emb.get_vertex(pos);
			std::cout << "\t n = " << n << ", pos = " << pos << ", src = " << src << ", dst = " << dst << "\n";
		}
		//std::cout << ", deg(d) = " << get_degree(graph, dst) << ", deg(q) = " << get_degree(query_graph, pos+1);

		// the first vertex should always has the smallest id (if it is not special)
		if (!fv && dst <= emb.get_vertex(0)) return false;
		// if the degree is smaller than that of its corresponding query vertex
		if (get_degree(graph, dst) < get_degree(query_graph, next_qnode)) return false;
		// if this vertex already exists in the embedding
		for (unsigned i = 0; i < n; ++i) if (dst == emb.get_vertex(i)) return false;
		///*
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
		//*/
		// if it is a redundant candidate (a neighbor of previous vertex)
		//for (unsigned i = 0; i < pos; ++i)
		//	if (is_connected(emb.get_vertex(i), dst)) return false;
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
	inline void extend_vertex(BaseEmbeddingQueue &in_queue, BaseEmbeddingQueue &out_queue) {
		galois::do_all(galois::iterate(in_queue),
			[&](const BaseEmbedding& emb) {
				unsigned n = emb.size();
				if (debug) std::cout << "current embedding: " << emb << "\n";
				//for (unsigned i = 0; i < n; ++i) {
					//if(!toExtend(n, emb, i)) continue;
					//VertexId src = emb.get_vertex(i);
				// get next query vertex
				VertexId next_qnode = get_query_vertex(n); // using matching order to get query vertex id
				// for each neighbor of the next query vertex in the query graph
				for (auto q_edge : query_graph->edges(next_qnode)) {
					VertexId q_dst = query_graph->getEdgeDst(q_edge);
					unsigned q_order = matching_order_map[q_dst]; // using query vertex id to get its matching order
					// if the neighbor is aready visited
					if (q_order < n) {
						// get the matched data vertex
						VertexId d_vertex = emb.get_vertex(q_order);
						// each neighbor of d_vertex is a candidate
						for (auto d_edge : graph->edges(d_vertex)) {
							GNode d_dst = graph->getEdgeDst(d_edge);
							if (toAdd(n, emb, d_dst, q_order)) {
								if (n < max_size-1) { // generate a new embedding and add it to the next queue
									BaseEmbedding new_emb(emb);
									new_emb.push_back(d_dst);
									out_queue.push_back(new_emb);
								} else {
									if (show) {
										BaseEmbedding new_emb(emb);
										new_emb.push_back(d_dst);
										std::cout << "Found embedding: " << new_emb << "\n";
									}
									total_num += 1; // if size = max_size, no need to add to the queue, just accumulate
								}
							}
						}
						break;
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), 
			galois::steal(), 
			galois::loopname("Extending")
		);
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

	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	VertexId curr_qnode = miner.get_query_vertex(0);
	EmbeddingQueueType queue, queue2;
	for (size_t i = 0; i < data_graph.size(); ++i) {
		if(get_degree(data_graph, i) < get_degree(query_graph, curr_qnode)) continue;
		EmbeddingType emb;
		emb.push_back(i);
		queue.push_back(emb);
	}
	unsigned level = 1;
	while(1) {
		if (show) queue.printout_embeddings(level, debug);
		miner.extend_vertex(queue, queue2);
		if (level == query_graph.size()-1) break; // if embedding size = k, done
		queue.swap(queue2);
		queue2.clear();
		level ++;
	}
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
}


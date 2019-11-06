#include "mgraph.h"
#include "res_man.h"

void print_graph(Graph &graph) {
	for (GNode n : graph) {
		std::cout << "vertex " << n << ": label = " << graph.getData(n) << " edgelist = [ ";
		for (auto e : graph.edges(n))
			std::cout << graph.getEdgeDst(e) << " ";
		std::cout << "]" << std::endl;
	}
}

void genGraph(MGraph &mg, Graph &g) {
	g.allocateFrom(mg.num_vertices(), mg.num_edges());
	g.constructNodes();
	for (size_t i = 0; i < mg.num_vertices(); i++) {
		g.getData(i) = mg.get_label(i);
		auto row_begin = mg.get_offset(i);
		auto row_end = mg.get_offset(i+1);
		g.fixEndEdge(i, row_end);
		for (auto offset = row_begin; offset < row_end; offset ++) {
			#ifdef ENABLE_LABEL
				//g.constructEdge(offset, mg.get_dest(offset), mg.get_weight(offset));
				g.constructEdge(offset, mg.get_dest(offset), 1); // do not consider edge labels currently
			#else
				g.constructEdge(offset, mg.get_dest(offset), 0);
			#endif
		}
	}
}

// relabel is needed when we use DAG as input graph, and it is disabled when we use symmetrized graph
unsigned read_graph(Graph &graph, std::string filetype, std::string filename, bool need_dag = false) {
	MGraph mgraph(need_dag);
	unsigned max_degree = 0;
	if (filetype == "txt") {
		printf("Reading .lg file: %s\n", filename.c_str());
		mgraph.read_txt(filename.c_str());
		genGraph(mgraph, graph);
	} else if (filetype == "adj") {
		printf("Reading .adj file: %s\n", filename.c_str());
		mgraph.read_adj(filename.c_str());
		genGraph(mgraph, graph);
	} else if (filetype == "mtx") {
		printf("Reading .mtx file: %s\n", filename.c_str());
		mgraph.read_mtx(filename.c_str(), true); //symmetrize
		genGraph(mgraph, graph);
	} else if (filetype == "gr") {
		printf("Reading .gr file: %s\n", filename.c_str());
		if(need_dag) {
			Graph g_temp;
			galois::graphs::readGraph(g_temp, filename);
			max_degree = mgraph.orientation(g_temp, graph);
		} else {
			galois::graphs::readGraph(graph, filename);
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
				graph.getData(vid) = 1;
				//for (auto e : graph.edges(n)) graph.getEdgeData(e) = 1;
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("assignVertexLabels"));
			std::vector<unsigned> degrees(graph.size());
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
				degrees[vid] = std::distance(graph.edge_begin(vid), graph.edge_end(vid));
			}, galois::loopname("computeMaxDegree"));
			max_degree = *(std::max_element(degrees.begin(), degrees.end()));
		}
	} else { printf("Unkown file format\n"); exit(1); }
	//print_graph(graph);
	if (filetype != "gr") {
		max_degree = mgraph.get_max_degree();
		mgraph.clean();
	}
	printf("max degree = %u\n", max_degree);
	return max_degree;
}


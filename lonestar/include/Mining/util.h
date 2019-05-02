
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
	for (int i = 0; i < mg.num_vertices(); i++) {
		g.getData(i) = mg.get_label(i);
		int row_begin = mg.get_offset(i);
		int row_end = mg.get_offset(i+1);
		//int num_neighbors = mg.out_degree(i);
		g.fixEndEdge(i, row_end);
		for (int offset = row_begin; offset < row_end; offset ++) {
			#ifdef ENABLE_LABEL
				g.constructEdge(offset, mg.get_dest(offset), mg.get_weight(offset));
			#else
				g.constructEdge(offset, mg.get_dest(offset), 0);
			#endif
		}
	}
}
/*
// construct the edge-list for later use. May not be necessary if Galois has this support
void construct_edgelist(Graph& graph, std::vector<LabeledEdge> &edgelist) {
	for (Graph::iterator it = graph.begin(); it != graph.end(); it ++) {
		// for each vertex
		GNode src = *it;
		auto& src_label = graph.getData(src);
		Graph::edge_iterator first = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
		Graph::edge_iterator last = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
		// foe each edge of this vertex
		for (auto e = first; e != last; ++ e) {
			GNode dst = graph.getEdgeDst(e);
			auto& dst_label = graph.getData(dst);
			LabeledEdge edge(src, dst, src_label, dst_label);
			edgelist.push_back(edge);
		}
	}
	assert(edgelist.size() == graph.sizeEdges());
}
*/

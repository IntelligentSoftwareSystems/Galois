
	void edge_process() {
		std::cout << "num_edges in edge_list = " << edge_list.size() << "\n\n";
		galois::do_all(galois::iterate(edge_list.begin(), edge_list.end()),
			[&](const Edge &edge) {
				EmbeddingList *emb_list = emb_lists.getLocal();
				UintList *id_list = id_lists.getLocal();
				UintList *old_id_list = old_id_lists.getLocal();
				build_egonet_from_edge(edge, *egonet, *emb_list, *id_list, *old_id_list);
				if (max_size > 3) dfs_extend(max_size-2, *egonet, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("KclSolver")
		);
	}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	bool need_dag = false;
	#ifdef USE_DAG
	galois::gPrint("Orientation enabled, using DAG\n");
	need_dag = true;
	#endif
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	int core = read_graph(graph, filetype, filename, false, need_dag);
	//int core = read_graph(graph, filetype, filename, true);
	Tinitial.stop();
	assert(k > 2);
	std::cout << "num_vertices " << graph.size() << " num_edges " << graph.sizeEdges() << "\n";
	std::cout << "core = " << core << "\n";
	//print_graph(graph);

	ResourceManager rm;
	int npatterns = 1;
	#ifdef USE_MAP
	npatterns = num_patterns[k-3];
	#endif
	DfsMiner miner(&graph, core, k);
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	#ifdef ALGO_EDGE
	miner.edge_process();
	#else
	miner.vertex_process();
	#endif
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

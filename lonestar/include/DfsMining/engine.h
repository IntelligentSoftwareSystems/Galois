
int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	bool need_dag = false;
	#ifdef USE_DAG
	std::cout << "Orientation enabled, using DAG\n";
	need_dag = true;
	#endif
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	int core = read_graph(graph, filetype, filename, false, need_dag);
	//int core = read_graph(graph, filetype, filename, true);
	Tinitial.stop();
	#ifdef EDGE_INDUCED
	assert(k > 1);
	#else
	assert(k > 2);
	#endif
	std::cout << "num_vertices " << graph.size() << " num_edges " << graph.sizeEdges() << "\n";
	if (show) std::cout << "k = " << k << "\n";
	if (show) std::cout << "core = " << core << "\n";
	//print_graph(graph);

	ResourceManager rm;
	#ifdef EDGE_INDUCED
	AppMiner miner(&graph);
	#else
	int npatterns = 1;
	#ifdef USE_MAP
	npatterns = num_patterns[k-3];
	#endif
	AppMiner miner(&graph, k, npatterns, need_dag, core);
	#endif
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	#ifdef EDGE_INDUCED
	miner.process();
	#else
	#ifdef ALGO_EDGE
	#ifdef USE_EGONET
	miner.edge_process();
	#else
	miner.edge_process_naive();
	#endif
	#else
	miner.vertex_process();
	#endif
	#endif
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

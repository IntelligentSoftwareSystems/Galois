
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
	AppMiner miner(&graph, k);
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
	#ifdef USE_ADHOC
	miner.edge_process_adhoc();
	#else
	#ifdef USE_EGONET
	//miner.edge_process();
	miner.edge_process_ego();
	#else
	miner.edge_process_naive();
	#endif // USE_EGONET
	#endif // USE_ADHOC
	#else
	miner.vertex_process_adhoc();
	#endif // ALGO_EDGE
	#endif // EDGE_INDUCED
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

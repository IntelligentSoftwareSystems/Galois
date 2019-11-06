
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
	unsigned max_degree = read_graph(graph, filetype, filename, need_dag);
	Tinitial.stop();
	std::cout << "num_vertices " << graph.size() << " num_edges " << graph.sizeEdges() << "\n";
	if (show) std::cout << "k = " << k << "\n";
	if (show) std::cout << "max_degree = " << max_degree << "\n";
	//print_graph(graph);

	ResourceManager rm;
	AppMiner miner(&graph);
	miner.init(max_degree, need_dag);

	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	#ifdef EDGE_INDUCED
	miner.process();
	#else
	#ifdef ALGO_EDGE
	#ifdef USE_OPT
	miner.edge_process_opt();
	#else
	miner.edge_process();
	#endif // USE_OPT
	#else
	#ifdef USE_OPT
	miner.vertex_process_opt();
	#else
	miner.vertex_process();
	#endif // USE_OPT
	#endif // ALGO_EDGE
	#endif // EDGE_INDUCED
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

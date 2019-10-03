#ifndef TRIANGLE
#ifdef EDGE_INDUCED
void edgeSolver(AppMiner &miner, EmbeddingList& emb_list) {
	unsigned level = 1;
	if (debug) emb_list.printout_embeddings(1);
	int num_freq_patterns = miner.init_aggregator();
	if (num_freq_patterns == 0) {
		std::cout << "No frequent pattern found\n\n";
		return;
	}
	miner.inc_total_num(num_freq_patterns);
	std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns << "\n";
	miner.init_filter(emb_list);
	if (show) emb_list.printout_embeddings(level);

	while (1) {
		miner.extend_edge(level, emb_list);
		level ++;
		if (show) emb_list.printout_embeddings(level, debug);
		miner.quick_aggregate(level, emb_list);
		miner.merge_qp_map(level+1);
		miner.canonical_aggregate();
		miner.merge_cg_map(level+1);

		num_freq_patterns = miner.support_count();
		if (show) std::cout << "num_frequent_patterns: " << num_freq_patterns << "\n";
		if (debug) miner.printout_agg();

		miner.inc_total_num(num_freq_patterns);
		if (num_freq_patterns == 0) break;
		if (level == k) break;
		miner.filter(level, emb_list);
		if (show) emb_list.printout_embeddings(level, debug);
	}
}
#else
void solver(AppMiner &miner, EmbeddingList &emb_list) {
	unsigned level = 1;
	while (1) {
		if (show) emb_list.printout_embeddings(level);
		#ifdef USE_MAP
		miner.extend_vertex(level, emb_list);
		#else
		miner.extend_vertex_kcl(level, emb_list);
		#endif
		if (level == k-2) break; 
		level ++;
	}
}
#endif
#endif

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
	read_graph(graph, filetype, filename, false, need_dag);
	Tinitial.stop();
	#ifdef EDGE_INDUCED
	assert(k > 1);
	#else
	assert(k > 2);
	#endif
	std::cout << "num_vertices " << graph.size() << " num_edges " << graph.sizeEdges() << "\n";
	//std::cout << "max_size = " << k << std::endl;
	//std::cout << "min_support = " << minsup << std::endl;
	//std::cout << "num_threads = " << numThreads << std::endl;

	ResourceManager rm;

	#ifdef TRIANGLE
	AppMiner miner(&graph);
	#else
	#ifdef EDGE_INDUCED
	AppMiner miner(&graph, k);
	miner.set_threshold(minsup);
	#else
	int npatterns = 1;
	#ifdef USE_MAP
	npatterns = num_patterns[k-3];
	#endif
	AppMiner miner(&graph, k, npatterns);
	#endif
	#endif

	#ifdef USE_EMB_LIST
	EmbeddingList emb_list;
	galois::StatTimer Tinitemb("EmbListInitTime");
	Tinitemb.start();
	#ifdef TRIANGLE
	emb_list.init(graph, 2, need_dag);
	#else
	#ifdef EDGE_INDUCED
	emb_list.init(graph, k+1);
	#else
	emb_list.init(graph, k, need_dag);
	Tinitemb.stop();
	#endif
	#endif
	#else
	EmbeddingQueueType queue, queue2;
	miner.init(queue); // insert single-edge (two-vertex) embeddings into the queue
	if(show) queue.printout_embeddings(0);
	#endif

	galois::StatTimer Tcomp("Compute");
	Tcomp.start();

	#ifdef TRIANGLE
	#ifdef USE_EMB_LIST
	miner.tc_solver(emb_list);
	#else
	miner.tc_solver();
	#endif // USE_EMB_LIST
	#else
	#ifdef USE_EMB_LIST
	#ifdef EDGE_INDUCED
	edgeSolver(miner, emb_list);
	#else
	solver(miner, emb_list);
	#endif // EDGE_INDUCED
	#else
	unsigned level = 1;
	while (1) {
		miner.extend_vertex(queue, queue2); // extend one more vertex
		if (level == k-2) break; // if embedding size = k, done
		if (show) queue.printout_embeddings(level);
		queue.swap(queue2);
		queue2.clean();
		level ++;
	}
	#endif // USE_EMB_LIST
	#endif // TRIANGLE

	#ifdef USE_MAP
	if (k >= 5) {
		// TODO: need to use unsigned long for the counters
		miner.merge_qp_map();
		miner.canonical_reduce();
		miner.merge_cg_map();
	}
	#endif
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	//std::cout << "graph average degree: " << (double)graph.sizeEdges() / (double)graph.size() << "\n";
	return 0;
}

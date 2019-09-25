#ifndef TRIANGLE
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
	assert(k > 2);
	std::cout << "num_vertices " << graph.size() << " num_edges " << graph.sizeEdges() << "\n";

	ResourceManager rm;

	#ifdef TRIANGLE
	AppMiner miner(&graph);
	#else
	int npatterns = 1;
	#ifdef USE_MAP
	npatterns = num_patterns[k-3];
	#endif
	AppMiner miner(&graph, k, npatterns);
	#endif

	#ifdef TRIANGLE
	#ifdef USE_EMB_LIST
	EmbeddingList emb_list;
	galois::StatTimer Tinitemb("EmbListInitTime");
	Tinitemb.start();
	emb_list.init(graph, 2, need_dag);
	Tinitemb.stop();
	#endif
	#else
	#ifdef USE_EMB_LIST
	EmbeddingList emb_list;
	galois::StatTimer Tinitemb("EmbListInitTime");
	Tinitemb.start();
	emb_list.init(graph, k, need_dag);
	Tinitemb.stop();
	#else
	EmbeddingQueueType queue, queue2;
	miner.init(queue); // insert single-edge (two-vertex) embeddings into the queue
	if(show) queue.printout_embeddings(0);
	#endif
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
	solver(miner, emb_list);
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

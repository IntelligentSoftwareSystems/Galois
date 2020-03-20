#include "galois/Galois.h"
#include "res_man.h"
#include "embedding_list_dfs.h"

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarMineStart(argc, argv, name, desc, url);
	AppMiner miner(k,numThreads);
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	miner.read_graph(filetype, filename);
	Tinitial.stop();
	ResourceManager rm;
	miner.init_edgelist();
	miner.init_emb_list();
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	miner.solver();
	Tcomp.stop();
	miner.print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

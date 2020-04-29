#include "mgraph.h"

// relabel is needed when we use DAG as input graph, and it is disabled when we use symmetrized graph
int read_graph(MGraph &mgraph, std::string filetype, std::string filename) {
	if (filetype == "txt") {
		printf("Reading .lg file: %s\n", filename.c_str());
		mgraph.read_txt(filename.c_str());
	} else if (filetype == "adj") {
		printf("Reading .adj file: %s\n", filename.c_str());
		mgraph.read_adj(filename.c_str());
	} else if (filetype == "mtx") {
		printf("Reading .mtx file: %s\n", filename.c_str());
		mgraph.read_mtx(filename.c_str(), true); //symmetrize
	} else if (filetype == "gr") {
		printf("Reading .gr file: %s\n", filename.c_str());
		mgraph.read_gr(filename.c_str());
	} else { printf("Unkown file format\n"); exit(1); }
	return 0;
}


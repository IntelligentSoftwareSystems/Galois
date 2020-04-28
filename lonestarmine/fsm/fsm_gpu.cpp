// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#define EDGE_INDUCED
#include "fsm.h"
#include "lonestarmine.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining in an undirected graph";
const char* url  = 0;

int main(int argc, char** argv) {
	LonestarMineStart(argc, argv, name, desc, url);
	if (filetype != "adj") {
		std::cout << "Only support adj format for FSM\n";
		exit(1);
	}
	AccType total = 0;
	fsm_gpu_solver(filename, k, minsup, total);
	std::cout << "\n\ttotal_num_frequent_patterns = " << total << "\n\n";
	return 0;
}


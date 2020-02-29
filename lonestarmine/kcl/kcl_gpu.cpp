// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "kcl.h"
#include "../lonestarmine.h"

const char* name = "k-cliques";
const char* desc = "Listing all k-cliques in an undirected graph";
const char* url  = 0;

int main(int argc, char** argv) {
	LonestarMineStart(argc, argv, name, desc, url);
	if (filetype != "gr") {
		std::cout << "Only support gr format\n";
		exit(1);
	}
	AccType total = 0;
	kcl_gpu_solver(filename, k, total);
	std::cout << "\n\ttotal_num_cliques = " << total << "\n\n";
	return 0;
}


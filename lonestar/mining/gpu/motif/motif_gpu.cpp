// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "motif.h"
#include "lonestarmine.h"

const char* name           = "k-cliques";
const char* desc           = "Listing all k-cliques in an undirected graph";
const char* url            = 0;
static int num_patterns[3] = {2, 6, 21};

int main(int argc, char** argv) {
  LonestarMineStart(argc, argv, name, desc, url);
  if (filetype != "gr") {
    std::cout << "Only support gr format\n";
    exit(1);
  }
  int npatterns = num_patterns[k - 3];
  std::cout << k << "-motif has " << npatterns << " patterns in total\n";
  std::vector<AccType> accumulators(npatterns);
  for (int i = 0; i < npatterns; i++)
    accumulators[i] = 0;

  motif_gpu_solver(filename, k, accumulators);
  return 0;
}

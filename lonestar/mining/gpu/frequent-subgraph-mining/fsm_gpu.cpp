// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#define EDGE_INDUCED
#include "galois/Galois.h"
#include "fsm.h"
#include "lonestarmine.h"
#include "llvm/Support/CommandLine.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining in an undirected graph";
const char* url  = 0;

int main(int argc, char** argv) {
  LonestarMineStart(argc, argv, name, desc, url);

  if (!simpleGraph || !symmetricGraph) {
    GALOIS_DIE("This application requires a symmetric simple graph input "
               " which is symmetric and has no multiple edges or self-loops;"
               " please use both -symmetricGraph and -simpleGraph flag "
               " to indicate the input is a symmetric simple graph");
  }

  if (filetype != "adj") {
    galois::gError("This application only supports adj format for FSM\n"
                   "Please add the -ft=adj flag\n");
    exit(1);
  }
  AccType total = 0;
  fsm_gpu_solver(filename, k, minsup, total);
  std::cout << "\n\ttotal_num_frequent_patterns = " << total << "\n\n";
  return 0;
}

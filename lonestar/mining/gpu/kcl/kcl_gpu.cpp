// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "galois/Galois.h"
#include "kcl.h"
#include "lonestarmine.h"
#include "llvm/Support/CommandLine.h"

const char* name = "k-cliques";
const char* desc = "Listing all k-cliques in an undirected graph";
const char* url  = 0;

int main(int argc, char** argv) {
  LonestarMineStart(argc, argv, name, desc, url);

  if (!simpleGraph || !symmetricGraph) {
    GALOIS_DIE("This application requires a symmetric simple graph input "
               " which is symmetric and has no multiple edges or self-loops;"
               " please use both -symmetricGraph and -simpleGraph flag "
               " to indicate the input is a symmetric simple graph");
  }

  if (filetype != "gr") {
    galois::gError("This application only supports gr format\n"
                   "Please add the -ft=gr flag\n");
    exit(1);
  }

  AccType total = 0;
  kcl_gpu_solver(filename, k, total);
  std::cout << "\n\ttotal_num_cliques = " << total << "\n\n";
  return 0;
}

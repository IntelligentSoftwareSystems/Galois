/* -*- mode: C++ -*- */

#include "gg.h"

const char *prog_opts = "";
const char *prog_usage = "";
const char *prog_args_usage = "";

int process_prog_arg(int argc, char *argv[], int arg_start) {
   return 1;
}

void process_prog_opt(char c, char *optarg) {
}

void output(CSRGraphTy &g, const char *output_file) {
  FILE *f;

  if(!output_file)
    return;

  if(strcmp(output_file, "-") == 0)
    f = stdout;
  else
    f = fopen(output_file, "w");
    

  for(int i = 0; i < g.nnodes; i++) {
    check_fprintf(f, "%d %d\n", i, g.node_data[i]);
    
    for(int j = g.getFirstEdge(i); j < g.getFirstEdge(i+1); j++) {
      index_type dst = g.getAbsDestination(j);
      check_fprintf(f, "\te %d: %d %d %d\n", j, dst, g.edge_data[j], g.node_data[dst] );
    }
  }
}

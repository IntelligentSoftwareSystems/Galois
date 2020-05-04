/* -*- mode: C++ -*- */

#include "gg.h"

const char *prog_opts = "";
const char *prog_usage = "";
const char *prog_args_usage = "";

int process_prog_arg(int argc, char *argv[], int arg_start) {
   return 1;
}

void process_prog_opt(char c, char *optarg) {
  ;
}

void debug_output(CSRGraphTy &g, unsigned int *valid_edges) {
  for(int i = 0; i < g.nnodes; i++) {    
    int start = g.row_start[i];
    for(int j = 0; j < valid_edges[i]; j++) {
      printf("%d %d\n", i, g.edge_dst[start + j]);
    }    
  }
}

void output(CSRGraphTy &g, const char *output_file) {
  FILE *f;

  if(!output_file)
    return;

  if(strcmp(output_file, "-") == 0)
    f = stdout;
  else
    f = fopen(output_file, "w");
    

  for(int i = 0; i < g.nedges; i++)
    check_fprintf(f, "%d %d\n", i, g.edge_dst[i]);
}

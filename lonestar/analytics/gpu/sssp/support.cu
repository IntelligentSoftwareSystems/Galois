/* -*- mode: C++ -*- */

#include "gg.h"
#include <cassert>

const char *prog_opts = "ls:d:";
const char *prog_usage = "[-l] [-d delta] [-s startNode]";
const char *prog_args_usage = "-l: enable thread block load balancer (by default false)";

int DELTA = 10000;
extern const int INF;
int start_node = 0;
extern bool enable_lb;

int process_prog_arg(int argc, char *argv[], int arg_start) {
   return 1;
}

void process_prog_opt(char c, char *optarg) {
  if(c == 'd') {
    DELTA = atoi(optarg);
    assert(DELTA > 0);
  }
  if(c == 'l') { 
	enable_lb = true;
  }
  if(c == 's') {
     start_node = atoi(optarg);
     assert(start_node >= 0);
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
    
  for(int i = 0; i < g.nnodes; i++) {
    if(g.node_data[i] == INF) {
      check_fprintf(f, "%d INF\n", i);
    } else {
      check_fprintf(f, "%d %d\n", i, g.node_data[i]);
    }
  }

}

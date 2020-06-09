#include "gg.h"
const char *prog_opts = "";
const char *prog_usage = "";
const char *prog_args_usage = "";

extern AppendOnlyList el;
int process_prog_arg(int argc, char *argv[], int arg_start) {
   return 1;
}

void process_prog_opt(char c, char *optarg) { }

void output(CSRGraphTy &g, const char *output_file) {
  FILE *f;
  if(!output_file) return;
  if(strcmp(output_file, "-") == 0) f = stdout;
  else f = fopen(output_file, "w");
  el.sort();
  int *e = el.list.cpu_rd_ptr();
  int edges = el.nitems();
  for(int i = 0; i < edges; i++)
    check_fprintf(f, "%d %d %d %d\n", i, e[i], g.getAbsDestination(e[i]), g.getAbsWeight(e[i]));
}

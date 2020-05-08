/* -*- mode: C++ -*- */

#include "gg.h"
#include <float.h>
#include <stdint.h>

struct pr_value {
  index_type node;
  float rank;
  inline bool operator< (const pr_value& rhs) const {
    return rank < rhs.rank;
  }
};

/* TODO: accept ALPHA and EPSILON */
const char *prog_opts = "nt:x:";
const char *prog_usage = "[-n] [-t top_ranks] [-x max_iterations]";
const char *prog_args_usage = "";

extern float *P_CURR, *P_NEXT;
extern const float ALPHA, EPSILON;
extern int MAX_ITERATIONS;

int NO_PRINT_PAGERANK = 0;
int PRINT_TOP = 0;
int MAX_ITERATIONS =  INT_MAX;

int process_prog_arg(int argc, char *argv[], int arg_start) {
   return 1;
}

void process_prog_opt(char c, char *optarg) {
  if(c == 'n')
    NO_PRINT_PAGERANK = 1;

  if(c == 't') {
    PRINT_TOP = atoi(optarg);    
  }

  if(c == 'x') {
    MAX_ITERATIONS = atoi(optarg);
  }
}

void output(CSRGraphTy &g, const char *output_file) {
  FILE *f;

  struct pr_value * pr;

  pr = (struct pr_value *) calloc(g.nnodes, sizeof(struct pr_value));

  if(pr == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    exit(1);
  }

  fprintf(stderr, "Calculating sum ...\n");
  float sum = 0;
  for(int i = 0; i < g.nnodes; i++) {
    pr[i].node = i;
    pr[i].rank = P_CURR[i];
    sum += P_CURR[i];
  }

  fprintf(stdout, "sum: %f (%d)\n", sum, g.nnodes);

  if(!output_file)
    return;

//  fprintf(stderr, "Sorting by rank ...\n");
//  std::stable_sort(pr, pr + g.nnodes);
//  fprintf(stderr, "Writing to file ...\n");

  if(strcmp(output_file, "-") == 0)
    f = stdout;
  else
    f = fopen(output_file, "w");

//  check_fprintf(f, "ALPHA %*e EPSILON %*e\n", FLT_DIG, ALPHA, FLT_DIG, EPSILON);

  if(PRINT_TOP == 0)
    PRINT_TOP = g.nnodes;

//  check_fprintf(f, "RANKS 1--%d of %d\n", PRINT_TOP, g.nnodes);

  /* for(int i = 1; i <= PRINT_TOP; i++) {
    if(NO_PRINT_PAGERANK) 
      check_fprintf(f, "%d %d\n", i, pr[g.nnodes - i].node);
    else 
      check_fprintf(f, "%d %d %*e\n", i, pr[g.nnodes - i].node, FLT_DIG, pr[g.nnodes - i].rank/sum);  
  } */
  for(int i = 0; i < g.nnodes; i++) {
    if(NO_PRINT_PAGERANK) 
      check_fprintf(f, "%d\n", pr[i].node);
    else 
      check_fprintf(f, "%d %f\n", pr[i].node, FLT_DIG, pr[i].rank);  
  }

  free(pr);
}

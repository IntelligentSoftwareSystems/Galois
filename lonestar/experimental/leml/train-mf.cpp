/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unistd.h>

#include "mf.h"
#include "bilinear.h"

#define Malloc(type, n) (type*)malloc((n) * sizeof(type))
void exit_with_help() {
  printf(
      "Usage: train-mf [options] data-dir/\n"
      "options:\n"
      "    -s type : set type of solver (default 0)\n"
      "    	 0 -- L2R_LS (Squared Loss)\n"
      "    	 1 -- L2R_LR (Logistic Regression)\n"
      "    	 2 -- L2R_SVC (Squared Hinge Loss)\n"
      "    	 10 -- L2R_LS (Squared Loss) Fully observation\n"
      "    	 11 -- L2R_LS (Squared Loss) Fully observation with reweighting\n"
      "    -k rank : set the rank (default 10)\n"
      "    -n threads : set the number of threads (default 8)\n"
      "    -l lambda : set the regularization parameter lambda (default 0.1)\n"
      "    -t max_iter: set the number of iterations (default 10)\n"
      "    -T max_tron_iter: set the number of iterations used in TRON "
      "(default 5)\n"
      "    -g max_cg_iter: set the number of iterations used in CG (default "
      "20)\n"
      "    -e epsilon : set inner termination criterion epsilon of TRON "
      "(default 0.1)\n"
      "    -w reweighting: apply reweight (default 0)\n"
      //	"    -P top-p: set top-p accruacy (default 20)\n"
      //	"    -q show_predict: set top-p accruacy (default 1)\n"
  );
  exit(1);
}

mf_parameter parse_command_line(int argc, char** argv, char* Y_src,
                                char* X1_src, char* X2_src, char* Yt_src) {
  mf_parameter param; // default values have been set by the constructor
  int i;

  // parse options
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-')
      break;
    if (++i >= argc)
      exit_with_help();
    switch (argv[i - 1][1]) {
    case 's':
      param.solver_type = atoi(argv[i]);
      break;

    case 'k':
      param.k = atoi(argv[i]);
      break;

    case 'n':
      param.threads = atoi(argv[i]);
      break;

    case 'l':
      param.Cp = 1 / (atof(argv[i]));
      param.Cn = param.Cp;
      break;

    case 't':
      param.maxiter = atoi(argv[i]);
      break;

    case 'T':
      param.max_tron_iter = atoi(argv[i]);
      break;

    case 'g':
      param.max_cg_iter = atoi(argv[i]);
      break;

    case 'e':
      param.eps = atof(argv[i]);
      break;

    case 'r':
      param.lrate = atof(argv[i]);
      break;

    case 'w':
      param.reweighting = atoi(argv[i]);
      break;

    case 'P':
      param.top_p = atoi(argv[i]);
      break;

    case 'q':
      param.verbose = atoi(argv[i]);
      break;

    default:
      fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
      exit_with_help();
      break;
    }
  }

  // determine filenames
  if (i >= argc)
    exit_with_help();

  sprintf(Y_src, "%s/Y.smat", argv[i]);
  sprintf(X1_src, "%s/X1.smat", argv[i]);
  sprintf(X2_src, "%s/X2.smat", argv[i]);
  sprintf(Yt_src, "%s/Yt.smat", argv[i]);

  if (param.solver_type == L2R_BILINEAR_LS_FULL)
    param.reweighting = 0;

  return param;
}

/*
void rand_init_old(double *M, int m, int n, std::default_random_engine &gen) {
    std::normal_distribution<double> distribution (0.0,1.0);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            M[i*n+j] = distribution(gen);
}
*/

double* rand_init(double* M, int m, int n) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      M[i * n + j] = drand48();
  return M;
}

void run_mf_train(mf_parameter param, smat_t& Y, smat_t& X1, smat_t& X2,
                  smat_t& Yt) {
  srand48(0UL);
  //	std::default_random_engine generator(0);
  int k     = param.k;
  double *W = NULL, *H = NULL;
  if (param.reweighting)
    param.reweighting_based_on(Y);
  printf("Cp => %g, Cn => %g\n", param.Cp, param.Cn);

  long f1 = X1.cols, f2 = X2.cols;

  W = Malloc(double, k* f1);
  H = Malloc(double, k* f2);
  // rand_init(W, (int)X.cols, k, generator);
  // rand_init(H, (int)Y.cols, k, generator);
  rand_init(W, (int)f1, k);
  rand_init(H, (int)f2, k);

  omp_set_num_threads(param.threads);

  mf_problem prob(&Y, &X1, &X2, k, W, H);
  mf_train(&prob, &param, W, H);

  if (W)
    free(W);
  if (H)
    free(H);
}

int main(int argc, char* argv[]) {
  char X1_src[1024], X2_src[1024], Y_src[1024], Yt_src[1024];
  char hostname[1024];
  if (gethostname(hostname, 1024) != 0)
    puts("Cannot get the hostname!");
  else
    printf("Running on Host: %s\n", hostname);
  for (int i = 0; i < argc; i++)
    printf("%s ", argv[i]);
  puts("");
  smat_t Y, X1, X2, Yt;
  mf_parameter param =
      parse_command_line(argc, argv, Y_src, X1_src, X2_src, Yt_src);
  X1.load_from_binary(X1_src);
  X2.load_from_binary(X2_src);
  Y.load_from_binary(Y_src);
  Yt.load_from_binary(Yt_src);

  run_mf_train(param, Y, X1, X2, Yt);
  return 0;
}

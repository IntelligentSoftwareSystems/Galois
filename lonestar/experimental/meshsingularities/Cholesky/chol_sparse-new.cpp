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

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <sys/time.h>

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/CilkInit.h"
#include "galois/runtime/TreeExec.h"

#define CACHE_LINE_SIZE (64)

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

/* matrix operations */

typedef enum {
  // COL_IDX means column indexes and this means Compressed Row Storage (Rows
  // are in ptr)
  // ROW_IDX means row indexes and this leads to Compressed Column Storage
  COL_IDX,
  ROW_IDX
} idx;

typedef struct {
  // vector of values
  double* x;
  // non-zero elements
  int nz;
  // PTR and IDX values
  int* ptr;
  int* ind;
  // size of matrix
  int n;
  // Compressed Row Storage or Compressed Column Storage?
  idx indexing;
} sm;

typedef struct {
  int* parents;
  int** children;
  int* n_children;
} sm_util;

typedef struct {
  int row;
  int col;
  double val;
} in_data;

static llvm::cl::opt<std::string>
    infile("matrixfile", llvm::cl::desc("File with input matrix"),
           llvm::cl::init(""));

sm* alloc_matrix(int nz, int n, idx indexing) {
  sm* mat = (sm*)calloc(1, sizeof(sm));

  posix_memalign((void**)&mat->x, CACHE_LINE_SIZE, sizeof(double) * nz);
  posix_memalign((void**)&mat->ptr, CACHE_LINE_SIZE, sizeof(mat->ptr) * n);
  posix_memalign((void**)&mat->ind, CACHE_LINE_SIZE, sizeof(mat->ind) * nz);

  mat->n        = n;
  mat->indexing = indexing;
  mat->nz       = nz;

  if (mat == NULL || mat->x == NULL || mat->ptr == NULL || mat->ind == NULL) {
    free(mat);
    free(mat->x);
    free(mat->ptr);
    free(mat->ind);
    return NULL;
  }

  return mat;
}

void free_matrix(sm* mat) {
  free(mat->x);
  free(mat->ptr);
  free(mat->ind);
  free(mat);
}

void fill_matrix(sm* matrix, in_data** input_data, int nz) {
  int n      = 0;
  matrix->nz = nz;

  // we assume lower-triangular matrix on input!

  if (matrix->indexing == COL_IDX) {
    for (n = 0; n < nz; ++n) {
      matrix->x[n]   = input_data[n]->val;
      matrix->ind[n] = input_data[n]->col;
      if (n > 0) {
        if (input_data[n]->row != input_data[n - 1]->row) {
          matrix->ptr[input_data[n]->row] = n;
        }
      } else {
        matrix->ptr[input_data[n]->row] = 0;
      }
    }
    matrix->ptr[input_data[nz - 1]->row + 1] = nz;
  } else {
    for (n = 0; n < nz; ++n) {
      matrix->x[n]   = input_data[n]->val;
      matrix->ind[n] = input_data[n]->row;
      if (n > 0) {
        if (input_data[n]->col != input_data[n - 1]->col) {
          matrix->ptr[input_data[n]->col] = n;
        }
      } else {
        matrix->ptr[input_data[n]->col] = 0;
      }
    }
    matrix->ptr[input_data[nz - 1]->col + 1] = nz;
  }
}

/* elimination tree */

sm_util* elim_tree(const sm* A) {
  sm_util* smutil;
  int i, k, p, inext, *parents, *ancestor, *w;
  int n, *Ap, *Ai;

  smutil  = (sm_util*)calloc(1, sizeof(sm_util));
  parents = smutil->parents = (int*)calloc(A->n, sizeof(int));

  n  = A->n;
  Ap = A->ptr;
  Ai = A->ind;

  /* allocate result */
  w        = (int*)calloc(A->n, sizeof(int));
  ancestor = w;

  for (k = 0; k < n; k++) {
    parents[k]  = -1;
    ancestor[k] = -1;

    for (p = Ap[k]; p < Ap[k + 1]; ++p) {
      i = Ai[p];
      while (i != -1 && i < k) {
        inext       = ancestor[i];
        ancestor[i] = k;
        if (inext == -1)
          parents[i] = k;
        i = inext;
      }
    }
  }
  free(w);

  smutil->children   = (int**)calloc(n + 1, sizeof(int*));
  smutil->n_children = (int*)calloc(n + 1, sizeof(int));
  for (i = 0; i < n; ++i) {
    int parent = parents[i];
    parent     = (parent == -1) ? n : parent;
    ++smutil->n_children[parent];
    smutil->children[parent] =
        (int*)realloc((void*)smutil->children[parent],
                      smutil->n_children[parent] * sizeof(int));
    smutil->children[parent][smutil->n_children[parent] - 1] = i;
  }

  return (smutil);
}

void free_tree(int n, sm_util* smutil) {
  free(smutil->parents);
  for (int i = 1; i < n + 1; ++i) {
    //     free((void*) smutil->children[i]);
  }
  free((void*)smutil->children);
  free((void*)smutil->n_children);
}

int cs_ereach(const sm* A, int k, const sm_util* util, int* s, int* w) {
  int i, p, n, len, top, *Ap, *Ai;
  int* parent;

  if (util == NULL || s == NULL || w == NULL)
    return (-1);

  parent = util->parents;

  top = n = A->n;
  Ap      = A->ptr;
  Ai      = A->ind;
  w[k]    = ~w[k];

  for (p = Ap[k]; p < Ap[k + 1]; p++) {
    i = Ai[p];
    if (i > k)
      continue;
    for (len = 0; w[i] > 0; i = parent[i]) {
      s[len++] = i;
      w[i]     = ~w[i];
    }
    while (len > 0)
      s[--top] = s[--len]; /* push path onto stack */
  }

  for (p = top; p < n; p++)
    w[s[p]] = ~w[s[p]];
  w[k] = ~w[k];

  return (top);
}

void debug_matrix(sm* matrix) {
  int i;
  printf("x = [");
  for (i = 0; i < matrix->nz; ++i) {
    printf("%f ", matrix->x[i]);
  }
  printf("]\n");
  printf("idx = [");
  for (i = 0; i < matrix->nz; ++i) {
    printf("%d ", matrix->ind[i]);
  }
  printf("]\n");
  printf("ptr = [");
  for (i = 0; i < matrix->n + 1; ++i) {
    printf("%d ", matrix->ptr[i]);
  }
  printf("]\n");
}

void debug_tree(sm* matrix, sm_util* smutil) {
  int i;
  printf("[");
  for (i = 0; i < matrix->n; ++i) {
    printf("%d, ", smutil->parents[i]);
  }
  printf("]\n");
}

void print_data(in_data** data, int nz) {
  int i;
  for (i = 0; i < nz; ++i) {
    printf("%d %d %e\n", data[i]->row, data[i]->col, data[i]->val);
  }
}

void chol_worker(sm* mat, sm_util* util, int* Ai, int* Ap, double* Ax, int n,
                 int i, int* s, int* diag) {
  double x;

  int p;
  int p_max;
  int q;
  int col;

  int k;
  int j;

  p     = diag[i];
  p_max = Ap[i + 1];

  q = cs_ereach(mat, i, util, s, diag);

  if (q < 0) {
    fprintf(stderr, "cs_ereach returned: %d\n", q);
    abort();
  }

  while (q < n) {
    // s[q] - col number in row pattern
    col = s[q];
    // printf("Accessing column: %d while in %d\n", col, i);

    j         = diag[col];
    x         = Ax[j];
    int j_max = Ap[col + 1];

    if (j > Ap[col + 1]) {
      // fprintf(stderr, "w[%d] > ptr!\n", col);
      q++;
      continue;
    }

    k = p;
    while (j < j_max && k < p_max) {
      if (Ai[j] == Ai[k]) {
        Ax[k] -= x * Ax[j];
        ++k;
        ++j;
      }
      while (Ai[j] < Ai[k] && j < j_max) {
        ++j;
      }
      while (Ai[j] > Ai[k] && k < p_max) {
        ++k;
      }
    }
    ++q;
    ++diag[col];
  }

  x     = sqrt(Ax[p]);
  Ax[p] = x;
  for (j = p + 1; j < p_max; ++j) {
    Ax[j] /= x;
  }
  ++diag[i];
}

void chol(sm* mat, sm_util* util) {
  int i, j, k, ki;
  int *s, *diag;

  s    = (int*)calloc(mat->n, sizeof(int));
  diag = (int*)calloc(mat->n, sizeof(int));

  int* Ai    = mat->ind;
  double* Ax = mat->x;
  int* Ap    = mat->ptr;
  int n      = mat->n;

  for (i = 0; i < n; ++i) {
    diag[i] = Ap[i];
    while (Ai[diag[i]] != i) {
      ++diag[i];
    }
  }

  for (i = 0; i < mat->n; ++i) {
    chol_worker(mat, util, Ai, Ap, Ax, n, i, s, diag);
  }

  free(s);
  free(diag);
}

struct PerThread {
  int* s;
  int* diag;
  PerThread(int n, sm* A) {
    s    = (int*)calloc(n, sizeof(int));
    diag = (int*)calloc(n, sizeof(int));
    int *Ai, *Ap;
    Ai = A->ind;
    Ap = A->ptr;
    for (int i = 0; i < n; ++i) {
      diag[i] = Ap[i];
      while (Ai[diag[i]] != i && i < Ap[i + 1]) {
        ++diag[i];
      }
    }
  };

  ~PerThread() {
    free((void*)s);
    free((void*)diag);
  };
};

typedef galois::runtime::PerThreadStorage<PerThread> PTS;

struct TreeExecModel {
  typedef std::vector<int> ChildList;
  typedef std::vector<ChildList> Children;
  sm_util* util;
  sm* A;

  Children children;
  ChildList roots;

  PTS* pts;

  int rootlock = 1;

  TreeExecModel(PTS* pts, sm* A, sm_util* util)
      : pts(pts), A(A), util(util), children(A->n, ChildList()) {
    for (int i = 0; i < A->n; ++i) {
      if (util->parents[i] != -1) {
        children[util->parents[i]].push_back(i);
      } else {
        roots.push_back(i);
      }
    }
    int* Ap = A->ptr;
    int* Ai = A->ind;
  }

  struct GaloisDivide {
    TreeExecModel* tm;
    GaloisDivide(TreeExecModel* tm) : tm(tm) {}

    template <typename C>
    void operator()(int node, C& ctx) {
      for (int j : tm->children[node]) {
        ctx.spawn(j);
      }
      if (__sync_val_compare_and_swap(&tm->rootlock, 1, 0)) {
        for (int j : tm->roots) {
          ctx.spawn(j);
        }
      }
    }
  };

  struct GaloisConquer {
    TreeExecModel* tm;
    GaloisConquer(TreeExecModel* tm) : tm(tm) {}

    void operator()(int node) {
      chol_worker(tm->A, tm->util, tm->A->ind, tm->A->ptr, tm->A->x, tm->A->n,
                  node, tm->pts->getLocal()->s, tm->pts->getLocal()->diag);
    }
  };

  void run() {
    galois::runtime::for_each_ordered_tree(*(roots.begin()), GaloisDivide(this),
                                           GaloisConquer(this), "LeftCholesky");
  }
};

int main(int argc, char** argv) {
  int nz = 0;
  int n  = 0;
  int i, j;
  double val;

  struct timeval t1, t2;

  in_data** input_data;
  sm* matrix;

  LonestarStart(argc, argv, 0, 0, 0);
  galois::StatManager statManager;

  FILE* fp = fopen(infile.c_str(), "r");
  if (fp == NULL) {
    printf("Cannot open file: %s\n", infile.c_str());
    return 2;
  }

  while (fscanf(fp, "%u %u  %lg", &i, &j, &val) != EOF) {
    ++nz;
    n = j > n ? j : n;
    n = i > n ? i : n;
  }
  fseek(fp, 0, SEEK_SET);
  ++n;

  printf("Mat stat: nz=%u n=%u\n", nz, n);
  matrix = alloc_matrix(nz, n, ROW_IDX);

  input_data    = (in_data**)calloc(nz, sizeof(in_data*));
  input_data[0] = (in_data*)calloc(nz, sizeof(in_data));
  n             = 0;
  if (input_data == NULL || input_data[0] == NULL) {
    fprintf(stderr, "Cannot allocate memory.");
    abort();
  }

  while (fscanf(fp, "%u %u %lg", &(input_data[n]->row), &(input_data[n]->col),
                &(input_data[n]->val)) != EOF) {
    ++n;
    input_data[n] = input_data[0] + n;
  }
  fill_matrix(matrix, input_data, nz);

  free(input_data[0]);
  free(input_data);
  fclose(fp);

  // debug_matrix(matrix);

  printf("Creating tree...\n");
  sm_util* tree = elim_tree(matrix);
  printf("Solving...\n");

  galois::StatTimer T("NumericTime");
  PTS pts(matrix->n, matrix);
  TreeExecModel tm(&pts, matrix, tree);

  T.start();
  tm.run();
  // chol(matrix, tree);
  T.stop();

  // debug_matrix(matrix);

  free_matrix(matrix);
  free_tree(n, tree);
  return (0);
}

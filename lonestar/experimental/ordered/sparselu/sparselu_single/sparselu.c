/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de
 * Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify */
/*  it under the terms of the GNU General Public License as published by */
/*  the Free Software Foundation; either version 2 of the License, or */
/*  (at your option) any later version. */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful, */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*  GNU General Public License for more details. */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License */
/*  along with this program; if not, write to the Free Software */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
 * USA            */
/**********************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libgen.h>
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include "bots.h"
#include "sparselu.h"

extern char bots_arg_file[256];

/***********************************************************************
 * checkmat:
 **********************************************************************/
int checkmat(float* M, float* N) {
  int i, j;
  float r_err;

  for (i = 0; i < bots_arg_size_1; i++) {
    for (j = 0; j < bots_arg_size_1; j++) {
      r_err = M[i * bots_arg_size_1 + j] - N[i * bots_arg_size_1 + j];
      if (r_err == 0.0)
        continue;

      if (r_err < 0.0)
        r_err = -r_err;

      if (M[i * bots_arg_size_1 + j] == 0) {
        bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; \n", i, j,
                     M[i * bots_arg_size_1 + j], i, j,
                     N[i * bots_arg_size_1 + j]);
        return FALSE;
      }
      r_err = r_err / M[i * bots_arg_size_1 + j];
      if (r_err > EPSILON) {
        bots_message(
            "Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; Relative Error=%f\n",
            i, j, M[i * bots_arg_size_1 + j], i, j, N[i * bots_arg_size_1 + j],
            r_err);
        return FALSE;
      }
    }
  }
  return TRUE;
}
/***********************************************************************
 * genmat:
 **********************************************************************/
static void synthetic_genmat(float* M[]) {
  int null_entry, init_val, i, j, ii, jj;
  float* p;
  int a = 0, b = 0;

  init_val = 1325;

  /* generating the structure */
  for (ii = 0; ii < bots_arg_size; ii++) {
    for (jj = 0; jj < bots_arg_size; jj++) {
      /* computing null entries */
      null_entry = FALSE;
      if ((ii < jj) && (ii % 3 != 0))
        null_entry = TRUE;
      if ((ii > jj) && (jj % 3 != 0))
        null_entry = TRUE;
      if (ii % 2 == 1)
        null_entry = TRUE;
      if (jj % 2 == 1)
        null_entry = TRUE;
      if (ii == jj)
        null_entry = FALSE;
      if (ii == jj - 1)
        null_entry = FALSE;
      if (ii - 1 == jj)
        null_entry = FALSE;
      /* allocating matrix */
      if (null_entry == FALSE) {
        a++;
        M[ii * bots_arg_size + jj] =
            (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
        if ((M[ii * bots_arg_size + jj] == NULL)) {
          bots_message("Error: Out of memory\n");
          exit(101);
        }
        /* initializing matrix */
        p = M[ii * bots_arg_size + jj];
        for (i = 0; i < bots_arg_size_1; i++) {
          for (j = 0; j < bots_arg_size_1; j++) {
            init_val = (3125 * init_val) % 65536;
            (*p)     = (float)((init_val - 32768.0) / 16384.0);
            p++;
          }
        }
      } else {
        b++;
        M[ii * bots_arg_size + jj] = NULL;
      }
    }
  }
  bots_debug("allo = %d, no = %d, total = %d, factor = %f\n", a, b, a + b,
             (float)((float)a / (float)(a + b)));
}

static void structure_from_file_genmat(float* M[]) {
  int a, b, jj;
  int num_blocks, max_id;

  int fd = open(bots_arg_file, O_RDONLY);
  if (fd == -1)
    abort();
  struct stat buf;
  if (fstat(fd, &buf) == -1)
    abort();
  void* base       = mmap(NULL, buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  uint64_t* fptr   = (uint64_t*)base;
  uint64_t version = *fptr++;
  if (version != 1)
    abort();
  uint64_t sizeof_edge = *fptr++;
  if (sizeof_edge != 4)
    abort();
  uint64_t num_nodes = *fptr++;
  uint64_t num_edges = *fptr++;
  uint64_t* out_idx  = fptr;
  fptr += num_nodes;
  uint32_t* fptr32 = (uint32_t*)fptr;
  uint32_t* outs   = fptr32;
  fptr32 += num_edges;
  if (num_edges % 2)
    fptr32 += 1;
  float* edge_data = (float*)fptr32;

  memset(M, 0, bots_arg_size * bots_arg_size * sizeof(*M));

  num_blocks = (num_nodes + bots_arg_size_1 - 1) / bots_arg_size_1;
  max_id     = bots_arg_size_1 * bots_arg_size;

  printf("full size: %d\n", num_blocks);

  /* generating the structure */
  uint32_t ii;
  for (ii = 0; ii < num_nodes; ++ii) {
    if (ii >= max_id)
      break;
    int bii        = ii / bots_arg_size_1;
    uint64_t begin = (ii == 0) ? out_idx[0] : out_idx[ii - 1];
    uint64_t end   = out_idx[ii];
    uint64_t edge;
    for (edge = begin; edge < end; ++edge) {
      /* computing null entries */
      int jj = outs[edge];
      if (jj >= max_id)
        continue;
      int bjj = jj / bots_arg_size_1;
      if (M[bii * bots_arg_size + bjj] == NULL) {
        a++;
        M[bii * bots_arg_size + bjj] =
            (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
        memset(M[bii * bots_arg_size + bjj], 0,
               bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
      }
      if (M[bii * bots_arg_size + bjj] == NULL) {
        bots_message("Error: Out of memory\n");
        exit(101);
      }
      if (M[bjj * bots_arg_size + bii] == NULL) {
        a++;
        M[bjj * bots_arg_size + bii] =
            (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
        memset(M[bjj * bots_arg_size + bii], 0,
               bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
      }
      if (M[bjj * bots_arg_size + bii] == NULL) {
        bots_message("Error: Out of memory\n");
        exit(101);
      }
      M[bii * bots_arg_size + bjj][(ii % bots_arg_size_1) * bots_arg_size_1 +
                                   (jj % bots_arg_size_1)] = edge_data[edge];
      M[bjj * bots_arg_size + bii][(jj % bots_arg_size_1) * bots_arg_size_1 +
                                   (ii % bots_arg_size_1)] = edge_data[edge];
    }
  }
  // Add identity diagonal as necessary
  for (ii = 0; ii < bots_arg_size; ++ii) {
    if (M[ii * bots_arg_size + ii] == NULL) {
      a++;
      M[ii * bots_arg_size + ii] =
          (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
      memset(M[ii * bots_arg_size + ii], 0,
             bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
    }
    for (jj = 0; jj < bots_arg_size_1; ++jj) {
      if (M[ii * bots_arg_size + ii][jj * bots_arg_size_1 + jj] == 0.0)
        M[ii * bots_arg_size + ii][jj * bots_arg_size_1 + jj] = 1.0;
    }
  }
  b = num_blocks * num_blocks - a;
  bots_debug("allo = %d, no = %d, total = %d, factor = %f\n", a, b, a + b,
             (float)((float)a / (float)(a + b)));
}

void genmat(float* M[]) {

  if (strlen(bots_arg_file) == 0)
    synthetic_genmat(M);
  else
    structure_from_file_genmat(M);
}

/***********************************************************************
 * print_structure:
 **********************************************************************/
void print_structure(char* name, float* M[]) {
  int ii, jj;
  bots_message("Structure for matrix %s @ 0x%p\n", name, M);
  for (ii = 0; ii < bots_arg_size; ii++) {
    for (jj = 0; jj < bots_arg_size; jj++) {
      if (M[ii * bots_arg_size + jj] != NULL) {
        bots_message("x");
      } else
        bots_message(" ");
    }
    bots_message("\n");
  }
  bots_message("\n");
}
/***********************************************************************
 * allocate_clean_block:
 **********************************************************************/
float* allocate_clean_block() {
  int i, j;
  float *p, *q;

  p = (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
  q = p;
  if (p != NULL) {
    for (i = 0; i < bots_arg_size_1; i++)
      for (j = 0; j < bots_arg_size_1; j++) {
        (*p) = 0.0;
        p++;
      }

  } else {
    bots_message("Error: Out of memory\n");
    exit(101);
  }
  return (q);
}

/***********************************************************************
 * lu0:
 **********************************************************************/
void lu0(float* diag) {
  int i, j, k;

  for (k = 0; k < bots_arg_size_1; k++)
    for (i = k + 1; i < bots_arg_size_1; i++) {
      diag[i * bots_arg_size_1 + k] =
          diag[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
      for (j = k + 1; j < bots_arg_size_1; j++)
        diag[i * bots_arg_size_1 + j] =
            diag[i * bots_arg_size_1 + j] -
            diag[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
    }
}

/***********************************************************************
 * bdiv:
 **********************************************************************/
void bdiv(float* diag, float* row) {
  int i, j, k;
  for (i = 0; i < bots_arg_size_1; i++)
    for (k = 0; k < bots_arg_size_1; k++) {
      row[i * bots_arg_size_1 + k] =
          row[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
      for (j = k + 1; j < bots_arg_size_1; j++)
        row[i * bots_arg_size_1 + j] =
            row[i * bots_arg_size_1 + j] -
            row[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
    }
}
/***********************************************************************
 * bmod:
 **********************************************************************/
void bmod(float* row, float* col, float* inner) {
  int i, j, k;
  for (i = 0; i < bots_arg_size_1; i++)
    for (j = 0; j < bots_arg_size_1; j++)
      for (k = 0; k < bots_arg_size_1; k++)
        inner[i * bots_arg_size_1 + j] =
            inner[i * bots_arg_size_1 + j] -
            row[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}
/***********************************************************************
 * fwd:
 **********************************************************************/
void fwd(float* diag, float* col) {
  int i, j, k;
  for (j = 0; j < bots_arg_size_1; j++)
    for (k = 0; k < bots_arg_size_1; k++)
      for (i = k + 1; i < bots_arg_size_1; i++)
        col[i * bots_arg_size_1 + j] =
            col[i * bots_arg_size_1 + j] -
            diag[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}

void sparselu_init(float*** pBENCH, char* pass) {
  *pBENCH = (float**)malloc(bots_arg_size * bots_arg_size * sizeof(float*));
  genmat(*pBENCH);
  print_structure(pass, *pBENCH);
}

void sparselu_par_call(float** BENCH) {
  int ii, jj, kk;

  bots_message(
      "Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
      bots_arg_size, bots_arg_size, bots_arg_size_1, bots_arg_size_1);
#pragma omp parallel
#pragma omp single nowait
#pragma omp task untied
  for (kk = 0; kk < bots_arg_size; kk++) {
    lu0(BENCH[kk * bots_arg_size + kk]);
    for (jj = kk + 1; jj < bots_arg_size; jj++)
      if (BENCH[kk * bots_arg_size + jj] != NULL)
#pragma omp task untied firstprivate(kk, jj) shared(BENCH)
      {
        fwd(BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj]);
      }
    for (ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] != NULL)
#pragma omp task untied firstprivate(kk, ii) shared(BENCH)
      {
        bdiv(BENCH[kk * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk]);
      }

#pragma omp taskwait

    for (ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] != NULL)
        for (jj = kk + 1; jj < bots_arg_size; jj++)
          if (BENCH[kk * bots_arg_size + jj] != NULL)
#pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
          {
            if (BENCH[ii * bots_arg_size + jj] == NULL)
              BENCH[ii * bots_arg_size + jj] = allocate_clean_block();
            bmod(BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj],
                 BENCH[ii * bots_arg_size + jj]);
          }

#pragma omp taskwait
  }
  bots_message(" completed!\n");
}

void sparselu_seq_call(float** BENCH) {
  int ii, jj, kk;

  for (kk = 0; kk < bots_arg_size; kk++) {
    lu0(BENCH[kk * bots_arg_size + kk]);
    for (jj = kk + 1; jj < bots_arg_size; jj++)
      if (BENCH[kk * bots_arg_size + jj] != NULL) {
        fwd(BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj]);
      }
    for (ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] != NULL) {
        bdiv(BENCH[kk * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk]);
      }
    for (ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] != NULL)
        for (jj = kk + 1; jj < bots_arg_size; jj++)
          if (BENCH[kk * bots_arg_size + jj] != NULL) {
            if (BENCH[ii * bots_arg_size + jj] == NULL)
              BENCH[ii * bots_arg_size + jj] = allocate_clean_block();
            bmod(BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj],
                 BENCH[ii * bots_arg_size + jj]);
          }
  }
}

void sparselu_fini(float** BENCH, char* pass) { print_structure(pass, BENCH); }

int sparselu_check(float** SEQ, float** BENCH) {
  int ii, jj, ok = 1;

  for (ii = 0; ((ii < bots_arg_size) && ok); ii++) {
    for (jj = 0; ((jj < bots_arg_size) && ok); jj++) {
      if ((SEQ[ii * bots_arg_size + jj] == NULL) &&
          (BENCH[ii * bots_arg_size + jj] != NULL))
        ok = FALSE;
      if ((SEQ[ii * bots_arg_size + jj] != NULL) &&
          (BENCH[ii * bots_arg_size + jj] == NULL))
        ok = FALSE;
      if ((SEQ[ii * bots_arg_size + jj] != NULL) &&
          (BENCH[ii * bots_arg_size + jj] != NULL))
        ok = checkmat(SEQ[ii * bots_arg_size + jj],
                      BENCH[ii * bots_arg_size + jj]);
    }
  }
  if (ok)
    return BOTS_RESULT_SUCCESSFUL;
  else
    return BOTS_RESULT_UNSUCCESSFUL;
}

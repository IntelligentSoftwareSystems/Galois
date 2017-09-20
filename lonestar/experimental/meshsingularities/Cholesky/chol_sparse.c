#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <sys/time.h>
/*
#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/Bag.h"
#include "galois/CilkInit.h"
#include "galois/Statistics.h"
#include "galois/runtime/TreeExec.h"
*/

#define CACHE_LINE_SIZE (64)

/* matrix operations */

typedef enum {
    COL_IDX,
    ROW_IDX
} idx;

typedef struct
{
  // vector of values
  double *x;
  // not zero elements
  unsigned int nz;
  // PTR and IDX values
  int *ptr;
  int *ind;
  // size of matrix
  unsigned int n;
  // Compressed Row Storage or Compressed Column Storage?
  idx indexing;
} sm;

typedef struct
{
    int *children;
    int *flying_ptrs;
} sm_util;

typedef struct
{
    unsigned int row;
    unsigned int col;
    double val;
} in_data;

sm * alloc_matrix(int nz, int n, idx indexing)
{
    sm *mat = (sm*) calloc(1, sizeof(sm));
 
    posix_memalign((void**) &mat->x, CACHE_LINE_SIZE, sizeof(double)*nz);
    posix_memalign((void**) &mat->ptr, CACHE_LINE_SIZE, sizeof(mat->ptr)*(n+1));
    posix_memalign((void**) &mat->ind, CACHE_LINE_SIZE, sizeof(mat->ind)*nz);

    mat->n = n;
    mat->indexing = indexing;
    mat->nz = nz;
    
    if (!mat || !mat->x || !mat->ptr || !mat->ind) {
        free(mat);
        free(mat->x);
        free(mat->ptr);
        free(mat->ind);
        return NULL;
    }

    return mat;
}

void free_matrix(sm *mat)
{
    free(mat->x);
    free(mat->ptr);
    free(mat->ind);
    free(mat);
}

void fill_matrix(sm *matrix, in_data ** input_data, int nz)
{
    int n = 0;
    matrix->nz = nz;
    
    // we assume lower-triangular matrix on input!
    
    if (matrix->indexing == COL_IDX) {
        for (n=0; n<nz; ++n) {
            matrix->x[n] = input_data[n]->val;
            matrix->ind[n] = input_data[n]->col;
            if (n>0) {
                if (input_data[n]->row != input_data[n-1]->row) {
                    matrix->ptr[input_data[n]->row] = n;
                }
            } else {
                matrix->ptr[input_data[n]->row] = 0;
            }
        }
        matrix->ptr[input_data[nz-1]->row+1] = nz;
    } else {
        for (n=0; n<nz; ++n) {
            matrix->x[n] = input_data[n]->val;
            matrix->ind[n] = input_data[n]->row;
            if (n>0) {
                if (input_data[n]->col != input_data[n-1]->col) {
                    matrix->ptr[input_data[n]->col] = n;
                }
            } else {
                matrix->ptr[input_data[n]->col] = 0;
            }
        }
        matrix->ptr[input_data[nz-1]->col+1] = nz;
    }
}

/* elimination tree */

sm_util *elim_tree (const sm *A)
{
    sm_util *smutil = malloc(sizeof(sm_util));
    int i, k, p, inext,  *children, *flying_ptrs, *ancestor, *prev, *w;
    unsigned n, *Ap, *Ai;

    children = smutil->children = (int *) calloc (n, sizeof (int)) ;
    flying_ptrs = smutil->flying_ptrs = (int*) calloc(n, sizeof(int));


    n = A->n;
    Ap = A->ptr;
    Ai = A->ind;

    /* allocate result */
    w = (int *) calloc(n, sizeof (int)) ;
    ancestor = w ;
    prev = w + n ;
/*
    for (k = 0 ; k < n ; k++) {
        flying_ptrs[k] = Ap[k];
        children [k] = -1 ;
        ancestor [k] = -1 ;
        for (p = Ap [k] ; p < Ap [k+1] ; p++) {
            i = Ai [p] ;
            for ( ; i != -1 && i < k ; i = inext) {
                inext = ancestor [i] ;
                ancestor [i] = k ;
                if (inext == -1) children [i] = k ;

            }
        }
    }
*/
    free(w);

    n = A->n;
    Ap = A->ptr;
    Ai = A->ind;

    children = smutil->children = (int *) calloc (n, sizeof (int)) ;
    flying_ptrs = smutil->flying_ptrs = (int*) calloc(n, sizeof(int));

    for (k=0; k<n; ++k) {
        children[k] = -1;
        flying_ptrs[k] = Ap[k];
    }

    for (k = 0 ; k < n ; ++k) {
        if (Ap[k+1]-Ap[k] != 1) {
            for (p = Ap[k+1]-2; p >= Ap [k]; --p) {
                i = Ai [p] ;
                if (children[i] == -1) {
                    children[i] = k;
                }
            }
        }
    }
    return smutil;
}

int bin_search(sm *mat, unsigned int r, unsigned int c)
{
    int imin, imax;
    int ikey = mat->indexing == COL_IDX ? c : r;

    if (mat->indexing == COL_IDX) {
        imin = mat->ptr[r];
        imax = mat->ptr[r+1];
    } else {
        imin = mat->ptr[c];
        imax = mat->ptr[c+1];
    }

    while (imax >= imin) {
        int mid = (imax+imin)/2;
        if (mat->ind[mid] == ikey) {
            return mid;
        }
        else if (mat->ind[mid] < ikey) {
            imin = mid + 1;
        }
        else {
            imax = mid-1;
        }
    }
    return -1;
}

void usage(char *progname)
{
	printf("Usage: %s <data file>\n", progname);
}

double get_time(struct timeval *t1, struct timeval *t2)
{
    return (t2->tv_sec - t1->tv_sec) + (t2->tv_usec - t1->tv_usec)/1000000.0;
}

void print_time(struct timeval *t1, struct timeval *t2)
{
    printf("%lf [s]\n", get_time(t1, t2));
}

int indata_cmp_row(const void *p1, const void *p2)
{
    in_data *d1 = (in_data*) p1;
    in_data *d2 = (in_data*) p2;
    if (d1->row < d2->row) return -1;
    if (d1->row > d2->row) return 1;
    return d1->col - d2->col;
}

int indata_cmp_col(const void *p1, const void *p2)
{
    in_data *d1 = (in_data*) p1;
    in_data *d2 = (in_data*) p2;
    if (d1->col < d2->col) return -1;
    if (d1->col > d2->col) return 1;
    return d1->row - d2->row;
}

void debug_matrix(sm *matrix)
{
    int i;
    printf("x = [");
    for (i=0;i<matrix->nz; ++i) {
        printf("%f ", matrix->x[i]);
    }
    printf("]\n");
    printf("idx = [");
    for (i=0;i<matrix->nz; ++i) {
        printf("%d ", matrix->ind[i]);
    }
    printf("]\n");
    printf("ptr = [");
    for (i=0;i<matrix->n+1; ++i) {
        printf("%d ", matrix->ptr[i]);
    }
    printf("]\n");

}

void children_count(sm *matrix, sm_util *etree)
{
    int i = 0;
    int max=-1;
    int *w = (int*) calloc(matrix->n, sizeof(int));

    if (w == NULL)
        return;

    for (i=0; i<matrix->n; ++i) {
        if (etree->children[i] >= 0) {
            w[etree->children[i]]++;
            if (w[etree->children[i]] > max) {
                max = w[etree->children[i]];
            }
        }
    }

    printf("Max: %d\n", max);

    free(w);
}

void print_data(in_data ** data, int nz)
{
    int i;
    for (i=0; i<nz; ++i) {
        printf("%d %d %e\n", data[i]->row, data[i]->col, data[i]->val);
    }
}

void chol(sm *mat, sm *mat_col)
{
    int i, j, k, ki;

    for (i=0; i<mat->n; ++i) {

        double x;
        double y;
        // assuming we have lower-triangular matrix L we can do this:
        int p = mat->ptr[i+1]-1;
        int q;
        
        x = mat->x[p];
        for (j=mat->ptr[i]; j<p; ++j) {
            x -= mat->x[j]*mat->x[j];
        }
        mat->x[p] = sqrt(x);

        for (j=mat_col->ptr[i]+1; j<mat_col->ptr[i+1]; ++j) {
            // current hotspot - Tim Davies has answer :)
            q = bin_search(mat, mat_col->ind[j], i);

            y = mat->x[q];
            for (k=mat->ptr[mat_col->ind[j]], ki=mat->ptr[i];
                 k<mat->ptr[mat_col->ind[j]+1] && mat->ind[k] < i;
                 ++k) {
                // below we perform simple incrementation, maybe we should perform binary search?
                while (mat->ind[ki] < mat->ind[k] && ki<mat->ptr[i+1]) ++ki;
                if (mat->ind[ki] == mat->ind[k])
                    y -= mat->x[k] * mat->x[ki];
            }
            mat->x[q] = y/mat->x[p];
        }
    }
}

int chol_new(sm *mat, sm_util *util)
{
    int *children = util->children;
    int *flying_ptrs = util->flying_ptrs;
    int i, j, k;

    for (i=0; i<mat->n; ++i) {
        int p = mat->ptr[i+1]-1;
        double x = mat->x[p];
        for (j=mat->ptr[i]; j<p; ++j) {
            // L_{i,i} = sqrt(L_{i,i} - sum_{j=0}^{i-1} L_{i,j})
            x -= mat->x[j]*mat->x[j];
        }
        mat->x[p] = sqrt(x);

        for (j=children[i]; j!=-1; j=children[j]) {

            while (mat->ind[flying_ptrs[j]] < i) {
                ++(flying_ptrs[j]);
            }

            //int max_ki = bin_search(mat, j, i);
            int max_ki = flying_ptrs[j];
            if (mat->ind[max_ki] == i) {

                int ki = mat->ptr[j];
                double y = mat->x[max_ki];
                int max_k = mat->ptr[i+1]-2;

                for (k=mat->ptr[i]; k < max_k; ++k) {
                    while (ki < max_ki && mat->ind[ki] < mat->ind[k]) ++ki;

                    if (mat->ind[ki] == mat->ind[k]) {
                        y -= mat->x[ki] * mat->x[k];
                    }

                }
                mat->x[max_ki] = y/mat->x[p];
            }
        }
    }

    return 0;
}

int main(int argc, char ** argv)
{
    unsigned int nz = 0;
    unsigned int n = 0;
    unsigned int i, j;
    double val;

    int old = 0;

    struct timeval t1, t2;

    in_data ** input_data;
    sm *matrix;
    sm *matrix_col;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    if (argc == 3) {
        old = 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        usage(argv[0]);
  	    return 2;
    }

    while (fscanf(fp, "%u %u  %lg", &i, &j, &val) != EOF) {
        ++nz;
        n = j > n ? j : n;
        n = i > n ? i : n;
    }
    fseek(fp, 0, SEEK_SET);
    ++n;;
    
    printf("Mat stat: nz=%u n=%u\n", nz, n);
    matrix = alloc_matrix(nz, n, COL_IDX);
    if (old)
        matrix_col = alloc_matrix(nz, n, ROW_IDX);

    input_data = (in_data**) calloc(nz, sizeof(in_data*));
    input_data[0] = (in_data*) calloc(nz, sizeof(in_data));
    n = 0;
    while (fscanf(fp, "%u %u %lg", &(input_data[n]->row), &(input_data[n]->col), &(input_data[n]->val)) != EOF) {
        ++n;
        input_data[n] = input_data[0]+n;
    }

    if (old) {
        qsort((void*) input_data[0], nz, sizeof(in_data), indata_cmp_col);
        fill_matrix(matrix_col, input_data, nz);
    }
    if (matrix->indexing == COL_IDX)
        qsort((void*) input_data[0], nz, sizeof(in_data), indata_cmp_row);
    else
        qsort((void*) input_data[0], nz, sizeof(in_data), indata_cmp_col);

    fill_matrix(matrix, input_data, nz);

    free(input_data[0]);
    free(input_data);
    fclose(fp);

    //debug_matrix(matrix);

    printf("Creating tree...\n");
    sm_util *tree = elim_tree(matrix);
    
    children_count(matrix, tree);
    
    printf("Solving...\n");

    if (!old) {
        gettimeofday(&t1, NULL);
        chol_new(matrix, tree);
        gettimeofday(&t2, NULL);
        printf("SOLVE time: ");
        print_time(&t1, &t2);
    }

    if (old) {
        gettimeofday(&t1, NULL);
        chol(matrix, matrix_col);
        gettimeofday(&t2, NULL);
        printf("SOLVE time: ");
        print_time(&t1, &t2);
    }
    
    free_matrix(matrix);

    if (old)
        free_matrix(matrix_col);
    return 0;

}

#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cstddef>

#include "dmat.h"



// ---------------------- Dense Matrix Mulitplication ----------------------
extern "C" {
//int dposv_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
int dposv_(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, ptrdiff_t *info);
double dnrm2_(ptrdiff_t *, double *, ptrdiff_t *);
double ddot_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
ptrdiff_t daxpy_(ptrdiff_t *, double *, double *, ptrdiff_t *, double *, ptrdiff_t *);
ptrdiff_t dscal_(ptrdiff_t *, double *, double *, ptrdiff_t *);
ptrdiff_t dcopy_(ptrdiff_t *n, double *sx, ptrdiff_t *incx, double *sy, ptrdiff_t *incy);

void dgemm_(
    char *transa, char *transb,
    ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k,
    double *alpha, double *a, ptrdiff_t *lda,
    double *b, ptrdiff_t *ldb,
    double *beta, double *c, ptrdiff_t *ldc
);
}

double do_dot_product(double *x, double *y, size_t size) {
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	return ddot_(&len, x, &inc, y, &inc);
}
// y = alpha*x + y
void do_axpy(double alpha, double *x, double *y, size_t size) {
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	daxpy_(&len, &alpha, x, &inc, y, &inc);
}
void do_copy(double *x, double *y, size_t size){
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	dcopy_(&len, x, &inc, y, &inc);
}
// A, B, C are stored in column major!
void dmat_x_dmat(double alpha, double *A, bool trans_A, double *B, bool trans_B, double beta, double *C, size_t m, size_t n, size_t k) {
	ptrdiff_t mm = (ptrdiff_t)m, nn = (ptrdiff_t)n, kk = (ptrdiff_t)k;
	ptrdiff_t lda = trans_A? kk:mm, ldb = trans_B? nn:kk, ldc = mm;
	char transpose = 'T', notranspose = 'N'; 
	char *transa = trans_A? &transpose: &notranspose;
	char *transb = trans_B? &transpose: &notranspose;
	dgemm_(transa, transb, &mm, &nn, &kk, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// Input: an n*k row-major matrix H
// Output: an k*k matrix H^TH
void doHTH(double *H, double *HTH, size_t n, size_t k) {
	bool transpose = true;
	dmat_x_dmat(1.0, H, !transpose, H, transpose, 0.0, HTH, k, k, n);
}
// Input: an n*k row-major matrix V and an k*k row-major symmetric matrix M
// Output: an n*k row-major matrix VM = alpha*V*M + beta*VM
void doVM(double alpha, double *V, double *M, double beta, double *VM, size_t n, size_t k) {
	bool transpose = true;
	dmat_x_dmat(alpha, M, !transpose, V, !transpose, beta, VM, k, n, k);
}

bool ls_solve_chol(double *A, const double *b, int n,  double *x) {
  ptrdiff_t nn=n, lda=n, ldb=n, nrhs=1, info; 
  //int nn=n, lda=n, ldb=n, nrhs=1, info; 
  char uplo = 'U';
  memcpy(x,  b, sizeof(double)*n);
  dposv_(&uplo, &nn, &nrhs, A, &lda, x, &ldb, &info);
  return (info == 0);
}

// Solving A X = B, the output X is stored in B
// A is an m*m matrix, B is an m*n matrix
bool ls_solve_chol_matrix(double *A, double *B, long m, long n) {
  ptrdiff_t mm=(ptrdiff_t)m, lda=(ptrdiff_t)m, ldb=(ptrdiff_t)m, nrhs=(ptrdiff_t)n, info; 
  //int nn=n, lda=n, ldb=n, nrhs=1, info; 
  char uplo = 'U';
  dposv_(&uplo, &mm, &nrhs, A, &lda, B, &ldb, &info);
  return (info == 0);
}

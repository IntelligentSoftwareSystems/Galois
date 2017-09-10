#ifndef DMAT_M_H
#define DMAT_M_H


// return x'*y
double do_dot_product(double *x, double *y, size_t size);

// y = alpha*x + y
void do_axpy(double alpha, double *x, double *y, size_t size);

// y = x
void do_copy(double *x, double *y, size_t size);

// C = alpha*A*B + beta*C
// C : m * n, k is the dimension of the middle
// A, B, C are stored in column major!
void dmat_x_dmat(double alpha, double *A, bool trans_A, double *B, bool trans_B, 
		double beta, double *C, size_t m, size_t n, size_t k);

// Input: an n*k row-major matrix H
// Output: an k*k matrix H^TH
void doHTH(double *H, double *HTH, size_t n, size_t k);

// Input: an n*k row-major matrix V and an k*k row-major symmetric matrix M
// Output: an n*k row-major matrix VM = alpha*V*M + beta*VM
void doVM(double alpha, double *V, double *M, double beta, double *VM, size_t n, size_t k);

// Solve Ax = b, A is symmetric positive definite
bool ls_solve_chol(double *A, const double *b, int n,  double *x);

// Solving AX = B, A is symmetric positive definite
// the output X is stored in B
// A is an m*m matrix, B is an m*n matrix
bool ls_solve_chol_matrix(double *A, double *B, long m, long n=1);

#endif

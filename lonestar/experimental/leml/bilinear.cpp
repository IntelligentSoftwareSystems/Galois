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

#include <time.h>
#include <cstddef>
#include <map>
#include "bilinear.h"
#include "multiple_linear.h"
#include "tron.h"
#include "smat.h"
#include "dmat.h"
#ifdef EXP_DOALL_GALOIS
#include "galois/Galois.h"
#endif

/*
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
*/

#define Malloc(type, n) (type*)malloc((n) * sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char* s) {
  fputs(s, stdout);
  fflush(stdout);
}

static void (*liblinear_print_string)(const char*) = &print_string_stdout;

/*
#if 1
static void info(const char *fmt,...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

*/
/*

// ---------------------- Dense Matrix Mulitplication ----------------------
extern "C" {
double dnrm2_(ptrdiff_t *, double *, ptrdiff_t *);
double ddot_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
ptrdiff_t daxpy_(ptrdiff_t *, double *, double *, ptrdiff_t *, double *,
ptrdiff_t *); ptrdiff_t dscal_(ptrdiff_t *, double *, double *, ptrdiff_t *);
ptrdiff_t dcopy_(ptrdiff_t *n, double *sx, ptrdiff_t *incx, double *sy,
ptrdiff_t *incy);

void dgemm_(
    char *transa, char *transb,
    ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k,
    double *alpha, double *a, ptrdiff_t *lda,
    double *b, ptrdiff_t *ldb,
    double *beta, double *c, ptrdiff_t *ldc
);
}

// return x'*y
double inner_product(double *x, double *y, size_t size) {
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
void dmat_x_dmat(double alpha, double *A, bool trans_A, double *B, bool trans_B,
double beta, double *C, size_t m, size_t n, size_t k) { ptrdiff_t mm =
(ptrdiff_t)m, nn = (ptrdiff_t)n, kk = (ptrdiff_t)k; ptrdiff_t lda = trans_A?
kk:mm, ldb = trans_B? nn:kk, ldc = mm; char transpose = 'T', notranspose = 'N';
    char *transa = trans_A? &transpose: &notranspose;
    char *transb = trans_B? &transpose: &notranspose;
    dgemm_(transa, transb, &mm, &nn, &kk, &alpha, A, &lda, B, &ldb, &beta, C,
&ldc);
}

// Input: an n*k row-major matrix H
// Output: an k*k matrix H^TH
void doHTH(double *H, double *HTH, size_t n, size_t k) {
    bool transpose = true;
    dmat_x_dmat(1.0, H, !transpose, H, transpose, 0.0, HTH, k, k, n);
}
// Input: an n*k row-major matrix V and an k*k row-major symmetric matrix M
// Output: an n*k row-major matrix VM = alpha*V*M + beta*VM
void doVM(double alpha, double *V, double *M, double beta, double *VM, size_t n,
size_t k) { bool transpose = true; dmat_x_dmat(alpha, M, !transpose, V,
!transpose, beta, VM, k, n, k);
}

*/

class l2r_bilinear_ls_fun_full_weight_fast : public function {
protected:
  double Cp, Cn, alpha, beta;
  double* z;                 // nnz_Y
  double* u;                 // nnz_Y
  double *HTH, *YH;          // m*k
  double *mk_buf1, *mk_buf2; // m*k
  const bilinear_problem* prob;

public:
  l2r_bilinear_ls_fun_full_weight_fast(const bilinear_problem* prob, double Cp,
                                       double Cn) {
    this->prob = prob;
    long nnz_Y = prob->Y->nnz;
    z          = new double[nnz_Y];
    YH         = new double[prob->m * prob->k];
    HTH        = new double[prob->k * prob->k];
    mk_buf1    = new double[prob->m * prob->k];
    mk_buf2    = new double[prob->m * prob->k];
    this->Cp   = Cp;
    this->Cn   = Cn;
    alpha      = Cp - Cn;
    beta       = Cn;
    smat_x_dmat(*(prob->Y), prob->H, prob->k, YH);
    doHTH(prob->H, HTH, prob->l, prob->k);
  }
  ~l2r_bilinear_ls_fun_full_weight_fast() {
    delete[] z;
    delete[] YH;
    delete[] HTH;
    delete[] mk_buf1;
    delete[] mk_buf2;
  }

  int get_nr_variable(void) { return (int)(prob->f * prob->k); }

  void barXv_withXV(double* XV, double* barXv) {
    smat_t& Y = *(prob->Y);
    double* H = prob->H;
    long k    = prob->k;

    // barXv(i,j) = (XV)(i,:) * H(j,:)', forall (i,j) in \Omega
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<size_t>(0),
                   boost::counting_iterator<size_t>(Y.rows), [&](size_t i) {
#else
#pragma omp parallel for schedule(dynamic, 50) shared(Y, barXv)
    for (size_t i = 0; i < Y.rows; ++i) {
#endif
                     size_t j = 0;
                     for (long idx = Y.row_ptr[i]; idx < Y.row_ptr[i + 1];
                          ++idx) {
                       barXv[idx] = 0;
                       j          = Y.col_idx[idx];
                       for (long s = 0; s < k; ++s)
                         barXv[idx] += XV[i * k + s] * H[j * k + s];
                     }
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
  }

  double fun(double* w) {
    long nnz_Y = prob->Y->nnz, w_size = get_nr_variable();
    long k    = prob->k;
    smat_t& X = *(prob->X);
    double* y = prob->Y->val_t;
    double f  = 0;

    double *XW = mk_buf1, *XWHTH = mk_buf2;
    smat_x_dmat(X, w, k, XW);
    doVM(1.0, XW, HTH, 0.0, XWHTH, prob->m, k);
    barXv_withXV(XW, z); // z = vec(XWH^T) with nnz_Y entries

    double Wnorm        = do_dot_product(w, w, w_size);
    double Ynorm        = do_dot_product(y, y, nnz_Y);
    double znorm        = do_dot_product(z, z, nnz_Y);
    double Y_dot_z      = do_dot_product(y, z, nnz_Y);
    double XW_dot_YH    = do_dot_product(XW, YH, prob->m * k);
    double XW_dot_XWHTH = do_dot_product(XW, XWHTH, prob->m * k);

    // loss1 = loss on the observed entries
    double loss1 = Ynorm + znorm - 2.0 * Y_dot_z;
    // loss2 = loss on all entries ( ||Y - XWH^T||_F^2 )
    double loss2 = Ynorm + XW_dot_XWHTH - 2.0 * XW_dot_YH;
    f            = alpha * loss1 + beta * loss2 + 0.5 * Wnorm;
    return (f);
  }

  void grad(double* w, double* g) {
    // Assume fun has been called before this function call
    // so: mk_buf2 = X*W*H^T*H; z = barXw
    smat_t &Y = *(prob->Y), Z = Y;
    smat_t &X  = *(prob->X), Xt;
    Xt         = X.transpose();
    double* y  = prob->Y->val_t;
    long nnz_Y = prob->Y->nnz;
    double *ZH = mk_buf1, *XWHTH = mk_buf2;

    // ZH = (Y-Z)H (i.e., DH in Alg 1)
    do_axpy(-1.0, y, z, nnz_Y);
    Z.val_t = z;
    smat_x_dmat(Z, prob->H, prob->k, ZH);

    // XWHTH - YH (i.e., AM - C in Alg 2)
    do_axpy(-1.0, YH, XWHTH, prob->m * prob->k);

    // 2.0 * X^T (alpha * DH + beta * (AM - C))
    if (alpha >= beta) {
      do_axpy(beta / alpha, XWHTH, ZH, prob->m * prob->k);
      smat_x_dmat(2.0 * alpha, Xt, ZH, prob->k, w, g);
    } else {
      do_axpy(alpha / beta, ZH, XWHTH, prob->m * prob->k);
      smat_x_dmat(2.0 * beta, Xt, XWHTH, prob->k, w, g);
    }
  }

  void Hv(double* s, double* Hs) {
    smat_t &Y = *(prob->Y), U = Y;
    smat_t &X  = *(prob->X), Xt;
    Xt         = X.transpose();
    double *XS = mk_buf1, *XSHTH = mk_buf2;

    // XS (i.e., B in Alg 1 & 2)
    smat_x_dmat(X, s, prob->k, XS);
    // U = XSH^T with only nnz_Y entries (i.e., U in Alg 1)
    barXv_withXV(XS, z);
    U.val_t = z;

    // XSHTH (i.e., BM in Alg 2)
    doVM(1.0, XS, HTH, 0.0, XSHTH, prob->m, prob->k);
    // UH in Alg 1
    double* UH = mk_buf1; // utlizing the space as XS is no longer required.
    smat_x_dmat(U, prob->H, prob->k, UH);

    // 2.0 * X^T (alpha * UH + beta * BM)
    if (alpha >= beta) {
      do_axpy(beta / alpha, XSHTH, UH, prob->m * prob->k);
      smat_x_dmat(2.0 * alpha, Xt, UH, prob->k, s, Hs);
    } else {
      do_axpy(alpha / beta, UH, XSHTH, prob->m * prob->k);
      smat_x_dmat(2.0 * beta, Xt, XSHTH, prob->k, s, Hs);
    }
  }
};

class l2r_bilinear_ls_fun_full_fast : public function {
protected:
  double Cp;
  double* z;
  double* D;
  double *mk_buf1, *mk_buf2, *mk_buf3;
  double* HTH;
  const bilinear_problem* prob;

public:
  l2r_bilinear_ls_fun_full_fast(const bilinear_problem* prob, double Cp) {
    this->prob = prob;
    long nnz_Y = prob->Y->nnz;
    z          = new double[nnz_Y];
    D          = new double[nnz_Y];
    mk_buf1    = new double[prob->m * prob->k];
    mk_buf2    = new double[prob->m * prob->k];
    mk_buf3    = new double[prob->m * prob->k];
    HTH        = new double[prob->k * prob->k];
    this->Cp   = Cp;
    doHTH(prob->H, HTH, prob->l, prob->k);
  }
  ~l2r_bilinear_ls_fun_full_fast() {
    delete[] z;
    delete[] D;
    delete[] mk_buf1;
    delete[] mk_buf2;
    delete[] mk_buf3;
    delete[] HTH;
  }

  int get_nr_variable(void) { return (int)(prob->f * prob->k); }

  double fun(double* w) {
    smat_t& Y  = *(prob->Y);
    smat_t& X  = *(prob->X);
    double f   = 0;
    double* y  = prob->Y->val_t;
    long nnz_Y = prob->Y->nnz, w_size = get_nr_variable();
    long k = prob->k;

    double Wnorm, Ynorm, XW_YHnorm, XW_XWHTHnorm;
    Wnorm = 0.5 * do_dot_product(w, w, w_size);
    Ynorm = Cp * do_dot_product(y, y, nnz_Y);
    f += Wnorm + Ynorm;

    double *XW = mk_buf1, *YH = mk_buf2;
    smat_x_dmat(X, w, k, XW);
    smat_x_dmat(Y, prob->H, k, YH);
    XW_YHnorm = Cp * do_dot_product(XW, YH, prob->m * k);
    f -= 2.0 * XW_YHnorm;

    double* XWHTH = mk_buf3;
    doVM(1.0, XW, HTH, 0.0, XWHTH, prob->m, k);
    XW_XWHTHnorm = Cp * do_dot_product(XW, XWHTH, prob->m * k);
    f += XW_XWHTHnorm;

    return (f);
  }
  void grad(double* w, double* g) {
    // Assume fun has been called before this function call
    // mk_buf2 = Y*H; mk_buf2 = X*W*H^T*H
    smat_t &X  = *(prob->X), Xt;
    Xt         = X.transpose();
    double *YH = mk_buf2, *XWHTH = mk_buf3;
    do_axpy(-1.0, YH, XWHTH, prob->m * prob->k);
    smat_x_dmat(2.0 * Cp, Xt, XWHTH, prob->k, w, g);
  }
  void Hv(double* s, double* Hs) {
    smat_t &X  = *(prob->X), Xt;
    Xt         = X.transpose();
    double *XS = mk_buf1, *XSHTH = mk_buf2;
    smat_x_dmat(X, s, prob->k, XS);
    doVM(1.0, XS, HTH, 0.0, XSHTH, prob->m, prob->k);
    smat_x_dmat(2.0 * Cp, Xt, XSHTH, prob->k, s, Hs);
  }
};
// For imcomplete label information
class l2r_bilinear_ls_fun_fast : public function {
protected:
  double* C;
  double* z;
  double* D;
  double* XV;
  const bilinear_problem* prob;

public:
  l2r_bilinear_ls_fun_fast(const bilinear_problem* prob, double* C) {
    this->prob = prob;
    long nnz_Y = prob->Y->nnz;
    z          = new double[nnz_Y];
    D          = new double[nnz_Y];
    XV         = new double[prob->m * prob->k]; // (XV)(i,s) = XV[i*k+s]
    this->C    = C;
  }
  ~l2r_bilinear_ls_fun_fast() {
    delete[] z;
    delete[] D;
    delete[] XV;
  }

  int get_nr_variable(void) { return (int)(prob->f * prob->k); }

  void barXv(double* v, double* barXv) {
    smat_t& Y = *(prob->Y);
    smat_t& X = *(prob->X);
    double* H = prob->H;
    long k    = prob->k;

    // XV = X*V
    smat_x_dmat(X, v, k, XV);

    // barXv(i,j) = (XV)(i,:) * H(j,:)', forall (i,j) in \Omega
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<size_t>(0),
                   boost::counting_iterator<size_t>(Y.rows), [&](size_t i) {
#else
#pragma omp parallel for schedule(dynamic, 50) shared(X, Y, barXv)
    for (size_t i = 0; i < Y.rows; ++i) {
#endif
                     size_t j = 0;
                     for (long idx = Y.row_ptr[i]; idx < Y.row_ptr[i + 1];
                          ++idx) {
                       barXv[idx] = 0;
                       j          = Y.col_idx[idx];
                       for (long s = 0; s < k; ++s)
                         barXv[idx] += XV[i * k + s] * H[j * k + s];
                     }
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
  }

  // barXTu = a * barXT * u + W0
  void barXTu(double* u, double a, double* W0, double* barXTu) {
    smat_t& Y = *(prob->Y);
    smat_t &X = *(prob->X), Xt;
    Xt        = X.transpose();
    double* H = prob->H;
    long k    = prob->k;
    smat_t U  = Y;
    U.val_t   = u; // m*k
    // UH = U*H
    double* UH = XV;
    smat_x_dmat(U, H, k, UH);
    smat_x_dmat(a, Xt, UH, k, W0, barXTu);
  }

  double fun(double* w) {
    double f    = 0;
    double* y   = prob->Y->val_t;
    long nnz_Y  = prob->Y->nnz;
    long w_size = get_nr_variable();

    f += 0.5 * do_dot_product(w, w, w_size);

    // z = barX*w - y
    barXv(w, z);
    do_axpy(-1.0, y, z, nnz_Y);

#pragma omp parallel for schedule(static) reduction(+ : f)
    for (long i = 0; i < nnz_Y; i++) {
      f += C[i] * z[i] * z[i];
    }
    return (f);
  }

  void grad(double* w, double* g) {
    // Assume fun has been called before this function call
    // z = barX*w-y
    long nnz_Y = prob->Y->nnz;
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<long>(0),
                   boost::counting_iterator<long>(nnz_Y), [&](long i) {
#else
#pragma omp parallel for schedule(static)
    for (long i = 0; i < nnz_Y; ++i) {
#endif
                     z[i] = C[i] * z[i];
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
    barXTu(z, 2, w, g);
  }
  void Hv(double* s, double* Hs) {
    long nnz_Y = prob->Y->nnz;
    barXv(s, D);
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<long>(0),
                   boost::counting_iterator<long>(nnz_Y), [&](long i) {
#else
#pragma omp parallel for schedule(static)
    for (long i = 0; i < nnz_Y; ++i) {
#endif
                     D[i] *= C[i];
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
    barXTu(D, 2, s, Hs);
  }
};

class l2r_bilinear_lr_fun_fast : public function {
protected:
  double* C;
  double* z;
  double* D;
  double* wa;
  double* XV;
  const bilinear_problem* prob;

public:
  l2r_bilinear_lr_fun_fast(const bilinear_problem* prob, double* C) {
    this->prob = prob;
    long nnz_Y = prob->Y->nnz;
    z          = new double[nnz_Y];
    D          = new double[nnz_Y];
    wa         = new double[nnz_Y];
    XV         = new double[prob->m * prob->k]; // (XV)(i,s) = XV[i*k+s]
    this->C    = C;
  }
  ~l2r_bilinear_lr_fun_fast() {
    delete[] z;
    delete[] D;
    delete[] wa;
    delete[] XV;
  }

  int get_nr_variable(void) { return (int)(prob->f * prob->k); }

  void barXv(double* v, double* barXv) {
    smat_t& Y = *(prob->Y);
    smat_t& X = *(prob->X);
    double* H = prob->H;
    long k    = prob->k;

    // XV = X*V
    smat_x_dmat(X, v, k, XV);

    // barXv(i,j) = (XV)(i,:) * H(j,:)', forall (i,j) in \Omega
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<size_t>(0),
                   boost::counting_iterator<size_t>(Y.rows), [&](size_t i) {
#else
#pragma omp parallel for schedule(dynamic, 50) shared(X, Y, barXv)
    for (size_t i = 0; i < Y.rows; ++i) {
#endif
                     size_t j = 0;
                     for (long idx = Y.row_ptr[i]; idx < Y.row_ptr[i + 1];
                          ++idx) {
                       barXv[idx] = 0;
                       j          = Y.col_idx[idx];
                       for (long s = 0; s < k; ++s)
                         barXv[idx] += XV[i * k + s] * H[j * k + s];
                     }
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
  }

  // barXTu = a * barXT * u + W0
  void barXTu(double* u, double a, double* W0, double* barXTu) {
    smat_t& Y = *(prob->Y);
    smat_t &X = *(prob->X), Xt;
    Xt        = X.transpose();
    double* H = prob->H;
    long k    = prob->k;
    smat_t U  = Y;
    U.val_t   = u; // m*k
    // UH = U*H
    double* UH = XV;
    smat_x_dmat(U, H, k, UH);
    smat_x_dmat(a, Xt, UH, k, W0, barXTu);
  }

  double fun(double* w) {
    double f        = 0;
    const double* y = prob->Y->val_t;
    long nnz_Y      = prob->Y->nnz;
    long w_size     = get_nr_variable();

    f += 0.5 * do_dot_product(w, w, w_size);

    barXv(w, z);
#pragma omp parallel for schedule(static) reduction(+ : f)
    for (long i = 0; i < nnz_Y; i++) {
      double yz = y[i] * z[i];
      if (yz >= 0)
        f += C[i] * log(1 + exp(-yz));
      else
        f += C[i] * (-yz + log(1 + exp(yz)));
    }
    return (f);
  }
  void grad(double* w, double* g) {
    // Assume fun() has been called before this function call
    // z = Xw
    const double* y = prob->Y->val_t;
    long nnz_Y      = prob->Y->nnz;

#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<long>(0),
                   boost::counting_iterator<long>(nnz_Y), [&](long i) {
#else
#pragma omp parallel for schedule(static)
    for (long i = 0; i < nnz_Y; ++i) {
#endif
                     z[i] = 1 / (1 + exp(-y[i] * z[i]));
                     D[i] = C[i] * z[i] * (1 - z[i]);
                     z[i] = C[i] * (z[i] - 1) * y[i];
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif

    barXTu(z, 1, w, g);
  }
  void Hv(double* s, double* Hs) {
    long nnz_Y = prob->Y->nnz;

    barXv(s, wa);
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<long>(0),
                   boost::counting_iterator<long>(nnz_Y), [&](long i) {
#else
#pragma omp parallel for schedule(static)
    for (long i = 0; i < nnz_Y; ++i) {
#endif
                     wa[i] *= D[i];
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
    barXTu(wa, 1, s, Hs);
  }
};

class l2r_bilinear_l2svc_fast_fun : public function {
protected:
  double* C;
  double* z;
  double* D;
  double* XV;
  int* I;
  const bilinear_problem* prob;

public:
  l2r_bilinear_l2svc_fast_fun(const bilinear_problem* prob, double* C) {
    this->prob = prob;
    long nnz_Y = prob->Y->nnz;
    z          = new double[nnz_Y];
    D          = new double[nnz_Y];
    I          = new int[nnz_Y];
    XV         = new double[prob->m * prob->k]; // (XV)(i,s) = XV[s*m+i]
    this->C    = C;
    memset(I, 0, sizeof(int) * nnz_Y);
  }
  ~l2r_bilinear_l2svc_fast_fun() {
    delete[] z;
    delete[] D;
    delete[] I;
    delete[] XV;
  }

  int get_nr_variable(void) { return (int)(prob->f * prob->k); }

  void barXv(double* v, double* barXv) {
    smat_t& Y = *(prob->Y);
    smat_t& X = *(prob->X);
    double* H = prob->H;
    long k    = prob->k;

    // XV = X*V
    smat_x_dmat(X, v, k, XV);

    // barXv(i,j) = (XV)(i,:) * H(j,:)', forall (i,j) in \Omega
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<size_t>(0),
                   boost::counting_iterator<size_t>(Y.rows), [&](size_t i) {
#else
#pragma omp parallel for schedule(dynamic, 50) shared(X, Y, barXv)
    for (size_t i = 0; i < Y.rows; ++i) {
#endif
                     size_t j = 0;
                     for (long idx = Y.row_ptr[i]; idx < Y.row_ptr[i + 1];
                          ++idx) {
                       barXv[idx] = 0;
                       j          = Y.col_idx[idx];
                       for (long s = 0; s < k; ++s)
                         barXv[idx] += XV[i * k + s] * H[j * k + s];
                     }
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
  }

  void subbarXv(double* v, double* barXv) {
    smat_t& Y = *(prob->Y);
    smat_t& X = *(prob->X);
    double* H = prob->H;
    long k    = prob->k;

    // XV = X*V
    smat_x_dmat(X, v, k, XV);

    // barXv(i,j) = (XV)(i,:) * H(j,:)', forall (i,j) in \Omega
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<size_t>(0),
                   boost::counting_iterator<size_t>(Y.rows), [&](size_t i) {
#else
#pragma omp parallel for schedule(dynamic, 50) shared(X, Y, barXv)
    for (size_t i = 0; i < Y.rows; ++i) {
#endif
                     size_t j = 0;
                     for (long idx = Y.row_ptr[i]; idx < Y.row_ptr[i + 1];
                          ++idx) {
                       if (I[idx] != 0) {
                         barXv[idx] = 0;
                         j          = Y.col_idx[idx];
                         for (long s = 0; s < k; ++s)
                           barXv[idx] += XV[i * k + s] * H[j * k + s];
                       } else
                         barXv[idx] = 0;
                     }
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
  }

  // barXTu = a * barXT * u + W0
  void barXTu(double* u, double a, double* W0, double* barXTu) {
    smat_t& Y = *(prob->Y);
    smat_t &X = *(prob->X), Xt;
    Xt        = X.transpose();
    double* H = prob->H;
    long k    = prob->k;
    smat_t U  = Y;
    U.val_t   = u; // m*k
    // UH = U*H
    double* UH = XV;
    smat_x_dmat(U, H, k, UH);
    smat_x_dmat(a, Xt, UH, k, W0, barXTu);
  }

  double fun(double* w) {
    double f        = 0;
    const double* y = prob->Y->val_t;
    long nnz_Y      = prob->Y->nnz;
    long w_size     = get_nr_variable();

    barXv(w, z);
    f = 0.5 * do_dot_product(w, w, w_size);

#pragma omp parallel for schedule(static) reduction(+ : f)
    for (long i = 0; i < nnz_Y; ++i) {
      z[i] *= y[i];
      double d = (1 - z[i]);
      if (d > 0) {
        f += C[i] * d * d;
        z[i] = C[i] * y[i] * (z[i] - 1);
        I[i] = 1;
      } else
        z[i] = 0;
    }
    /*
    long cnt = 0;
    for(long i=0;i<nnz_Y;++i)
        if(I[i]!=0) cnt++;
    printf("#I[i]!=0 %ld ratio %g%%\n", cnt, (double)(100*cnt)/(double)nnz_Y);
    */
    return (f);
  }
  void grad(double* w, double* g) {
    // Assume fun has been called before this function call
    barXTu(z, 2.0, w, g);
  }

  void Hv(double* s, double* Hs) {
    long nnz_Y = prob->Y->nnz;
    subbarXv(s, D);
#ifdef EXP_DOALL_GALOIS
    galois::do_all(boost::counting_iterator<long>(0),
                   boost::counting_iterator<long>(nnz_Y), [&](long i) {
#else
#pragma omp parallel for schedule(static)
    for (long i = 0; i < nnz_Y; ++i) {
#endif
                     D[i] *= C[i];
#ifdef EXP_DOALL_GALOIS
                   });
#else
    }
#endif
    barXTu(D, 2.0, s, Hs);
  }
};

static void construct_C_array(double* C, smat_t* Y, double Cp, double Cn) {
  double* y    = Y->val_t;
  size_t nnz_Y = Y->nnz;
#ifdef EXP_DOALL_GALOIS
  galois::do_all(boost::counting_iterator<size_t>(0),
                 boost::counting_iterator<size_t>(nnz_Y), [&](size_t i) {
#else
#pragma omp parallel for
  for (size_t i = 0; i < nnz_Y; ++i) {
#endif
                   C[i] = y[i] > 0 ? Cp : Cn;
#ifdef EXP_DOALL_GALOIS
                 });
#else
  }
#endif
}

// sample K numbers from 0 ~ N-1
std::vector<int> subsample(int K, int N) {
  std::vector<int> subset(N);
  for (int i = 0; i < N; i++)
    subset[i] = i;
  for (int i = 0; i < K; i++) {
    int j     = i + rand() % (N - i);
    int tmp   = subset[i];
    subset[i] = subset[j];
    subset[j] = tmp;
  }
  subset.resize(K);
  return subset;
}

// input w is the initial W.
void bilinear_train(const bilinear_problem* prob,
                    const bilinear_parameter* param, double* w,
                    double* walltime, double* cputime) {
  double* y  = prob->Y->val_t;
  size_t pos = 0, neg = 0, nnz_Y = prob->Y->nnz;
  size_t w_size       = (size_t)(prob->f * prob->k);
  double time_start   = omp_get_wtime();
  double eps          = param->eps;
  clock_t clock_start = clock();

  if (prob->W)
    for (size_t i = 0; i < w_size; ++i)
      w[i] = prob->W[i];

  for (size_t idx = 0; idx < nnz_Y; ++idx)
    if (y[idx] > 0)
      pos++;
  neg = nnz_Y - pos;

  double primal_solver_classification_tol =
      eps * (double)std::max(std::min(pos, neg), 1UL) / (double)nnz_Y;
  double primal_solver_regression_tol = eps * (double)std::max(nnz_Y, 1UL) /
                                        (double)(prob->Y->cols * prob->Y->rows);
  function* fun_obj = NULL;
  switch (param->solver_type) {
  case L2R_BILINEAR_LS: {
    double* C = MALLOC(double, nnz_Y);
    construct_C_array(C, prob->Y, param->Cp, param->Cn);
    fun_obj = new l2r_bilinear_ls_fun_fast(prob, C);
    TRON tron_obj(fun_obj, param->eps, param->max_tron_iter,
                  param->max_cg_iter);
    tron_obj.set_print_string(liblinear_print_string);
    tron_obj.tron(w, false); // prob->W is the initial
    delete fun_obj;
    free(C);
    break;
  }
  case L2R_BILINEAR_LR: {
    double* C = MALLOC(double, nnz_Y);
    construct_C_array(C, prob->Y, param->Cp, param->Cn);
    fun_obj = new l2r_bilinear_lr_fun_fast(prob, C);
    TRON tron_obj(fun_obj, primal_solver_classification_tol,
                  param->max_tron_iter, param->max_cg_iter);
    tron_obj.set_print_string(liblinear_print_string);
    tron_obj.tron(w, false); // prob->W is the initial
    delete fun_obj;
    free(C);
    break;
  }
  case L2R_BILINEAR_SVC: {
    double* C = MALLOC(double, nnz_Y);
    construct_C_array(C, prob->Y, param->Cp, param->Cn);
    fun_obj = new l2r_bilinear_l2svc_fast_fun(prob, C);
    TRON tron_obj(fun_obj, primal_solver_classification_tol,
                  param->max_tron_iter, param->max_cg_iter);
    tron_obj.set_print_string(liblinear_print_string);
    tron_obj.tron(w, false); // prob->W is the initial
    delete fun_obj;
    free(C);
    break;
  }
  case L2R_BILINEAR_LS_FULL: {
    fun_obj = new l2r_bilinear_ls_fun_full_fast(prob, param->Cp);
    // TRON tron_obj(fun_obj, param->eps, param->max_tron_iter,
    // param->max_cg_iter);
    TRON tron_obj(fun_obj, primal_solver_regression_tol, param->max_tron_iter,
                  param->max_cg_iter);
    tron_obj.set_print_string(liblinear_print_string);
    tron_obj.tron(w, false); // prob->W is the initial
    delete fun_obj;
    break;
  }
  case L2R_BILINEAR_LS_FULL_WEIGHTED: {
    fun_obj =
        new l2r_bilinear_ls_fun_full_weight_fast(prob, param->Cp, param->Cn);
    // TRON tron_obj(fun_obj, param->eps, param->max_tron_iter,
    // param->max_cg_iter);
    TRON tron_obj(fun_obj, primal_solver_regression_tol, param->max_tron_iter,
                  param->max_cg_iter);
    tron_obj.set_print_string(liblinear_print_string);
    tron_obj.tron(w, false); // prob->W is the initial
    delete fun_obj;
    break;
  }
  default:
    fprintf(stderr, "ERROR: unknown solver_type\n");
    break;
  }

  if (walltime)
    *walltime += (omp_get_wtime() - time_start);
  if (cputime)
    *cputime += ((double)(clock() - clock_start) / CLOCKS_PER_SEC);
}

//-----------------Prediction
// Evaluation--------------------------------------------------
typedef std::vector<double> dvec_t;
typedef std::vector<int> ivec_t;
// only for auc
class Comp {
  const double* dec_val;

public:
  Comp(const double* ptr) : dec_val(ptr) {}
  bool operator()(int i, int j) const { return dec_val[i] > dec_val[j]; }
};

double auc(const dvec_t& dec_values, const ivec_t& ty, ivec_t& indices,
           double threshold, size_t* numerator, size_t* denom1,
           size_t* denom2) {
  double roc  = 0;
  size_t size = dec_values.size();

  for (unsigned i = 0; i < size; ++i)
    indices[i] = i;

  std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

  int tp = 0, fp = 0;
  bool first = true;
  for (size_t i = 0; i < size; i++) {
    if (first and dec_values[indices[i]] < threshold) {
      *numerator += tp;
      *denom1 += i;
      first = false;
    }
    if (ty[indices[i]] == 1)
      tp++;
    else if (ty[indices[i]] == -1) {
      roc += tp;
      fp++;
    }
  }
  *denom2 += tp;

  if (tp == 0 || fp == 0)
    roc = 0;
  else
    roc = roc / tp / fp;

  return roc;
}

void bilinear_predict_full(const bilinear_problem* prob, const double* H,
                           bilinear_eval_result* eval_result) {
  smat_t& Y        = *(prob->Y);
  smat_t& X        = *(prob->X);
  int nr_threads   = omp_get_max_threads();
  int rank         = (int)prob->k;
  int top_p        = eval_result->top_p;
  size_t nr_labels = prob->l;
  size_t nr_insts  = prob->m;
  std::vector<ivec_t> trueY_set(nr_threads, ivec_t(nr_labels));
  std::vector<ivec_t> indices_set(nr_threads, ivec_t(nr_labels));
  std::vector<dvec_t> predY_set(nr_threads, dvec_t(nr_labels));
  std::vector<dvec_t> top_acc_set(nr_threads, dvec_t(top_p));
  std::vector<ivec_t> top_real_acc_set(nr_threads, ivec_t(top_p));
  std::vector<size_t> micro_F_numerator(nr_threads);
  std::vector<size_t> micro_F_denom1(nr_threads), micro_F_denom2(nr_threads);
  std::vector<size_t> hamming_loss_set(nr_threads);
  dvec_t sum_auc_set(nr_threads);

  double time_start   = omp_get_wtime();
  clock_t clock_start = clock();
  double* W           = MALLOC(double, nr_labels* rank);
  // transpose W = H^T
  for (size_t i = 0; i < nr_labels; i++)
    for (int t = 0; t < rank; t++)
      W[i + t * nr_labels] = H[t + i * rank];
  double* XW = MALLOC(double, prob->m * prob->l);
  smat_x_dmat(X, W, nr_labels, XW);

  // size_t true_insts = 0;
#ifdef EXP_DOALL_GALOIS
  galois::do_all(
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(nr_insts), [&](size_t inst) {
#else
#pragma omp parallel for
  for (size_t inst = 0; inst < nr_insts; ++inst) {
#endif
#ifdef EXP_DOALL_GALOIS
        int tid = galois::substrate::ThreadPool::getTID();
#else
    int tid = omp_get_thread_num(); // thread ID
#endif
        ivec_t& trueY        = trueY_set[tid];
        ivec_t& indices      = indices_set[tid];
        dvec_t& predY        = predY_set[tid];
        dvec_t& top_acc      = top_acc_set[tid];
        ivec_t& top_real_acc = top_real_acc_set[tid];
        double& sum_auc      = sum_auc_set[tid];
        size_t& numerator    = micro_F_numerator[tid];
        size_t& denom1       = micro_F_denom1[tid];
        size_t& denom2       = micro_F_denom2[tid];
        size_t& hamming_loss = hamming_loss_set[tid];

        for (size_t j = 0; j < nr_labels; j++) {
          trueY[j] = -1;
          predY[j] = XW[inst * nr_labels + j];
        }
        for (long idx = Y.row_ptr[inst]; idx < Y.row_ptr[inst + 1]; ++idx)
          trueY[Y.col_idx[idx]] = (int)Y.val_t[idx]; // it should always be 1

        double threshold = eval_result->threshold;
        for (size_t j = 0; j < nr_labels; j++)
          if ((trueY[j] > 0) != (predY[j] > threshold))
            hamming_loss++;
        int nr_true_pos = (int)Y.nnz_of_row((int)inst);
        sum_auc +=
            auc(predY, trueY, indices, threshold, &numerator, &denom1, &denom2);
        int correct_cnt = 0;
        for (int p = 0; p < top_p; p++) {
          if (trueY[indices[p]] != -1)
            correct_cnt++;
          if (nr_true_pos != 0)
            top_acc[p] +=
                (double)correct_cnt / (double)std::min(p + 1, nr_true_pos);
          else
            top_acc[p] += 1.0;
          top_real_acc[p] += correct_cnt;
        }
#ifdef EXP_DOALL_GALOIS
      });
#else
  }
#endif

  size_t& numerator = micro_F_numerator[0];
  size_t &denom1 = micro_F_denom1[0], &denom2 = micro_F_denom2[0];
  size_t& hamming_loss = hamming_loss_set[0];
  double& sum_auc      = sum_auc_set[0];
  dvec_t& top_acc      = top_acc_set[0];
  ivec_t& top_real_acc = top_real_acc_set[0];
  for (int tid = 1; tid < nr_threads; tid++) {
    sum_auc += sum_auc_set[tid];
    hamming_loss += hamming_loss_set[tid];
    for (int p = 0; p < top_p; p++) {
      top_acc[p] += top_acc_set[tid][p];
      top_real_acc[p] += top_real_acc_set[tid][p];
    }
    numerator += micro_F_numerator[tid];
    denom1 += micro_F_denom1[tid];
    denom2 += micro_F_denom2[tid];
  }
  eval_result->avg_auc  = sum_auc / (double)nr_insts;
  eval_result->micro_F1 = (double)(2 * numerator) / (double)(denom1 + denom2);
  eval_result->hamming_loss =
      (double)hamming_loss / ((double)(nr_insts * nr_labels));
  for (int p = 0; p < top_p; p++) {
    // if(p>0) top_acc[p] += top_acc[p-1];
    eval_result->avg_top_acc[p] = 100.0 * top_acc[p] / (double)(nr_insts);
    eval_result->avg_top_real_acc[p] =
        100.0 * top_real_acc[p] / (double)(nr_insts * (p + 1));
  }
  free(XW);
  eval_result->walltime += (omp_get_wtime() - time_start);
  eval_result->cputime += ((double)(clock() - clock_start) / CLOCKS_PER_SEC);
  // printf("testing on %ld instances (%g s)\n", nr_insts, omp_get_wtime() -
  // time_start );
}
void bilinear_predict(const bilinear_problem* prob, const double* W,
                      const double* H, bilinear_eval_result* eval_result) {
  smat_t& Y        = *(prob->Y);
  smat_t& X        = *(prob->X);
  int nr_threads   = omp_get_max_threads();
  int rank         = (int)prob->k;
  int top_p        = eval_result->top_p;
  size_t nr_labels = prob->l;
  size_t nr_insts  = prob->m;
  std::vector<ivec_t> trueY_set(nr_threads, ivec_t(nr_labels));
  std::vector<ivec_t> indices_set(nr_threads, ivec_t(nr_labels));
  std::vector<dvec_t> predY_set(nr_threads, dvec_t(nr_labels));
  std::vector<dvec_t> top_acc_set(nr_threads, dvec_t(top_p));
  std::vector<ivec_t> top_real_acc_set(nr_threads, ivec_t(top_p));
  std::vector<size_t> micro_F_numerator(nr_threads);
  std::vector<size_t> micro_F_denom1(nr_threads), micro_F_denom2(nr_threads);
  std::vector<size_t> hamming_loss_set(nr_threads);
  dvec_t sum_auc_set(nr_threads);

  double time_start   = omp_get_wtime();
  clock_t clock_start = clock();
  double* XW          = MALLOC(double, prob->m * prob->k);
  smat_x_dmat(X, W, rank, XW);

  // size_t true_insts = 0;
#ifdef EXP_DOALL_GALOIS
  galois::do_all(
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(nr_insts), [&](size_t inst) {
#else
#pragma omp parallel for
  for (size_t inst = 0; inst < nr_insts; ++inst) {
#endif
#ifdef EXP_DOALL_GALOIS
        int tid = galois::substrate::ThreadPool::getTID();
#else
    int tid = omp_get_thread_num(); // thread ID
#endif
        ivec_t& trueY        = trueY_set[tid];
        ivec_t& indices      = indices_set[tid];
        dvec_t& predY        = predY_set[tid];
        dvec_t& top_acc      = top_acc_set[tid];
        ivec_t& top_real_acc = top_real_acc_set[tid];
        double& sum_auc      = sum_auc_set[tid];
        size_t& numerator    = micro_F_numerator[tid];
        size_t& denom1       = micro_F_denom1[tid];
        size_t& denom2       = micro_F_denom2[tid];
        size_t& hamming_loss = hamming_loss_set[tid];

        for (size_t j = 0; j < nr_labels; j++) {
          trueY[j] = -1;
          predY[j] = 0;
          for (int t = 0; t < rank; t++)
            predY[j] += XW[inst * rank + t] * H[j * rank + t];
        }
        for (long idx = Y.row_ptr[inst]; idx < Y.row_ptr[inst + 1]; ++idx)
          trueY[Y.col_idx[idx]] = (int)Y.val_t[idx]; // it should always be 1

        double threshold = eval_result->threshold;
        for (size_t j = 0; j < nr_labels; j++)
          if ((trueY[j] > 0) != (predY[j] > threshold))
            hamming_loss++;
        int nr_true_pos = (int)Y.nnz_of_row((int)inst);
        sum_auc +=
            auc(predY, trueY, indices, threshold, &numerator, &denom1, &denom2);
        int correct_cnt = 0;
        for (int p = 0; p < top_p; p++) {
          if (trueY[indices[p]] != -1)
            correct_cnt++;
          if (nr_true_pos != 0)
            top_acc[p] +=
                (double)correct_cnt / (double)std::min(p + 1, nr_true_pos);
          else
            top_acc[p] += 1.0;
          top_real_acc[p] += correct_cnt;
        }
#ifdef EXP_DOALL_GALOIS
      });
#else
  }
#endif

  size_t& numerator = micro_F_numerator[0];
  size_t &denom1 = micro_F_denom1[0], &denom2 = micro_F_denom2[0];
  size_t& hamming_loss = hamming_loss_set[0];
  double& sum_auc      = sum_auc_set[0];
  dvec_t& top_acc      = top_acc_set[0];
  ivec_t& top_real_acc = top_real_acc_set[0];
  for (int tid = 1; tid < nr_threads; tid++) {
    sum_auc += sum_auc_set[tid];
    hamming_loss += hamming_loss_set[tid];
    for (int p = 0; p < top_p; p++) {
      top_acc[p] += top_acc_set[tid][p];
      top_real_acc[p] += top_real_acc_set[tid][p];
    }
    numerator += micro_F_numerator[tid];
    denom1 += micro_F_denom1[tid];
    denom2 += micro_F_denom2[tid];
  }
  eval_result->avg_auc  = sum_auc / (double)nr_insts;
  eval_result->micro_F1 = (double)(2 * numerator) / (double)(denom1 + denom2);
  eval_result->hamming_loss =
      (double)hamming_loss / ((double)(nr_insts * nr_labels));
  for (int p = 0; p < top_p; p++) {
    // if(p>0) top_acc[p] += top_acc[p-1];
    eval_result->avg_top_acc[p] = 100.0 * top_acc[p] / (double)(nr_insts);
    eval_result->avg_top_real_acc[p] =
        100.0 * top_real_acc[p] / (double)(nr_insts * (p + 1));
  }
  free(XW);
  eval_result->walltime += (omp_get_wtime() - time_start);
  eval_result->cputime += ((double)(clock() - clock_start) / CLOCKS_PER_SEC);
  // printf("testing on %ld instances (%g s)\n", nr_insts, omp_get_wtime() -
  // time_start );
}

/*
void bilinear_predict(const bilinear_problem *prob, const double *W, const
double *H, int top_p, double *avg_auc, double *micro_F1, double *avg_top_acc,
double *avg_top_real_acc, double *walltime, double *cputime) { smat_t &Y =
*(prob->Y); smat_t &X = *(prob->X); int nr_threads = omp_get_max_threads(); int
rank = (int)prob->k; size_t nr_labels = prob->l; size_t nr_insts = prob->m;
    std::vector<ivec_t> trueY_set(nr_threads, ivec_t(nr_labels));
    std::vector<ivec_t> indices_set(nr_threads, ivec_t(nr_labels));
    std::vector<dvec_t> predY_set(nr_threads, dvec_t(nr_labels));
    //std::vector<ivec_t> top_acc_set(nr_threads, ivec_t(top_p));
    std::vector<dvec_t> top_acc_set(nr_threads, dvec_t(top_p));
    std::vector<ivec_t> top_real_acc_set(nr_threads, ivec_t(top_p));
    std::vector<size_t> micro_F_numerator(nr_threads);
    std::vector<size_t> micro_F_denom1(nr_threads), micro_F_denom2(nr_threads);
    dvec_t sum_auc_set(nr_threads);

    double time_start = omp_get_wtime();
    clock_t clock_start = clock();
    double *XW = MALLOC(double, prob->m*prob->k);
    smat_x_dmat(X, W, rank, XW);

    //size_t true_insts = 0;
#pragma omp parallel for
    for(size_t inst = 0; inst < nr_insts; ++inst){
        int tid = omp_get_thread_num(); // thread ID
        ivec_t &trueY = trueY_set[tid];
        ivec_t &indices = indices_set[tid];
        dvec_t &predY = predY_set[tid];
        dvec_t &top_acc = top_acc_set[tid];
        ivec_t &top_real_acc = top_real_acc_set[tid];
        double &sum_auc = sum_auc_set[tid];
        size_t &numerator = micro_F_numerator[tid];
        size_t &denom1 = micro_F_denom1[tid];
        size_t &denom2 = micro_F_denom2[tid];

        size_t nr_pos = 0;
        for(size_t j = 0; j < nr_labels; j++){
            trueY[j] = -1; predY[j] = 0;
            for(int t = 0; t < rank; t++)
                predY[j] += XW[inst*rank+t]*H[j*rank+t];
            if(predY[j] >= 0) nr_pos ++;
        }
        for(long idx = Y.row_ptr[inst]; idx < Y.row_ptr[inst+1]; ++idx)
            trueY[Y.col_idx[idx]] = (int)Y.val_t[idx]; // it should always be 1
        int nr_true_pos = (int)Y.nnz_of_row((int)inst);

        sum_auc += auc(predY, trueY, indices, &numerator, &denom1, &denom2);
        int correct_cnt = 0;
        for(int p = 0; p < top_p; p++) {
            if(trueY[indices[p]]!=-1)
                correct_cnt++;
            if(nr_true_pos!=0)
                top_acc[p] += (double)correct_cnt/(double)std::min(p+1,
nr_true_pos); else top_acc[p] += 1.0; top_real_acc[p] += correct_cnt;
        }
    }

    size_t &numerator = micro_F_numerator[0];
    size_t &denom1 = micro_F_denom1[0], &denom2 = micro_F_denom2[0];
    double &sum_auc = sum_auc_set[0];
    dvec_t &top_acc = top_acc_set[0];
    ivec_t &top_real_acc = top_real_acc_set[0];
    for(int tid = 1; tid < nr_threads; tid++) {
        sum_auc += sum_auc_set[tid];
        for(int p = 0; p < top_p; p++){
            top_acc[p] += top_acc_set[tid][p];
            top_real_acc[p] += top_real_acc_set[tid][p];
        }
        numerator += micro_F_numerator[tid];
        denom1 += micro_F_denom1[tid];
        denom2 += micro_F_denom2[tid];
    }
    *avg_auc = sum_auc/(double)nr_insts;
    *micro_F1 = (double)(2*numerator)/(double)(denom1+denom2);
    for(int p = 0; p < top_p; p++) {
        //if(p>0) top_acc[p] += top_acc[p-1];
        avg_top_acc[p] = 100.0*top_acc[p]/(double)(nr_insts);
        avg_top_real_acc[p] = 100.0*top_real_acc[p]/(double)(nr_insts*(p+1));
    }
    free(XW);
    if(walltime) *walltime += (omp_get_wtime() - time_start);
    if(cputime) *cputime += ((double)(clock()-clock_start)/CLOCKS_PER_SEC);
    printf("testing on %ld instances (%g s)\n", nr_insts, omp_get_wtime() -
time_start );
}

*/

class spherical_cluster {
public:
  size_t n;
  int dim, nr_cluster;
  double* centroids;
  std::vector<ivec_t> clusters;
  // centroid is a nr_cluster*dim matrix stored in row-majord order
  spherical_cluster(size_t n, int dim, int nr_cluster, double* cluster_idx,
                    double* centroids) {
    this->n          = n;
    this->dim        = dim;
    this->nr_cluster = nr_cluster;
    this->centroids  = centroids;
    clusters.resize(nr_cluster);
    printf("nr_cluster %d n %ld dim %d\n", nr_cluster, n, dim);
    for (size_t i = 0; i < n; i++) {
      /*
      if((int)(cluster_idx[i]) >= 50 || (int)(cluster_idx[i] < 0)) {
          printf("i=%d cluster_idx %g (%d)\n", i, cluster_idx[i],
      (int)(cluster_idx[i]));
      } */
      clusters[(int)(cluster_idx[i] - 1)].push_back((int)i);
    }
  }
  double dist_to_a_cluster(double* x, int cid) {
    double ret = 0.0, xnorm = 0.0, cnorm = 0.0;
    for (int t = 0; t < dim; t++) {
      ret += x[t] * centroids[cid * dim + t];
      xnorm += x[t] * x[t];
      cnorm += centroids[cid * dim + t] * centroids[cid * dim + t];
    }
    return -ret / (sqrt(xnorm * cnorm));
  }
  const ivec_t& get_cluster_by_point(double* x) {
    int cid         = 0;
    double min_dist = dist_to_a_cluster(x, cid);
    for (int c = 1; c < nr_cluster; c++) {
      double dist = dist_to_a_cluster(x, c);
      if (dist < min_dist) {
        min_dist = dist;
        cid      = c;
      }
    }
    return clusters[cid];
  }
};

void bilinear_predict_with_clustering(
    const bilinear_problem* prob, const double* W, const double* H, int top_p,
    int nr_cluster, double* cluster_idx, double* centroids, double* avg_top_acc,
    double* avg_top_real_acc, double* walltime, double* cputime) {
  smat_t& Y        = *(prob->Y);
  smat_t& X        = *(prob->X);
  int nr_threads   = omp_get_max_threads();
  int rank         = (int)prob->k;
  size_t nr_labels = prob->l;
  size_t nr_insts  = prob->m;

  spherical_cluster sc(nr_labels, rank, nr_cluster, cluster_idx, centroids);

  // std::vector<ivec_t> trueY_set(nr_threads, ivec_t(nr_labels));
  std::vector<std::map<unsigned, double>> trueY_set(nr_threads);
  std::vector<ivec_t> indices_set(nr_threads, ivec_t(nr_labels));
  std::vector<dvec_t> predY_set(nr_threads, dvec_t(nr_labels));
  // std::vector<ivec_t> top_acc_set(nr_threads, ivec_t(top_p));
  std::vector<dvec_t> top_acc_set(nr_threads, dvec_t(top_p));
  std::vector<ivec_t> top_real_acc_set(nr_threads, ivec_t(top_p));
  dvec_t sum_auc_set(nr_threads);

  double time_start   = omp_get_wtime();
  clock_t clock_start = clock();
  double* XW          = MALLOC(double, prob->m * prob->k);
  smat_x_dmat(X, W, rank, XW);

  // size_t true_insts = 0;
  //#pragma omp parallel for
  for (size_t inst = 0; inst < nr_insts; ++inst) {
    int tid = omp_get_thread_num(); // thread ID
    // ivec_t &trueY = trueY_set[tid];
    std::map<unsigned, double>& trueY = trueY_set[tid];
    ivec_t& indices                   = indices_set[tid];
    dvec_t& predY                     = predY_set[tid];
    dvec_t& top_acc                   = top_acc_set[tid];
    ivec_t& top_real_acc              = top_real_acc_set[tid];
    // double &sum_auc = sum_auc_set[tid];

    trueY.clear();
    for (long idx = Y.row_ptr[inst]; idx < Y.row_ptr[inst + 1]; ++idx)
      trueY[Y.col_idx[idx]] = Y.val_t[idx];
    const ivec_t& label_cluster = sc.get_cluster_by_point(&XW[inst * rank]);
    int cluster_size            = (int)label_cluster.size();
    for (int j = 0; j < cluster_size; j++) {
      indices[j] = j;
      predY[j]   = 0.0;
      for (int t = 0; t < rank; t++)
        predY[j] += XW[inst * rank + t] * H[label_cluster[j] * rank + t];
    }
    std::sort(indices.begin(), indices.begin() + cluster_size, Comp(&predY[0]));

    // sum_auc += auc(predY, trueY, indices, &numerator, &denom1, &denom2);
    int nr_true_pos = (int)Y.nnz_of_row((int)inst);
    int correct_cnt = 0;
    for (int p = 0; p < top_p; p++) {
      correct_cnt += (int)trueY.count(label_cluster[indices[p]]);
      if (nr_true_pos != 0)
        top_acc[p] +=
            (double)correct_cnt / (double)std::min(p + 1, nr_true_pos);
      else
        top_acc[p] += 1.0;
      top_real_acc[p] += correct_cnt;
    }
  }

  //	double &sum_auc = sum_auc_set[0];
  dvec_t& top_acc      = top_acc_set[0];
  ivec_t& top_real_acc = top_real_acc_set[0];
  for (int tid = 1; tid < nr_threads; tid++) {
    for (int p = 0; p < top_p; p++) {
      top_acc[p] += top_acc_set[tid][p];
      top_real_acc[p] += top_real_acc_set[tid][p];
    }
  }
  for (int p = 0; p < top_p; p++) {
    // if(p>0) top_acc[p] += top_acc[p-1];
    avg_top_acc[p] = 100.0 * top_acc[p] / (double)(nr_insts);
    avg_top_real_acc[p] =
        100.0 * top_real_acc[p] / (double)(nr_insts * (p + 1));
  }
  free(XW);
  if (walltime)
    *walltime += (omp_get_wtime() - time_start);
  if (cputime)
    *cputime += ((double)(clock() - clock_start) / CLOCKS_PER_SEC);
  printf("testing on %ld instances (%g s)\n", nr_insts,
         omp_get_wtime() - time_start);
}

/*
   int main(int argc, char *argv[]){
   int k = atoi(argv[1]);
   char *input_file_name = argv[2];
   smat_t R;
//testset_t T;
load(input_file_name,R,T);

R.save_binary_to_file("mogo.smat");

double norm2 = 0;
for(size_t idx = 0; idx < R.nnz; idx++)
norm2 += R.val[idx]*R.val[idx];
printf("m = %ld n = %ld nnz = %ld norm2 %.10g\n", R.rows, R.cols, R.nnz, norm2);

smat_t Rt;
Rt.load_binary("mogo.smat");
norm2 = 0;
double norm22 = 0;
for(size_t idx = 0; idx < Rt.nnz; idx++) {
norm2 += Rt.val[idx]*Rt.val[idx];
norm22 += Rt.val_t[idx]*Rt.val_t[idx];
}
printf("m = %ld n = %ld nnz = %ld norm2 %.10g norm22 %.10g\n", Rt.rows, Rt.cols,
Rt.nnz, norm2, norm22);


return 0;
}

*/

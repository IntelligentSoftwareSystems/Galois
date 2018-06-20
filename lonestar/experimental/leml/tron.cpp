/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stddef.h>
#include "tron.h"
#include <omp.h>
#ifdef EXP_DOALL_GALOIS
#include "galois/Galois.h"
#endif

#ifndef min
template <class T>
static inline T min(T x, T y) {
  return (x < y) ? x : y;
}
#endif

#ifndef max
template <class T>
static inline T max(T x, T y) {
  return (x > y) ? x : y;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);
*/

/*
#define dnrm2_ dnrm2_my
#define ddot_ ddot_my
#define daxpy_ daxpy_my
#define dscal_ dscal_my
#define ptrdiff_t int
*/

extern double dnrm2_(ptrdiff_t*, double*, ptrdiff_t*);
extern double ddot_(ptrdiff_t*, double*, ptrdiff_t*, double*, ptrdiff_t*);
extern int daxpy_(ptrdiff_t*, double*, double*, ptrdiff_t*, double*,
                  ptrdiff_t*);
extern int dscal_(ptrdiff_t*, double*, double*, ptrdiff_t*);
extern int dcopy_(ptrdiff_t* n, double* sx, ptrdiff_t* incx, double* sy,
                  ptrdiff_t* incy);

#ifdef __cplusplus
}
#endif

static void default_print(const char* buf) {
  fputs(buf, stdout);
  fflush(stdout);
}

void TRON::info(const char* fmt, ...) {
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);
  (*tron_print_string)(buf);
}

TRON::TRON(const function* fun_obj, double eps, int max_iter, int max_cg_iter) {
  this->fun_obj     = const_cast<function*>(fun_obj);
  this->eps         = eps;
  this->max_iter    = max_iter;
  this->max_cg_iter = max_cg_iter;
  tron_print_string = default_print;
}

TRON::~TRON() {}

void TRON::tron(double* w, bool set_w_to_zero) {
  // Parameters for updating the iterates.
  double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

  // Parameters for updating the trust region size delta.
  double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

  ptrdiff_t n = fun_obj->get_nr_variable();
  int i, cg_iter;
  double delta, snorm, one = 1.0;
  double alpha, f, fnew, prered, actred, gs;
  int search = 1, iter = 1;
  ptrdiff_t inc = 1;
  double* s     = new double[n];
  double* r     = new double[n];
  double* w_new = new double[n];
  double* g     = new double[n];

  if (set_w_to_zero)
    for (i = 0; i < n; i++)
      w[i] = 0;

  f = fun_obj->fun(w);
  fun_obj->grad(w, g);
  // delta = dnrm2_(&n, g, &inc);
  delta         = sqrt(ddot_(&n, g, &inc, g, &inc));
  double gnorm1 = delta;
  double gnorm  = gnorm1;

  if (gnorm <= eps * gnorm1 || gnorm1 < eps)
    search = 0;

  iter = 1;

  while (iter <= max_iter && search) {
    cg_iter = trcg(delta, g, s, r);

    // memcpy(w_new, w, sizeof(double)*n);
    dcopy_(&n, w, &inc, w_new, &inc);
    daxpy_(&n, &one, s, &inc, w_new, &inc);

    gs     = ddot_(&n, g, &inc, s, &inc);
    prered = -0.5 * (gs - ddot_(&n, s, &inc, r, &inc));
    fnew   = fun_obj->fun(w_new);

    // Compute the actual reduction.
    actred = f - fnew;

    // On the first iteration, adjust the initial step bound.
    // snorm = dnrm2_(&n, s, &inc);
    snorm = sqrt(ddot_(&n, s, &inc, s, &inc));
    if (iter == 1)
      delta = min(delta, snorm);

    // Compute prediction alpha*snorm of the step.
    if (fnew - f - gs <= 0)
      alpha = sigma3;
    else
      alpha = max(sigma1, -0.5 * (gs / (fnew - f - gs)));

    // Update the trust region bound according to the ratio of actual to
    // predicted reduction.
    if (actred < eta0 * prered)
      delta = min(max(alpha, sigma1) * snorm, sigma2 * delta);
    else if (actred < eta1 * prered)
      delta = max(sigma1 * delta, min(alpha * snorm, sigma2 * delta));
    else if (actred < eta2 * prered)
      delta = max(sigma1 * delta, min(alpha * snorm, sigma3 * delta));
    else
      delta = max(delta, min(alpha * snorm, sigma3 * delta));

    info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n",
         iter, actred, prered, delta, f, gnorm, cg_iter);

    if (actred > eta0 * prered) {
      iter++;
      // memcpy(w, w_new, sizeof(double)*n);
      dcopy_(&n, w_new, &inc, w, &inc);
      f = fnew;
      fun_obj->grad(w, g);

      // gnorm = dnrm2_(&n, g, &inc);
      gnorm = sqrt(ddot_(&n, g, &inc, g, &inc));
      if (gnorm <= eps * gnorm1)
        break;
    }
    if (f < -1.0e+32) {
      info("WARNING: f < -1.0e+32\n");
      break;
    }
    if (fabs(actred) <= 0 && prered <= 0) {
      info("WARNING: actred and prered <= 0\n");
      break;
    }
    if (fabs(actred) <= 1.0e-12 * fabs(f) &&
        fabs(prered) <= 1.0e-12 * fabs(f)) {
      info("WARNING: actred and prered too small\n");
      break;
    }
  }

  delete[] g;
  delete[] r;
  delete[] w_new;
  delete[] s;
}

int TRON::trcg(double delta, double* g, double* s, double* r) {
  int i;
  ptrdiff_t n   = fun_obj->get_nr_variable();
  ptrdiff_t inc = 1;
  double one    = 1;
  double* d     = new double[n];
  double* Hd    = new double[n];
  double rTr, rnewTrnew, alpha, beta, cgtol;

#ifdef EXP_DOALL_GALOIS
  galois::do_all(boost::counting_iterator<int>(0),
                 boost::counting_iterator<int>(n), [&](int i) {
#else
#pragma omp parallel for
  for (i = 0; i < n; i++) {
#endif
                   s[i] = 0;
                   r[i] = -g[i];
                   d[i] = r[i];
#ifdef EXP_DOALL_GALOIS
                 });
#else
  }
#endif
  // cgtol = 0.1*dnrm2_(&n, g, &inc);
  cgtol = 0.1 * sqrt(ddot_(&n, g, &inc, g, &inc));

  int cg_iter = 0;
  rTr         = ddot_(&n, r, &inc, r, &inc);
  while (1) {
    if (sqrt(ddot_(&n, r, &inc, r, &inc)) <= cgtol)
      break;
    if (max_cg_iter > 0 && cg_iter >= max_cg_iter)
      break;
    cg_iter++;
    fun_obj->Hv(d, Hd);

    alpha = rTr / ddot_(&n, d, &inc, Hd, &inc);
    daxpy_(&n, &alpha, d, &inc, s, &inc);
    if (sqrt(ddot_(&n, s, &inc, s, &inc)) > delta) {
      info("cg reaches trust region boundary\n");
      alpha = -alpha;
      daxpy_(&n, &alpha, d, &inc, s, &inc);

      double std = ddot_(&n, s, &inc, d, &inc);
      double sts = ddot_(&n, s, &inc, s, &inc);
      double dtd = ddot_(&n, d, &inc, d, &inc);
      double dsq = delta * delta;
      double rad = sqrt(std * std + dtd * (dsq - sts));
      if (std >= 0)
        alpha = (dsq - sts) / (std + rad);
      else
        alpha = (rad - std) / dtd;
      daxpy_(&n, &alpha, d, &inc, s, &inc);
      alpha = -alpha;
      daxpy_(&n, &alpha, Hd, &inc, r, &inc);
      break;
    }
    alpha = -alpha;
    daxpy_(&n, &alpha, Hd, &inc, r, &inc);
    rnewTrnew = ddot_(&n, r, &inc, r, &inc);
    beta      = rnewTrnew / rTr;
    // dscal_(&n, &beta, d, &inc);
    double tmp = beta - 1.0;
    daxpy_(&n, &tmp, d, &inc, d, &inc);
    daxpy_(&n, &one, r, &inc, d, &inc);
    rTr = rnewTrnew;
  }

  delete[] d;
  delete[] Hd;

  return (cg_iter);
}

double TRON::norm_inf(int n, double* x) {
  double dmax = fabs(x[0]);
  for (int i = 1; i < n; i++)
    if (fabs(x[i]) >= dmax)
      dmax = fabs(x[i]);
  return (dmax);
}

void TRON::set_print_string(void (*print_string)(const char* buf)) {
  tron_print_string = print_string;
}

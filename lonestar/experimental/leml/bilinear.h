#ifndef _LIBBILINEAR_H
#define _LIBBILINEAR_H

#include "smat.h"

#include <algorithm>

// Solvers
enum {
  L2R_BILINEAR_LS               = 0,
  L2R_BILINEAR_LR               = 1,
  L2R_BILINEAR_SVC              = 2,
  L2R_BILINEAR_LS_FULL          = 10,
  L2R_BILINEAR_LS_FULL_WEIGHTED = 11, // fully observed solvers
  L2R_WSABIE = 20,  // multiclass rank estimation with length constraints +
                    // random sampling
  L2R_WSABIE2 = 21, // multiclass rank estimation with norm regularizaiton +
                    // random perumuation
  L2R_WSABIE3 =
      22, // similar to L2R_WSABIE but with W = I (full rank, k = nr_feats)
  L2R_WSABIE4 =
      23, // similar to L2R_WSABIE2 but with W = I (full rank, k = nr_feats)
  L2R_WSABIE_new  = 24, // multilabel rank estimation with length constraints
  L2R_WSABIE_new2 = 25, // multilabel rank estimation with length constraints
  L2R_WSABIE_new3 = 26, // multilabel rank estimation with length constraints
};

// Reweighting Scheme
enum { RW_NO = 0, RW_OM = 1, RW_PN = 2 };

// min_W loss( Y_ij, x_i^T W h_j) + 0.5*|W|^2
class bilinear_problem {
public:
  smat_t* Y; // m*l sparse matrix
  smat_t* X; // m*f sparse matrix
  double* H; // l*k array row major H(j,s) = H[k*j+s]
  double* W; // f*k array row major W(t,s) = W[k*t+s]
  long m;    // #instances
  long f;    // #features
  long l;    // #labels
  long k;    // low-rank dimension
  bilinear_problem(smat_t* Y, smat_t* X, double* H, int k, double* W = NULL) {
    this->Y = Y;
    this->X = X;
    this->k = k;
    this->m = X->rows;
    this->f = X->cols;
    this->l = Y->cols;
    this->W = W;
    this->H = H;
  }
};

class bilinear_parameter {
public:
  int solver_type;
  double Cp, Cn;
  int max_tron_iter, max_cg_iter;
  double eps;
  int verbose;
  int threads;
  bilinear_parameter() {
    solver_type = L2R_BILINEAR_LS;
    Cp = Cn       = 5;
    max_tron_iter = 5;
    max_cg_iter   = 20;
    eps           = 0.1;
    verbose       = 0;
    threads       = 4;
  }
  void reweighting_based_on(smat_t& Y, int scheme = -1) {
    double* y = Y.val_t;
    long mn   = Y.cols * Y.rows;
    long nnz = Y.nnz, nz = mn - nnz;
    long pos = 0, neg = 0;
    for (long i = 0; i < nnz; ++i)
      if (y[i] > 0)
        pos++;
    neg = nnz - pos;
    if (scheme < 0) {
      scheme = 1;
      switch (solver_type) {
      case L2R_BILINEAR_LS_FULL:
        scheme = RW_NO; // no weighting
        break;
      case L2R_BILINEAR_LS_FULL_WEIGHTED:
        scheme = RW_OM;
        break;
      case L2R_BILINEAR_LS:
      case L2R_BILINEAR_LR:
        scheme = RW_PN;
        break;
      default:
        scheme = RW_PN;
        break;
      }
    }
    // neg=0 => zero as missing
    if (neg == 0 && scheme == RW_PN) {
      fprintf(stderr, "No negative entries. Change to RW_OM\n");
      scheme = RW_OM;
    }
    if (nz == 0 && scheme == RW_OM) {
      fprintf(stderr, "No zero entries. Change to RW_PN\n");
      scheme = RW_PN;
    }
    if (solver_type == L2R_BILINEAR_LS_FULL && scheme != RW_NO) {
      fprintf(stderr, "This solver assumes Cp = Cn. Change to no weighting\n");
      scheme = RW_NO;
    }
    switch (scheme) {
    case RW_NO:
      Cn = Cp;
      break;
    case RW_OM:
      // Cp *= (double)mn/(double)nnz;
      Cn = Cp * std::max(0.05, (double)nnz / (double)nz);
      break;
    case RW_PN:
      // Cp *= (double)nnz/(double)pos;
      Cn = Cp * std::max(0.05, (double)pos / (double)neg);
      break;
    default:
      fprintf(stderr, "ERROR: unknown reweighing scheme %d\n", scheme);
      break;
    }
  }
};

class bilinear_eval_result {
public:
  int top_p;
  double threshold;
  double *avg_top_acc, *avg_top_real_acc;
  double avg_auc, micro_F1, hamming_loss, walltime, cputime;
  // set threshold_ to 0 for the case with missing value
  bilinear_eval_result(int top_p_ = 10, double threshold_ = 0.5)
      : top_p(top_p_), threshold(threshold_), avg_top_acc(NULL),
        avg_top_real_acc(NULL) {
    avg_top_acc      = MALLOC(double, top_p);
    avg_top_real_acc = MALLOC(double, top_p);
    avg_auc = micro_F1 = hamming_loss = walltime = cputime = 0;
  }
  ~bilinear_eval_result() {
    if (avg_top_acc)
      free(avg_top_acc);
    if (avg_top_real_acc)
      free(avg_top_real_acc);
  }
};

#ifdef __cplusplus
extern "C" {
#endif

std::vector<int> subsample(int K, int N);

void bilinear_train(const bilinear_problem* prob,
                    const bilinear_parameter* param, double* W,
                    double* walltime = NULL, double* cputime = NULL);

void bilinear_predict(const bilinear_problem* prob, const double* W,
                      const double* H, bilinear_eval_result* eval_result);

// temporary use for k = f (especailly for Wsabie)
void bilinear_predict_full(const bilinear_problem* prob, const double* H,
                           bilinear_eval_result* eval_result);

void bilinear_predict_with_clustering(
    const bilinear_problem* prob, const double* W, const double* H, int top_p,
    int nr_cluster, double* cluster_idx, double* centroids, double* avg_top_acc,
    double* avg_top_real_acc, double* walltime, double* cputime);

#ifdef __cplusplus
}
#endif

#endif /* _LIBBILINEAR_H */

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

#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Reduction.h"
#include "galois/ParallelSTL.h"
#include "galois/substrate/PaddedLock.h"
#include "Lonestar/BoilerPlate.h"
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#ifdef HAS_EIGEN
#include <Eigen/Sparse>
#endif

#include <iostream>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <vector>

/**           CONFIG           **/

static const char* const name =
    "Stochastic Gradient Descent for Linear Support Vector Machines";
static const char* const desc = "Implements a linear support vector machine "
                                "using stochastic gradient descent";
static const char* const url = "sgdsvm";

enum class UpdateType {
  Wild,
  WildOrig,
  ReplicateByThread,
  ReplicateBySocket,
  CycleBySocket,
  Staleness
};

enum class AlgoType {
  SGDL1,
  SGDL2,
  SGDLR,
  DCDL1,
  DCDL2,
  DCDLR,
  CDLasso,
  LeastSquares,
  GLMNETL1RLR,
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputTrainGraphFilename(
    cll::Positional, cll::desc("<training graph input file>"), cll::Required);
static cll::opt<std::string> inputTrainLabelFilename(
    cll::Positional, cll::desc("<training label input file>"), cll::Required);
static cll::opt<std::string> inputTestGraphFilename(
    cll::Positional, cll::desc("<testing graph input file>"), cll::Required);
static cll::opt<std::string> inputTestLabelFilename(
    cll::Positional, cll::desc("<testing label input file>"), cll::Required);
static cll::opt<double>
    creg("creg", cll::desc("the regularization parameter C"), cll::init(1.0));
static cll::opt<bool>
    shuffleSamples("shuffle", cll::desc("shuffle samples between iterations"),
                   cll::init(true));
static cll::opt<unsigned> SEED("seed", cll::desc("random seed"),
                               cll::init(~0U));
static cll::opt<bool> printObjective("printObjective",
                                     cll::desc("print objective value"),
                                     cll::init(true));
static cll::opt<bool> printAccuracy("printAccuracy",
                                    cll::desc("print accuracy value"),
                                    cll::init(true));
static cll::opt<double> tol("tol", cll::desc("convergence tolerance"),
                            cll::init(0.1));
static cll::opt<unsigned>
    maxIterations("maxIterations", cll::desc("maximum number of iterations"),
                  cll::init(1000));
static cll::opt<bool>
    useshrink("useshrink",
              cll::desc("use rhinking strategy for coordinate descent"),
              cll::init(true));
static cll::opt<unsigned> fixedIterations(
    "fixedIterations",
    cll::desc("run specific number of iterations, ignoring convergence"),
    cll::init(0));
static cll::opt<UpdateType> updateType(
    "update", cll::desc("Update type:"),
    cll::values(clEnumValN(UpdateType::Wild, "wild",
                           "unsynchronized (default)"),
                clEnumValN(UpdateType::WildOrig, "wildorig", "unsynchronized"),
                clEnumValN(UpdateType::ReplicateByThread, "replicateByThread",
                           "thread replication"),
                clEnumValN(UpdateType::ReplicateBySocket, "replicateBySocket",
                           "socket replication"),
                clEnumValN(UpdateType::CycleBySocket, "cycleBySocket",
                           "socket replication"),
                clEnumValN(UpdateType::Staleness, "staleness", "stale reads"),
                clEnumValEnd),
    cll::init(UpdateType::Wild));
static cll::opt<AlgoType> algoType(
    "algo", cll::desc("Algorithm:"),
    cll::values(
        clEnumValN(AlgoType::SGDL1, "sgdl1",
                   "primal stochastic gradient descent hinge loss (default)"),
        clEnumValN(AlgoType::SGDL2, "sgdl2",
                   "primal stochastic gradient descent square-hinge loss"),
        clEnumValN(AlgoType::SGDLR, "sgdlr",
                   "primal stochastic gradient descent logistic regression"),
        clEnumValN(AlgoType::DCDL1, "dcdl1",
                   "Dual coordinate descent hinge loss"),
        clEnumValN(AlgoType::DCDL2, "dcdl2",
                   "Dual coordinate descent square-hinge loss"),
        clEnumValN(AlgoType::DCDLR, "dcdlr",
                   "Dual coordinate descent logistic regression"),
        clEnumValN(AlgoType::CDLasso, "cdlasso", "Coordinate descent Lasso"),
        clEnumValN(AlgoType::GLMNETL1RLR, "l1rlr",
                   "new GLMENT for L1-regularized Logistic Regression"),
#ifdef HAS_EIGEN
        clEnumValN(AlgoType::LeastSquares, "ls",
                   "minimize l2 norm of residual as least squares problem"),
#endif
        clEnumValEnd),
    cll::init(AlgoType::SGDL1));

/**          DATA TYPES        **/

typedef struct Node {
  // double w; //weight - relevant for variable nodes
  union {
    int field; // variable nodes - (1/variable count), sample nodes - label
    int y;     // sample node label
  };

  union {
    double alpha;   // sample node
    double w;       // variable node
    double exp_wTx; // sample node: newGLMNET;
  };

  union {
    double QD;
    double xTx;
    double xTd;   // sample node: newGLMNET
    double Hdiag; // variable node: newGLMNET
  };

  union {
    double alpha2;
    double b;
    double exp_wTx_new; // sample node: newGLMNET;
    double Grad;        // variable node: newGLMNET
  };

  union {
    double D;   // sample node: newGLMNET
    double wpd; // variable node: newGLMNET
  };

  union {
    double tau;       // sample node: used for newGLMNET
    double xjneg_sum; // variable node: newGLMNET
  };

  Node() : w(0.0), field(0), QD(0.0), alpha2(0.0), D(0.0), tau(0.0) {}
} Node;

using Graph =
    galois::graphs::LC_CSR_Graph<Node, double>::with_out_of_line_lockable<
        true>::type ::with_numa_alloc<true>::type;
using GNode = Graph::GraphNode;
typedef galois::InsertBag<GNode> Bag;

/**         CONSTANTS AND PARAMETERS       **/
unsigned NUM_SAMPLES        = 0;
unsigned NUM_VARIABLES      = 0;
unsigned NUM_TEST_SAMPLES   = 0;
unsigned NUM_TEST_VARIABLES = 0;

unsigned variableNodeToId(GNode variable_node) {
  return ((unsigned)variable_node) - NUM_SAMPLES;
}

galois::substrate::PerThreadStorage<double*> thread_weights;
galois::substrate::PerSocketStorage<double*> socket_weights;
galois::LargeArray<double> old_weights;

// undef to test specialization for dense feature space
//#define DENSE
//#define DENSE_NUM_FEATURES 500

template <typename T, UpdateType UT>
class DiffractedCollection {
  galois::substrate::PerThreadStorage<unsigned> counts;
  galois::substrate::PerThreadStorage<T*> thread;
  galois::substrate::PerSocketStorage<T*> socket;
  galois::LargeArray<T> old;
  size_t size;
  unsigned num_threads;
  unsigned num_sockets;
  unsigned block_size;

  template <typename GetFn>
  void doMerge(const GetFn& getFn) {
    bool byThread =
        UT == UpdateType::ReplicateByThread || UT == UpdateType::Staleness;
    double* local = byThread ? *thread.getLocal() : *socket.getLocal();
    unsigned n    = byThread ? num_threads : num_sockets;

    if (true || UT == UpdateType::CycleBySocket ||
        UT == UpdateType::ReplicateBySocket) {
      galois::do_all(boost::counting_iterator<unsigned>(0),
                     boost::counting_iterator<unsigned>(size), [&](unsigned i) {
                       if (byThread) {
                         int index = i % num_threads;
                         local[i]  = (*thread.getRemote(index))[i];
                       } else {
                         int index = i % num_sockets;
                         local[i]  = (*socket.getRemoteByPkg(index))[i];
                       }
                       auto& v = getFn(i);
                       if (UT == UpdateType::Staleness)
                         v = (old[i] = local[i]);
                       else
                         v = local[i];
                     });
    } else {
      galois::do_all(boost::counting_iterator<unsigned>(0),
                     boost::counting_iterator<unsigned>(size), [&](unsigned i) {
                       for (unsigned j = 1; j < n; j++) {
                         double o = byThread ? (*thread.getRemote(j))[i]
                                             : (*socket.getRemoteByPkg(j))[i];
                         local[i] += o;
                       }
                       local[i] /= n;
                       auto& v = getFn(i);
                       if (UT == UpdateType::Staleness)
                         v = (old[i] = local[i]);
                       else
                         v = local[i];
                     });
    }

    galois::on_each([&](unsigned tid, unsigned total) {
      switch (UT) {
      case UpdateType::Staleness:
      case UpdateType::ReplicateByThread:
        if (tid)
          std::copy(local, local + size, *thread.getLocal());
        break;
      case UpdateType::CycleBySocket:
      case UpdateType::ReplicateBySocket:
        if (tid && galois::substrate::getThreadPool().isLeader(tid))
          std::copy(local, local + size, *socket.getLocal());
        break;
      default:
        abort();
      }
    });
  }

public:
  DiffractedCollection(size_t n) : size(n), block_size(0) {
    num_threads = galois::getActiveThreads();
    num_sockets = galois::substrate::getThreadPool().getCumulativeMaxSocket(
                      num_threads - 1) +
                  1;

    if (UT == UpdateType::Staleness)
      old.create(n);

    if (UT == UpdateType::Staleness || UT == UpdateType::ReplicateByThread) {
      galois::on_each([n, this](unsigned tid, unsigned total) {
        T* p               = new T[n];
        *thread.getLocal() = p;
        std::fill(p, p + n, 0);
      });
    } else if (UT == UpdateType::ReplicateBySocket ||
               UT == UpdateType::CycleBySocket) {
      galois::on_each([n, this](unsigned tid, unsigned total) {
        if (galois::substrate::getThreadPool().isLeader(tid)) {
          T* p               = new T[n];
          *socket.getLocal() = p;
          std::fill(p, p + n, 0);
        }
      });
    }

    if (UT == UpdateType::CycleBySocket) {
      block_size = size / (num_threads * num_sockets);
    }
  }

  struct Accessor {
    T* rptr;
    T* wptr;
    T* bptr;

    Accessor(T* p) : rptr(p), wptr(p), bptr(nullptr) {}
    Accessor(T* p1, T* p2) : rptr(p1), wptr(p2), bptr(nullptr) {}
    Accessor(T* p1, T* p2, T* p3) : rptr(p1), wptr(p2), bptr(p3) {}

    T& read(T& addr, ptrdiff_t x) {
      switch (UT) {
      case UpdateType::WildOrig:
      case UpdateType::Wild:
        return addr;
      default:
        return rptr[x];
      }
    }
    T& write(T& addr, ptrdiff_t x) {
      switch (UT) {
      case UpdateType::WildOrig:
      case UpdateType::Wild:
        return addr;
      default:
        return wptr[x];
      }
    }
    void writeBig(T& addr, const T& value) {
      if (!bptr)
        return;
      assert(rptr == wptr);
      bptr[std::distance(wptr, &addr)] = value;
    }
  };

  Accessor get() {
    // XXX
    if ((UT == UpdateType::ReplicateBySocket ||
         UT == UpdateType::CycleBySocket) &&
        num_sockets > 1) {
      unsigned tid       = galois::substrate::ThreadPool::getTID();
      unsigned my_socket = galois::substrate::ThreadPool::getSocket();
      if (UT == UpdateType::ReplicateBySocket || block_size == 0) {
        unsigned next = (my_socket + 1) % num_sockets;
        return Accessor{*socket.getLocal(), *socket.getLocal(),
                        *socket.getRemoteByPkg(next)};
      } else if (UT == UpdateType::CycleBySocket) {
        unsigned v     = (*counts.getLocal())++;
        unsigned index = v / block_size;
        unsigned cur   = (my_socket + index) % num_sockets;
        unsigned next  = (my_socket + index + 1) % num_sockets;
        return Accessor{*socket.getRemoteByPkg(cur),
                        *socket.getRemoteByPkg(cur),
                        *socket.getRemoteByPkg(next)};
      } else {
        abort();
      }
    }

    switch (UT) {
    case UpdateType::Wild:
    case UpdateType::WildOrig:
    case UpdateType::Staleness:
      return Accessor{&old[0], *thread.getLocal()};
    case UpdateType::CycleBySocket:
    case UpdateType::ReplicateBySocket:
      return Accessor{*socket.getLocal()};
    case UpdateType::ReplicateByThread:
      return Accessor{*thread.getLocal()};
    default:
      abort();
    }
  }

  template <typename GetFn>
  void merge(const GetFn& getFn) {
    switch (UT) {
    case UpdateType::Wild:
    case UpdateType::WildOrig:
      return;
    case UpdateType::CycleBySocket:
    case UpdateType::Staleness:
    case UpdateType::ReplicateBySocket:
    case UpdateType::ReplicateByThread:
      return doMerge(getFn);
    default:
      abort();
    }
  }
};

template <UpdateType UT>
struct LogisticRegression {
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  double learningRate;
  galois::GAccumulator<size_t>& bigUpdates;
  bool has_other;

  AlgoType alg_type;
  double* QD;
  double* xTx;
  double* alpha;
  double innereps;
  size_t* newton_iter;

  double diag;
  double C;

#ifdef DENSE
  Node* baseNodeData;
  ptrdiff_t edgeOffset;
  double* baseEdgeData;
#endif
  LogisticRegression(Graph& _g, double _lr, galois::GAccumulator<size_t>& b)
      : g(_g), learningRate(_lr), bigUpdates(b) {
    has_other = galois::substrate::getThreadPool().getCumulativeMaxSocket(
                    galois::getActiveThreads() - 1) > 1;
    alg_type = AlgoType::SGDL1;
#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset   = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }

  LogisticRegression(Graph& _g, galois::GAccumulator<size_t>& b, double* _alpha,
                     double* _qd, bool useL1Loss)
      : g(_g), bigUpdates(b), alpha(_alpha), QD(_qd) {
    has_other = galois::substrate::getThreadPool().getCumulativeMaxSocket(
                    galois::getActiveThreads() - 1) > 1;

    diag = 0.5 / creg;
    C    = std::numeric_limits<double>::max();
    if (useL1Loss) {
      diag = 0;
      C    = creg;
    }

    if (useL1Loss)
      alg_type = AlgoType::DCDL1;
    else
      alg_type = AlgoType::DCDL2;
#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset   = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }

  LogisticRegression(Graph& _g, galois::GAccumulator<size_t>& b, double* _alpha,
                     double* _xTx, double _innereps, size_t* _newton_iter)
      : g(_g), bigUpdates(b), alpha(_alpha), xTx(_xTx), innereps(_innereps),
        newton_iter(_newton_iter) {
    has_other = galois::substrate::getThreadPool().getCumulativeMaxSocket(
                    galois::getActiveThreads() - 1) > 1;

    C        = creg;
    alg_type = AlgoType::DCDLR;

#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset   = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    Node& sample_data = g.getData(n);

    // Gather
    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data =
          g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      double weight = var_data.w;
      dot += weight * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
    }

    int label = sample_data.field;
    double d  = 0.0;

    // For Coordinate Descent
    if (alg_type == AlgoType::DCDLR) { // Added by Rofu
      int yi      = label > 0 ? 1 : -1;
      double ywTx = dot * yi, xisq = sample_data.xTx;
      double alpha[2] = {sample_data.alpha, sample_data.alpha2};
      // double &alpha = sample_data.alpha, &alpha2 = sample_data.alpha2;
      double a = xisq, b = ywTx;

      // Decide to minimize g_1(z) or g_2(z)
      int ind1 = 0, ind2 = 1, sign = 1;
      if (0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0) {
        ind1 = 1;
        ind2 = 0;
        sign = -1;
      }

      //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 +
      //  sign*b(z-alpha_old)
      double alpha_old = alpha[ind1];
      double z         = alpha_old;
      if (C - z < 0.5 * C)
        z = 0.1 * z;
      double gp = a * (z - alpha_old) + sign * b + log(z / (C - z));

      // Newton method on the sub-problem
      const double eta         = 0.1; // xi in the paper
      const int max_inner_iter = 100;
      int inner_iter           = 0;
      while (inner_iter <= max_inner_iter) {
        if (fabs(gp) < innereps)
          break;
        double gpp  = a + C / (C - z) / z;
        double tmpz = z - gp / gpp;
        if (tmpz <= 0)
          z *= eta;
        else // tmpz in (0, C)
          z = tmpz;
        gp = a * (z - alpha_old) + sign * b + log(z / (C - z));
        inner_iter++;
      }
      *newton_iter += inner_iter;
      alpha[ind1]        = z;
      alpha[ind2]        = C - z;
      sample_data.alpha  = alpha[0];
      sample_data.alpha2 = alpha[1];
      d                  = sign * (z - alpha_old) * yi;
      if (d == 0)
        return;
    }

    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data =
          g.getData(variable_node, galois::MethodFlag::UNPROTECTED);

      double delta = -d * g.getEdgeData(edge_it);
      var_data.w -= delta;
    }
  }
};

typedef struct {
  std::vector<double> QD;
  std::vector<double> alpha;
  double diag;
  double C;

  double PG;
  double PGmax_old;
  double PGmin_old;
  // double PGmax_new;
  // double PGmin_new;
  size_t active_size;
  galois::GReduceMax<double> PGmax_new;
  galois::GReduceMin<double> PGmin_new;

  std::vector<bool> isactive;
} DCD_parameters;

template <UpdateType UT>
struct linearSVM_DCD {
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  galois::GAccumulator<size_t>& bigUpdates;
  Bag* next_bag;
  bool has_other;

  double diag;
  double C;
  DCD_parameters* params;

#ifdef DENSE
  Node* baseNodeData;
  ptrdiff_t edgeOffset;
  double* baseEdgeData;
#endif

  linearSVM_DCD(Graph& _g, galois::GAccumulator<size_t>& b,
                DCD_parameters* _params, Bag* _next_bag = NULL)
      : g(_g), bigUpdates(b), diag(_params->diag), C(_params->C),
        params(_params), next_bag(_next_bag) {
    has_other = galois::substrate::getThreadPool().getCumulativeMaxSocket(
                    galois::getActiveThreads() - 1) > 1;

#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset   = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {

    // if (!params->isactive[n]) return;
    Node& sample_data = g.getData(n);

    // Gather
    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data =
          g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      double weight = var_data.w;
      dot += weight * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
    }

    int label = sample_data.field;
    double d  = 0.0;

    double& nowalpha = sample_data.alpha;
    double a         = nowalpha;
    double G         = dot * label - 1 + nowalpha * diag;
    double PG        = 0;

    if (useshrink == true) {
      if (a == 0) {
        if (G > params->PGmax_old) {
          //(params->isactive)[n] = false;
          // params->active_size--;
          return;
        } else if (G < 0) {
          PG = G;
        }
      } else if (a == C) {
        if (G < params->PGmin_old) {
          //(params->isactive)[n] = false;
          // params->active_size--;
          return;
        } else if (G > 0) {
          PG = G;
        }
      } else {
        PG = G;
      }
      next_bag->push(n);

      params->PGmax_new.update(PG);
      params->PGmin_new.update(PG);
      // params->PGmax_new = std::max(params->PGmax_new, PG);
      // params->PGmin_new = std::min(params->PGmin_new, PG);
    }

    nowalpha = std::min(std::max(a - G / sample_data.QD, 0.0), C);
    d        = (nowalpha - a) * label;
    if (d == 0.0)
      return;

    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data =
          g.getData(variable_node, galois::MethodFlag::UNPROTECTED);

      double delta = -d * g.getEdgeData(edge_it);
      var_data.w -= delta;
    }
  }
};

// SGD for linearSVM and logistic regression -- only for wild update
struct LinearSGDWild {
  Graph& g;
  double learningRate;
  bool has_other;

  LinearSGDWild(Graph& _g, double _lr) : g(_g), learningRate(_lr) {}

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    Node& sample_data = g.getData(n);
    double invcreg    = 1.0 / creg;
    // Gather
    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data =
          g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      double weight = var_data.w;
      dot += weight * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
    }

    int label = sample_data.field;

    bool bigUpdate = true;
    if ((algoType == AlgoType::SGDL1) || (algoType == AlgoType::SGDL2))
      bigUpdate = label * dot < 1;

    double d = 0.0;
    if (algoType == AlgoType::SGDL1)
      d = 1.0;
    else if (algoType == AlgoType::SGDL2)
      d = 2 * (1 - label * dot);
    else if (algoType == AlgoType::SGDLR)
      d = 1 / (1 + exp(dot * label));

    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data =
          g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      int varCount   = var_data.field;
      double rfactor = var_data.QD;
      double delta   = 0;
      if (bigUpdate == true)
        delta = learningRate *
                (var_data.w * rfactor -
                 d * label *
                     g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED));
      else
        delta = learningRate * (var_data.w * rfactor);
      var_data.w -= delta;
    }
  }
};

// SGD for linearSVM and logistic regression
template <UpdateType UT>
struct LinearSGD {
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  DiffractedCollection<double, UT>& dstate;
  double learningRate;

#ifdef DENSE
  Node* baseNodeData;
  ptrdiff_t edgeOffset;
  double* baseEdgeData;
#endif

  LinearSGD(Graph& _g, DiffractedCollection<double, UT>& d, double _lr)
      : g(_g), dstate(d), learningRate(_lr) {
#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset   = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    galois::PerIterAllocTy& alloc = ctx.getPerIterAlloc();

    // Store edge data in iteration-local temporary to reduce cache misses
#ifdef DENSE
    const ptrdiff_t size = DENSE_NUM_FEATURES;
#else
    ptrdiff_t size = std::distance(g.edge_begin(n), g.edge_end(n));
#endif
    // regularized factors
    double* rfactors = (double*)alloc.allocate(sizeof(double) * size);
    // document weights
    double* dweights = (double*)alloc.allocate(sizeof(double) * size);
    // model weights
    double* mweights = (double*)alloc.allocate(sizeof(double) * size);
    // write destinations
    double** wptrs = (double**)alloc.allocate(sizeof(double*) * size);

    // Gather
    size_t cur = 0;
    double dot = 0.0;
    auto d     = dstate.get();
#ifdef DENSE
    double* myEdgeData = &baseEdgeData[size * n];
    for (cur = 0; cur < size;) {
      int varCount = baseNodeData[cur].field;
#else
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data =
          g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      int varCount  = var_data.field;
#endif

      double weight;
#ifdef DENSE
      weight = d.read(baseNodeData[cur].w, cur + edgeOffset);
#else
      wptrs[cur]    = &d.write(var_data.w, variableNodeToId(variable_node));
      weight        = d.read(var_data.w, variableNodeToId(variable_node));
#endif
      mweights[cur] = weight;
#ifdef DENSE
      dweights[cur] = myEdgeData[cur];
#else
      dweights[cur] = g.getEdgeData(edge_it);
#endif
      if (UT == UpdateType::WildOrig) {
        rfactors[cur] = (creg * varCount);
      } else {
        rfactors[cur] = mweights[cur] / (creg * varCount);
      }
      dot += mweights[cur] * dweights[cur];
      cur += 1;
    }

    Node& sample_data = g.getData(n);
    int label         = sample_data.field;

    bool bigUpdate = algoType == AlgoType::SGDLR || label * dot < 1;
    double dd      = 0.0;
    // TODO account for these ops
    if (algoType == AlgoType::SGDL1)
      dd = 1.0;
    else if (algoType == AlgoType::SGDL2)
      dd = 2 * (1 - label * dot);
    else if (algoType == AlgoType::SGDLR)
      dd = 1 / (1 + exp(dot * label));

    for (cur = 0; cur < size; ++cur) {
      double delta;
      if (UT == UpdateType::WildOrig) {
        if (bigUpdate)
          delta = learningRate *
                  (*wptrs[cur] / rfactors[cur] - dd * label * dweights[cur]);
        else
          delta = *wptrs[cur] / rfactors[cur];
      } else {
        if (bigUpdate)
          delta = learningRate * (rfactors[cur] - dd * label * dweights[cur]);
        else
          delta = rfactors[cur];
      }
#ifdef DENSE
      d.write(baseNodeData[cur].w, cur + edgeOffset) = mweights[cur] - delta;
#else
      if (UT == UpdateType::WildOrig) {
        double v    = *wptrs[cur] - delta;
        *wptrs[cur] = v;
        if (bigUpdate)
          d.writeBig(*wptrs[cur], v);
      } else {
        double v    = mweights[cur] - delta;
        *wptrs[cur] = v;
        if (bigUpdate)
          d.writeBig(*wptrs[cur], v);
      }
#endif
    }
  }
};

typedef struct {
  double Gmax_old;
  double Gnorm1_init;
  galois::GReduceMax<double> Gmax_new;
  galois::GAccumulator<double> Gnorm1_new;
} CD_parameters;

// Primal CD for Lasso
template <UpdateType UT>
struct Lasso_CD {
  Graph& g;
  double lambda;
  CD_parameters* params;
  Bag* next_bag;

  enum { COMP, UPDATE };
  struct Task {
    int type;
    GNode target;
    // int target;
    double arg;
  };

  struct Initializer : public std::unary_function<GNode, Task> {
    Task operator()(GNode arg) const { return {COMP, arg, 0.0}; }
  };

  Lasso_CD(Graph& _g, CD_parameters* _params, Bag* _next_bag = NULL)
      : g(_g), params(_params), next_bag(_next_bag) {
    lambda = 0.5 / creg;
  }

  //  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
  void operator()(const Task& t, galois::UserContext<Task>& ctx) {
    if (t.type == COMP) {
      do_comp(t, ctx);
    } else if (t.type == UPDATE) {
      do_update(t, ctx);
    }
  }

  void do_update(const Task& t, galois::UserContext<Task>& ctx) {
    Node& var_data = g.getData(t.target);
    var_data.alpha += t.arg;
  }

  void do_comp(const Task& t, galois::UserContext<Task>& ctx) {
    GNode n        = t.target;
    Node& var_data = g.getData(n);
    double& w      = var_data.w;

    if (var_data.xTx == 0.0)
      return;
    double wold = w;
    double ainv = var_data.xTx;

    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode sample_node = g.getEdgeDst(edge_it);
      Node& sample_data =
          g.getData(sample_node, galois::MethodFlag::UNPROTECTED);
      double r = sample_data.alpha;
      dot += r * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
    }

    double violation = 0;
    if (useshrink) {
      double G  = dot * 2 * creg;
      double Gp = G + 1;
      double Gn = G - 1;
      if (wold == 0) {
        if (Gp < 0)
          violation = -Gp;
        else if (Gn > 0)
          violation = Gn;
        else if (Gp > (params->Gmax_old / NUM_SAMPLES) &&
                 Gn < -(params->Gmax_old / NUM_SAMPLES))
          return;
      } else if (wold > 0) {
        violation = std::fabs(Gp);
      } else {
        violation = std::fabs(Gn);
      }

      params->Gmax_new.update(violation);
      params->Gnorm1_new += violation;
      next_bag->push(n);
    }
    double z       = wold - dot * ainv;
    double lambda1 = lambda * ainv;

    double wnew = std::max(std::fabs(z) - lambda1, 0.0);
    if (z < 0)
      wnew = -wnew;
    double delta = wnew - wold;
    if (std::fabs(delta) > 1e-12) {
      w = wnew;
      for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
        GNode sample_node = g.getEdgeDst(edge_it);
        //			Node& sample_data = g.getData(sample_node,
        //galois::MethodFlag::UNPROTECTED);

        double update_val =
            delta * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
        ctx.push(Task{UPDATE, sample_node, update_val});

        //			sample_data.alpha += delta*g.getEdgeData(edge_it,
        //galois::MethodFlag::UNPROTECTED);
      }
    }
  }
};

double getDualObjective(Graph& g, const std::vector<GNode>& trainingSamples,
                        double* diag, const std::vector<double>& alpha) {
  // 0.5 * w^Tw + C * sum_i [max(0, 1 - y_i * w^T * x_i)]^2
  galois::GAccumulator<double> objective;

  galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode n) {
    Node& data = g.getData(n);
    int label  = g.getData(n).field;
    objective += alpha[n] * (alpha[n] * diag[label + 1] - 2);
  });

  galois::do_all(boost::counting_iterator<size_t>(0),
                 boost::counting_iterator<size_t>(NUM_VARIABLES),
                 [&](size_t i) {
                   double v = g.getData(i + NUM_SAMPLES).w;
                   objective += v * v;
                 });
  return objective.reduce();
}

void printParameters(const std::vector<GNode>& trainingSamples,
                     const std::vector<GNode>& testingSamples) {
  std::cout << "Input Train graph file: " << inputTrainGraphFilename << "\n";
  std::cout << "Input Train label file: " << inputTrainLabelFilename << "\n";
  std::cout << "Input Test graph file: " << inputTestGraphFilename << "\n";
  std::cout << "Input Test label file: " << inputTestLabelFilename << "\n";
  std::cout << "Threads: " << galois::getActiveThreads() << "\n";
  std::cout << "Train Samples: " << NUM_SAMPLES << "\n";
  std::cout << "Test Samples: " << NUM_TEST_SAMPLES << "\n";
  std::cout << "Variables: " << NUM_VARIABLES << "\n";
  std::cout << "Test Variables: " << NUM_TEST_VARIABLES << "\n";
  std::cout << "Training samples: " << trainingSamples.size() << "\n";
  std::cout << "Testing samples: " << testingSamples.size() << "\n";
  std::cout << "Algo type: ";
  switch (algoType) {
  case AlgoType::SGDL1:
    std::cout << "primal stocahstic gradient descent for hinge Loss";
    break;
  case AlgoType::SGDL2:
    std::cout << "primal stocahstic gradient descent for square-hinge Loss";
    break;
  case AlgoType::SGDLR:
    std::cout << "primal stocahstic gradient descent for logistic Loss";
    break;
  case AlgoType::DCDL1:
    std::cout << "dual coordinate descent hinge loss parallel";
    break;
  case AlgoType::DCDL2:
    std::cout << "dual coordinate descent square-hinge loss parallel";
    break;
  case AlgoType::DCDLR:
    std::cout << "dual coordinate descent logsitic regression";
    break;
  case AlgoType::CDLasso:
    std::cout << "coordinate descent lasso";
    break;
  case AlgoType::GLMNETL1RLR:
    std::cout << "new GLMNET l1r-lr";
    break;
  case AlgoType::LeastSquares:
    std::cout << "least squares";
    break;
  default:
    abort();
  }
  std::cout << "\n";

  std::cout << "Update type: ";
  switch (updateType) {
  case UpdateType::Wild:
    std::cout << "wild";
    break;
  case UpdateType::WildOrig:
    std::cout << "wild orig";
    break;
  case UpdateType::ReplicateByThread:
    std::cout << "replicate by thread";
    break;
  case UpdateType::ReplicateBySocket:
    std::cout << "replicate by socket";
    break;
  case UpdateType::CycleBySocket:
    std::cout << "cycle by socket";
    break;
  case UpdateType::Staleness:
    std::cout << "stale reads";
    break;
  default:
    abort();
  }
  std::cout << "\n";
}

void initializeVariableCounts(Graph& g) {
  for (auto n : g) {
    for (auto edge_it : g.out_edges(n)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& data          = g.getData(variable_node);
      data.field++; // increase count of variable occurrences
    }
  }
  for (auto n : g) {
    if (n >= NUM_SAMPLES) {
      Node& data = g.getData(n);
      if (data.field != 0)
        data.QD = 1.0 / (creg * data.field);
    }
  }
}

unsigned loadb(Graph& g, std::string filename) {
  std::ifstream infile(filename);

  unsigned sample_id;
  double bi;
  int num_labels = 0;
  while (infile >> sample_id >> bi) {
    g.getData(sample_id).b = bi;
    ++num_labels;
  }

  return num_labels;
}

unsigned loadLabels(Graph& g, std::string filename) {
  std::ifstream infile(filename);

  unsigned sample_id;
  int label;
  int num_labels = 0;
  while (infile >> sample_id >> label) {
    if (label > 0)
      g.getData(sample_id).field = 1;
    else
      g.getData(sample_id).field = -1;
    ++num_labels;
  }

  return num_labels;
}

size_t getNumCorrect(Graph& g_test, std::vector<GNode>& testing_samples,
                     Graph& g_train) {
  galois::GAccumulator<size_t> correct;

  std::vector<double> w_vec(NUM_VARIABLES);
  galois::do_all(g_train.begin() + NUM_SAMPLES, g_train.end(), [&](GNode n) {
    Node& data             = g_train.getData(n);
    w_vec[n - NUM_SAMPLES] = data.w;
  });

  galois::do_all(testing_samples.begin(), testing_samples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g_test.getData(n);
    int label  = data.field;
    for (auto edge_it : g_test.out_edges(n)) {
      GNode variable_node = g_test.getEdgeDst(edge_it);
      if ((variable_node - NUM_TEST_SAMPLES) < NUM_VARIABLES) {
        double weight = g_test.getEdgeData(edge_it);
        sum += w_vec[variable_node - NUM_TEST_SAMPLES] * weight;
      }
    }

    if (sum <= 0.0 && label == -1) {
      correct += 1;
    } else if (sum > 0.0 && label == 1) {
      correct += 1;
    }
  });

  return correct.reduce();
}

double getTestRMSE(Graph& g_test, std::vector<GNode>& testing_samples,
                   Graph& g_train) {
  galois::GAccumulator<double> square_err;

  std::vector<double> w_vec(NUM_VARIABLES);
  galois::do_all(g_train.begin() + NUM_SAMPLES, g_train.end(), [&](GNode n) {
    Node& data             = g_train.getData(n);
    w_vec[n - NUM_SAMPLES] = data.w;
  });

  double wnorm = 0;
  for (auto i : w_vec)
    wnorm += i * i;
  //  printf("wnorm: %lf, umvariables: %d\n", wnorm, NUM_VARIABLES);

  galois::do_all(testing_samples.begin(), testing_samples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g_test.getData(n);
    double b   = data.b;
    for (auto edge_it : g_test.out_edges(n)) {
      GNode variable_node = g_test.getEdgeDst(edge_it);
      if ((variable_node - NUM_TEST_SAMPLES) < NUM_VARIABLES) {
        double weight = g_test.getEdgeData(edge_it);
        sum += w_vec[variable_node - NUM_TEST_SAMPLES] * weight;
      }
    }
    square_err += (sum - b) * (sum - b);
  });

  double err = square_err.reduce();
  //  printf("err: %lf\n", err);
  return sqrt(err / NUM_TEST_SAMPLES);
  //  return correct.reduce();
}

double getPrimalObjective(Graph& g, const std::vector<GNode>& trainingSamples) {
  // 0.5 * w^Tw + C * sum_i loss(w^T * x_i, y_i)
  //// 0.5 * w^Tw + C * sum_i [max(0, 1 - y_i * w^T * x_i)]^2
  galois::GAccumulator<double> objective;

  galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g.getData(n);
    int label  = data.field;
    double b;
    if (algoType == AlgoType::CDLasso)
      b = data.b;
    for (auto edge_it : g.out_edges(n)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& data          = g.getData(variable_node);
      double weight       = g.getEdgeData(edge_it);
      sum += data.w * weight;
    }

    double o;
    if ((algoType == AlgoType::DCDL2) || (algoType == AlgoType::SGDL2)) {
      o = std::max(0.0, 1 - label * sum);
      o = o * o;
    } else if ((algoType == AlgoType::DCDLR) || (algoType == AlgoType::SGDLR) ||
               (algoType == AlgoType::GLMNETL1RLR))
      o = log(1 + exp(-label * sum));
    else if ((algoType == AlgoType::DCDL1) || (algoType == AlgoType::SGDL1))
      o = std::max(0.0, 1 - label * sum);
    else if (algoType == AlgoType::CDLasso)
      o = (sum - b) * (sum - b);
    objective += o;
  });

  galois::GAccumulator<double> norm;
  galois::do_all(
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(NUM_VARIABLES), [&](size_t i) {
        double v = g.getData(i + NUM_SAMPLES).w;
        if (algoType == AlgoType::CDLasso || algoType == AlgoType::GLMNETL1RLR)
          norm += std::fabs(v);
        else
          norm += 0.5 * v * v;
      });
  return objective.reduce() * creg + norm.reduce();
}

void runDCD(Graph& g_train, Graph& g_test, std::mt19937& gen,
            std::vector<GNode>& trainingSamples,
            std::vector<GNode>& testingSamples) {
  galois::TimeAccumulator accumTimer;
  accumTimer.start();

  // allocate storage for weights from previous iteration
  old_weights.create(NUM_VARIABLES);
  if (updateType == UpdateType::ReplicateByThread ||
      updateType == UpdateType::Staleness) {
    galois::on_each([](unsigned tid, unsigned total) {
      double* p                  = new double[NUM_VARIABLES];
      *thread_weights.getLocal() = p;
      std::fill(p, p + NUM_VARIABLES, 0);
    });
  }
  if (updateType == UpdateType::ReplicateBySocket) {
    galois::on_each([](unsigned tid, unsigned total) {
      if (galois::substrate::getThreadPool().isLeader(tid)) {
        double* p                  = new double[NUM_VARIABLES];
        *socket_weights.getLocal() = p;
        std::fill(p, p + NUM_VARIABLES, 0);
      }
    });
  }

  galois::StatTimer DcdTime("DcdTime");

  // Initialization for DCD
  double diag[] = {0.5 / creg, 0, 0.5 / creg};
  double ub[]   = {std::numeric_limits<double>::max(), 0,
                 std::numeric_limits<double>::max()};
  if (algoType == AlgoType::DCDL1 or algoType == AlgoType::DCDLR) {
    diag[0] = 0;
    diag[2] = 0;
    ub[0]   = creg;
    ub[2]   = creg;
  }

  DCD_parameters params;
  params.C                   = ub[0];
  params.diag                = diag[0];
  params.QD                  = std::vector<double>(NUM_SAMPLES, 0);
  params.alpha               = std::vector<double>(NUM_SAMPLES, 0);
  std::vector<double>& QD    = params.QD;
  std::vector<double>& alpha = params.alpha;
  params.PGmax_old           = std::numeric_limits<double>::max();
  params.PGmin_old           = std::numeric_limits<double>::lowest();
  params.isactive            = std::vector<bool>(NUM_SAMPLES, true);
  params.active_size         = NUM_SAMPLES;

  Bag bags[2];
  Bag *cur_bag = &bags[0], *next_bag = &bags[1];

  // For LR
  double innereps     = 1e-2;
  double innereps_min = 1e-8; // min(1e-8, eps);

  if (algoType == AlgoType::DCDLR)
    alpha.resize(2 * NUM_SAMPLES, 0);

  printf("asdfasdfasdf\n");
  // initialize model w to zero
  //	galois::do_all(boost::counting_iterator<size_t>(0),
  //boost::counting_iterator<size_t>(NUM_VARIABLES), [&](size_t i) {
  //			g_train.getData(i+NUM_VARIABLES).w = 0;
  //		});

  printf("asdfasdfasdf\n");
  galois::StatTimer QDTime("QdTime");
  QDTime.start();
  std::vector<double> xTx(NUM_SAMPLES);
  // for (auto ii = g.begin(), ei = g.begin() + NUM_SAMPLES; ii != ei; ++ii) {
  auto ts_begin = trainingSamples.begin();
  auto ts_end   = trainingSamples.end();

  printf("asdfasdfasdf\n");

  for (auto ii = ts_begin, ei = ts_end; ii != ei; ++ii) {
    int& label     = g_train.getData(*ii).field;
    auto& nodedata = g_train.getData(*ii);
    cur_bag->push(*ii);

    if (label != 1 && label != -1) {
      label = label <= 0 ? -1 : 1;
    }
    if (algoType == AlgoType::DCDLR) {
      alpha[2 * (*ii)]     = std::min(0.001 * ub[label + 1], 1e-8);
      alpha[2 * (*ii) + 1] = ub[label + 1] - alpha[2 * (*ii)];
      nodedata.alpha       = std::min(0.001 * ub[label + 1], 1e-8);
      nodedata.alpha2      = ub[label + 1] - nodedata.alpha2;
    } else {
      alpha[*ii]     = 0;
      nodedata.alpha = 0;
    }

    for (auto edge : g_train.out_edges(*ii)) {
      double val = g_train.getEdgeData(edge);
      // xTx[*ii] += val*val;
      nodedata.xTx += val * val;
      auto variable_node = g_train.getEdgeDst(edge);
      Node& data         = g_train.getData(variable_node);
      data.w += label * nodedata.alpha * val;
      /*
      if(algoType == AlgoType::DCDLR) {
          data.w += label*alpha[2*(*ii)]*val;
      } else {
          data.w += label*alpha[*ii]*val;
      }
      */
    }

    if (algoType == AlgoType::DCDLR)
      nodedata.QD = nodedata.xTx + diag[label + 1];
    // QD[*ii] = diag[label+1] + xTx[*ii];
    // g_train.getData(*ii).QD=QD[*ii];
  }
  QDTime.stop();
  printf("QDTIME~!!!!!!!!!! %lf\n", QDTime.get() / 1e3);

  unsigned iterations = maxIterations;
  double minObj       = std::numeric_limits<double>::max();
  if (fixedIterations)
    iterations = fixedIterations;

  bool is_terminate = false;
  std::vector<GNode> active_set;

  for (unsigned iter = 1; iter <= iterations && is_terminate == false; ++iter) {
    DcdTime.start();

    // params.PGmax_new = std::numeric_limits<double>::lowest();
    // params.PGmin_new = std::numeric_limits<double>::max();
    params.PGmax_new.reset();
    params.PGmin_new.reset();

    // include shuffling time in the time taken per iteration
    // also: not parallel

    if (useshrink) {
      active_set.clear();
      for (auto& gg : *cur_bag)
        active_set.push_back(gg);
    }
    if (shuffleSamples) {
      if (useshrink) {
        std::shuffle(active_set.begin(), active_set.end(), gen);
      } else {
        std::shuffle(trainingSamples.begin(), trainingSamples.end(), gen);
      }
    }

    size_t newton_iter = 0;
    auto ts_begin      = trainingSamples.begin();
    auto ts_end        = trainingSamples.end();
    auto ln            = galois::loopname("LinearSVM");
    if (algoType == AlgoType::DCDLR)
      ln = galois::loopname("LogisticRegression");

    auto wl = galois::wl<galois::worklists::PerSocketChunkFIFO<32>>();
    //		auto wl = galois::wl<galois::worklists::StableIterator<true> >();
    galois::GAccumulator<size_t> bigUpdates;

    printf("pgmax_old: %lf, pgmin_old: %lf\n", params.PGmax_old,
           params.PGmin_old);

    UpdateType type = updateType;
    switch (type) {
    case UpdateType::Wild:
    case UpdateType::WildOrig:
      if (algoType == AlgoType::DCDLR) {
        galois::for_each(ts_begin, ts_end,
                         LogisticRegression<UpdateType::Wild>(
                             g_train, bigUpdates, &alpha[0], &xTx[0], innereps,
                             &newton_iter),
                         ln, wl);
      } else if (useshrink) {
        cur_bag->clear();
        printf("active set size: %zu\n", active_set.size());
        galois::for_each(active_set.begin(), active_set.end(),
                         linearSVM_DCD<UpdateType::Wild>(g_train, bigUpdates,
                                                         &params, cur_bag),
                         ln, wl);
      } else {
        galois::for_each(
            ts_begin, ts_end,
            linearSVM_DCD<UpdateType::Wild>(g_train, bigUpdates, &params), ln,
            wl);
      }
      break;
    case UpdateType::ReplicateBySocket:
      //        galois::for_each(ts_begin, ts_end,
      //        linearSVM<UpdateType::ReplicateBySocket>(g, learning_rate,
      //        bigUpdates), ln, wl); break;
    case UpdateType::ReplicateByThread:
      //        galois::for_each(ts_begin, ts_end,
      //        linearSVM<UpdateType::ReplicateByThread>(g, learning_rate,
      //        bigUpdates), ln, wl); break;
    case UpdateType::Staleness:
      //        galois::for_each(ts_begin, ts_end,
      //        linearSVM<UpdateType::Staleness>(g, learning_rate, bigUpdates),
      //        ln, wl);
      printf("ERROR: only support Wild updates\n");
      return;
      break;
    default:
      abort();
    }

    if (useshrink == true) {
      size_t active_size = std::distance(cur_bag->begin(), cur_bag->end());
      double PGmax_local = params.PGmax_new.reduce();
      double PGmin_local = params.PGmin_new.reduce();
      printf("now dual gap: %lf, active_size: %zu\n", PGmax_local - PGmin_local,
             active_size);
      // if ( params.PGmax_new - params.PGmin_new <= tol )
      if (PGmax_local - PGmin_local <= tol) {
        if (active_size == NUM_SAMPLES)
          is_terminate = true;
        else {
          /*
          params.active_size = NUM_SAMPLES;
          for ( int i=0 ; i<NUM_SAMPLES ; i++ )
              params.isactive[i] = true;
              */

          cur_bag->clear();
          for (auto ii = ts_begin; ii != ts_end; ii++)
            cur_bag->push(*ii);
          params.PGmax_old = std::numeric_limits<double>::max();
          params.PGmin_old = std::numeric_limits<double>::lowest();
        }
      } else {
        params.PGmax_old = PGmax_local;
        params.PGmin_old = PGmin_local;
        printf("Pgmaxlocal: %.17g, pgmin_local: %.17g\n", PGmax_local,
               PGmin_local);
        if (params.PGmax_old <= 1e-300) {
          printf("hihi\n");
          params.PGmax_old = std::numeric_limits<double>::max();
        }
        if (params.PGmin_old >= 0)
          params.PGmin_old = std::numeric_limits<double>::lowest();
      }
    }

    DcdTime.stop();

    if (algoType == AlgoType::DCDLR)
      if (newton_iter <= trainingSamples.size() / 10)
        innereps = std::max(1e-8, 0.1 * innereps);
    size_t numBigUpdates = bigUpdates.reduce();

    /*
    //swap weights from past iteration and this iteration
    if (type != UpdateType::Wild && type != UpdateType::WildOrig) {
        bool byThread = type == UpdateType::ReplicateByThread || type ==
    UpdateType::Staleness; double *localw = byThread ?
    *thread_weights.getLocal() : *socket_weights.getLocal(); unsigned
    num_threads = galois::getActiveThreads(); unsigned num_sockets =
    galois::runtime::LL::getMaxSocketForThread(num_threads-1) + 1;
        galois::do_all(boost::counting_iterator<unsigned>(0),
    boost::counting_iterator<unsigned>(NUM_VARIABLES), [&](unsigned i) {
                unsigned n = byThread ? num_threads : num_sockets;
                for (unsigned j = 1; j < n; j++) {
                double o = byThread ?
                (*thread_weights.getRemote(j))[i] :
                (*socket_weights.getRemoteByPkg(j))[i];
                localw[i] += o;
                }
                localw[i] /=  n;
                GNode variable_node = (GNode) (i + NUM_SAMPLES);
                Node& var_data = g.getData(variable_node,
    galois::MethodFlag::UNPROTECTED); var_data.w = localw[i]; old_weights[i] =
    var_data.w;
                });
        galois::on_each([&](unsigned tid, unsigned total) {
                switch (type) {
                case UpdateType::Staleness:
                case UpdateType::ReplicateByThread:
                if (tid)
                std::copy(localw, localw + NUM_VARIABLES,
    *thread_weights.getLocal()); break; case UpdateType::ReplicateBySocket: if
    (tid && galois::runtime::LL::isSocketLeader(tid)) std::copy(localw, localw +
    NUM_VARIABLES, *socket_weights.getLocal()); break; default: abort();
                }
                });
    }
*/
    accumTimer.stop();

    std::cout << "iter " << iter << " walltime " << DcdTime.get() / 1e3;

    if (printObjective)
      std::cout << " f " << getPrimalObjective(g_train, trainingSamples);
    if (printAccuracy)
      std::cout << " accuracy "
                << getNumCorrect(g_test, testingSamples, g_train) /
                       (double)testingSamples.size();

    std::cout << "\n";

    if (algoType != AlgoType::DCDLR) {
      // Verify whether w = \sum_i alpha_i x_i
      std::vector<double> realw(NUM_SAMPLES);
      for (auto ii = g_train.begin(), ei = g_train.begin() + NUM_SAMPLES;
           ii != ei; ++ii) {
        double alphai = alpha[*ii];
        int& label    = g_train.getData(*ii).field;
        if (label != 1 && label != -1) {
          label = label <= 0 ? -1 : 1;
        }
        for (auto edge : g_train.out_edges(*ii)) {
          double val          = g_train.getEdgeData(edge);
          GNode variable_node = g_train.getEdgeDst(edge);
          realw[variableNodeToId(variable_node)] += val * label * alphai;
        }
      }

      double diff = 0;
      for (auto ii = 0; ii < NUM_VARIABLES; ii++) {
        Node& var_data = g_train.getData(ii + NUM_SAMPLES);
        double w       = var_data.w;
        diff += (realw[ii] - w) * (realw[ii] - w);
      }
      printf("diff: %lf\n", diff);
    }
  }

  if (!fixedIterations)
    std::cout << "Failed to converge\n";
}

template <UpdateType UT>
void runPrimalSgd_(Graph& g_train, Graph& g_test, std::mt19937& gen,
                   std::vector<GNode>& trainingSamples,
                   std::vector<GNode>& testingSamples) {
  galois::TimeAccumulator accumTimer;
  accumTimer.start();

  DiffractedCollection<double, UT> dstate(NUM_VARIABLES);

  galois::StatTimer sgdTime("SgdTime");

  unsigned iterations = maxIterations;
  double minObj       = std::numeric_limits<double>::max();
  if (fixedIterations)
    iterations = fixedIterations;

  for (unsigned iter = 1; iter <= iterations; ++iter) {
    sgdTime.start();

    // include shuffling time in the time taken per iteration
    // also: not parallel
    if (shuffleSamples)
      std::shuffle(trainingSamples.begin(), trainingSamples.end(), gen);

    double learning_rate = 30 / (100.0 + iter);
    auto ts_begin        = trainingSamples.begin();
    auto ts_end          = trainingSamples.end();
    auto ln              = galois::loopname("LinearSVM");
    auto wl = galois::wl<galois::worklists::PerSocketChunkFIFO<32>>();

    if (UT == UpdateType::Wild)
      galois::for_each(ts_begin, ts_end, LinearSGDWild(g_train, learning_rate),
                       ln, wl);
    else
      galois::for_each(ts_begin, ts_end,
                       LinearSGD<UT>(g_train, dstate, learning_rate), ln, wl);

    sgdTime.stop();

    dstate.merge([&g_train](ptrdiff_t x) -> double& {
      return g_train.getData(x + NUM_SAMPLES).w;
    });

    accumTimer.stop();

    std::cout << "iter " << iter << " walltime " << sgdTime.get() / 1e3;

    double obj = getPrimalObjective(g_train, trainingSamples);
    if (printObjective)
      std::cout << " f " << obj;
    if (printAccuracy)
      std::cout << " accuracy "
                << getNumCorrect(g_test, testingSamples, g_train) /
                       (double)testingSamples.size();

    std::cout << "\n";

    if (!fixedIterations) {
      if (std::fabs((obj - minObj) / minObj) < tol) {
        std::cout << "Converged in " << iter << " iterations.\n";
        return;
      }
      minObj = std::min(obj, minObj);
    }
  }

  if (!fixedIterations)
    std::cout << "Failed to converge\n";
}

void runPrimalSgd(Graph& g_train, Graph& g_test, std::mt19937& gen,
                  std::vector<GNode>& trainingSamples,
                  std::vector<GNode>& testingSamples) {
  switch (updateType) {
  case UpdateType::Wild:
    return runPrimalSgd_<UpdateType::Wild>(g_train, g_test, gen,
                                           trainingSamples, testingSamples);
  case UpdateType::WildOrig:
    return runPrimalSgd_<UpdateType::WildOrig>(g_train, g_test, gen,
                                               trainingSamples, testingSamples);
  case UpdateType::ReplicateBySocket:
    return runPrimalSgd_<UpdateType::ReplicateBySocket>(
        g_train, g_test, gen, trainingSamples, testingSamples);
  case UpdateType::CycleBySocket:
    return runPrimalSgd_<UpdateType::CycleBySocket>(
        g_train, g_test, gen, trainingSamples, testingSamples);
  case UpdateType::ReplicateByThread:
    return runPrimalSgd_<UpdateType::ReplicateByThread>(
        g_train, g_test, gen, trainingSamples, testingSamples);
  case UpdateType::Staleness:
    return runPrimalSgd_<UpdateType::Staleness>(
        g_train, g_test, gen, trainingSamples, testingSamples);
  default:
    abort();
  }
}

void runCD(Graph& g_train, Graph& g_test, std::mt19937& gen,
           std::vector<GNode>& trainingSamples,
           std::vector<GNode>& testingSamples) {
  galois::TimeAccumulator accumTimer;
  accumTimer.start();
  galois::StatTimer CdTime("CdTime");

  unsigned iterations = maxIterations;
  if (fixedIterations)
    iterations = fixedIterations;

  bool is_terminate = false;

  std::vector<GNode> variables(g_train.begin() + NUM_SAMPLES, g_train.end());

  for (auto ii = variables.begin(), ei = variables.end(); ii != ei; ii++) {
    auto& nodedata = g_train.getData(*ii);
    nodedata.w     = 0;
    nodedata.xTx   = 0;

    for (auto edge : g_train.out_edges(*ii)) {
      double val = g_train.getEdgeData(edge);
      nodedata.xTx += val * val;
    }
    nodedata.xTx = 1.0 / nodedata.xTx;
  }

  for (auto ii = trainingSamples.begin(), ei = trainingSamples.end(); ii != ei;
       ii++) {
    auto& nodedata = g_train.getData(*ii);
    auto& label    = g_train.getData(*ii).b;
    nodedata.alpha = label * (-1);
  }

  CD_parameters params;
  params.Gmax_old = std::numeric_limits<double>::max();

  Bag cur_bag;
  std::vector<GNode> active_set;
  for (auto ii = variables.begin(), ei = variables.end(); ii != ei; ii++) {
    cur_bag.push(*ii);
  }

  for (unsigned iter = 1; iter <= iterations && is_terminate == false; ++iter) {
    CdTime.start();

    params.Gmax_new.reset();
    params.Gnorm1_new.reset();

    if (useshrink) {
      active_set.clear();
      for (auto& gg : cur_bag) {
        active_set.push_back(gg);
      }
    }
    if (shuffleSamples) {
      if (useshrink) {
        std::shuffle(active_set.begin(), active_set.end(), gen);
      } else {
        std::shuffle(variables.begin(), variables.end(), gen);
      }
    }

    auto ln = galois::loopname("PrimalCD");
    auto wl = galois::wl<galois::worklists::PerSocketChunkLIFO<32>>();
    //		auto wl = galois::wl<galois::worklists::StableIterator<true> >();

    UpdateType type = updateType;
    switch (type) {
    case UpdateType::Wild:
    case UpdateType::WildOrig:
      if (useshrink) {
        cur_bag.clear();
        printf("active set size: %zu\n", active_set.size());
      } else {
        galois::for_each(
            boost::transform_iterator<Lasso_CD<UpdateType::Wild>::Initializer,
                                      boost::counting_iterator<int>>(
                NUM_SAMPLES),
            boost::transform_iterator<Lasso_CD<UpdateType::Wild>::Initializer,
                                      boost::counting_iterator<int>>(
                NUM_SAMPLES + NUM_VARIABLES),
            Lasso_CD<UpdateType::Wild>(g_train, &params), ln, wl);
        //                  	galois::for_each(variables.begin(),
        //                  variables.end(), Lasso_CD<UpdateType::Wild>(g_train,
        //                  &params), ln, wl);
      }
      break;
    case UpdateType::ReplicateBySocket:
      //        galois::for_each(ts_begin, ts_end,
      //        linearSVM<UpdateType::ReplicateBySocket>(g, learning_rate,
      //        bigUpdates), ln, wl); break;
    case UpdateType::ReplicateByThread:
      //        galois::for_each(ts_begin, ts_end,
      //        linearSVM<UpdateType::ReplicateByThread>(g, learning_rate,
      //        bigUpdates), ln, wl); break;
    case UpdateType::Staleness:
      //        galois::for_each(ts_begin, ts_end,
      //        linearSVM<UpdateType::Staleness>(g, learning_rate, bigUpdates),
      //        ln, wl);
      printf("ERROR: only support Wild updates\n");
      return;
      break;
    default:
      abort();
    }

    if (useshrink == true) {
      size_t active_size  = std::distance(cur_bag.begin(), cur_bag.end());
      double Gmax_local   = params.Gmax_new.reduce();
      double Gnorm1_local = params.Gnorm1_new.reduce();
      if (iter == 1)
        params.Gnorm1_init = Gnorm1_local;
      printf("gnorm1: %lf, Gmax_new: %lf\n", Gnorm1_local, Gmax_local);
      if (Gnorm1_local <= tol * (params.Gnorm1_init)) {
        cur_bag.clear();
        for (auto ii = variables.begin(), ei = variables.end(); ii != ei; ii++)
          cur_bag.push(*ii);

        params.Gmax_old = std::numeric_limits<double>::max();
        tol             = tol * 0.1;
      } else
        params.Gmax_old = Gmax_local;
    }

    CdTime.stop();
    accumTimer.stop();

    std::cout << "iter " << iter << " walltime " << CdTime.get() / 1e3;

    std::cout.precision(10);
    if (printObjective)
      std::cout << " f " << getPrimalObjective(g_train, trainingSamples);
    if (printAccuracy)
      std::cout << " rmse " << getTestRMSE(g_test, testingSamples, g_train);

    std::cout << "\n";
  }

  if (!fixedIterations)
    std::cout << "Failed to converge\n";
}

// new GLMNET for L1R Logistic Regression
typedef struct {
  double Gmax_old;
  double Gnorm1_init;
  double QP_Gmax_old;
  galois::GReduceMax<double> Gmax_new, QP_Gmax_new;
  galois::GAccumulator<double> Gnorm1_new, QP_Gnorm1_new;
} GLMNET_parameters;

// cd for the subproblem of glmenet for L1R-LR
template <UpdateType UT>
struct glmnet_cd { // {{{
  typedef int tt_does_not_need_aborts;
  typedef GLMNET_parameters param_t;
  Graph& g_train;
  DiffractedCollection<double, UT>& dstate;
  GLMNET_parameters& params;
  Bag& cd_bag;
  size_t nr_samples;
  double nu;

  glmnet_cd(Graph& _g, DiffractedCollection<double, UT>& d, param_t& _p,
            Bag& bag, size_t _nr_samples)
      : g_train(_g), dstate(d), params(_p), cd_bag(bag),
        nr_samples(_nr_samples), nu(1e-12) {}

  void operator()(GNode feat_j, galois::UserContext<GNode>& ctx) {
    auto& j_data = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
    auto& H      = j_data.Hdiag;
    auto& Grad_j = j_data.Grad;
    auto& wpd_j  = j_data.wpd;
    auto& w_j    = j_data.w;
    double G     = Grad_j + (wpd_j - w_j) * nu;
    auto d       = dstate.get();

    for (auto& edge :
         g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
      auto& x_ij  = g_train.getEdgeData(edge);
      auto dst    = g_train.getEdgeDst(edge);
      auto& ddata = g_train.getData(dst, galois::MethodFlag::UNPROTECTED);
      G += x_ij * ddata.D * d.read(ddata.xTd, dst);
    }
    double Gp        = G + 1;
    double Gn        = G - 1;
    double violation = 0;
    if (wpd_j == 0) {
      if (Gp < 0)
        violation = -Gp;
      else if (Gn > 0)
        violation = Gn;
      else if (Gp > params.QP_Gmax_old / nr_samples &&
               Gn < -params.QP_Gmax_old / nr_samples) {
        return;
      }
    } else if (wpd_j > 0)
      violation = fabs(Gp);
    else
      violation = fabs(Gn);
    cd_bag.push(feat_j);
    params.QP_Gmax_new.update(violation);
    params.QP_Gnorm1_new.update(violation);
    double z = 0;
    if (Gp < H * wpd_j)
      z = -Gp / H;
    else if (Gn > H * wpd_j)
      z = -Gn / H;
    else
      z = -wpd_j;
    if (fabs(z) < 1.0e-12)
      return;
    z = std::min(std::max(z, -10.0), 10.0);
    wpd_j += z;
    for (auto& edge :
         g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
      auto& x_ij  = g_train.getEdgeData(edge);
      auto dst    = g_train.getEdgeDst(edge);
      auto& ddata = g_train.getData(dst, galois::MethodFlag::UNPROTECTED);
      double& l   = d.read(ddata.xTd, dst);
      double v    = l + x_ij * z;
      d.write(ddata.xTd, dst) = v;
      // d.writeBig(l, v);
      // d.write(ddata.xTd, dst) += x_ij*z;
    }
  }
}; // }}}

struct glmnet_qp_construct { // {{{
  typedef GLMNET_parameters param_t;
  Graph& g_train;
  GLMNET_parameters& params;
  Bag& cd_bag;
  size_t nr_samples;
  double nu;
  glmnet_qp_construct(Graph& _g, param_t& _p, Bag& bag, size_t _nr_samples)
      : g_train(_g), params(_p), cd_bag(bag), nr_samples(_nr_samples),
        nu(1e-12) {}
  void operator()(GNode feat_j, galois::UserContext<GNode>& ctx) {
    auto& j_data  = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
    auto& w_j     = j_data.w;
    auto& Hdiag_j = j_data.Hdiag;
    auto& Grad_j  = j_data.Grad;
    auto& xjneg_sum_j = j_data.xjneg_sum;
    Hdiag_j           = nu;
    Grad_j            = 0;
    double tmp        = 0;
    for (auto& edge :
         g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
      auto x_ij  = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
      auto& self = g_train.getData(g_train.getEdgeDst(edge),
                                   galois::MethodFlag::UNPROTECTED);
      Hdiag_j += x_ij * x_ij * self.D;
      tmp += x_ij * self.tau;
    }
    Grad_j = -tmp + xjneg_sum_j;

    double Gp        = Grad_j + 1;
    double Gn        = Grad_j - 1;
    double violation = 0;
    if (w_j == 0) {
      if (Gp < 0)
        violation = -Gp;
      else if (Gn > 0)
        violation = Gn;
      // outer-level shrinking
      else if (Gp > params.Gmax_old / nr_samples &&
               Gn < -params.Gmax_old / nr_samples) {
        return;
      }

    } else if (w_j > 0)
      violation = fabs(Gp);
    else
      violation = fabs(Gn);
    cd_bag.push(feat_j);
    params.Gmax_new.update(violation);
    params.Gnorm1_new.update(violation);
  }
}; // }}}

template <UpdateType UT>
void runGLMNET_(Graph& g_train, Graph& g_test, std::mt19937& gen,
                std::vector<GNode>& trainingSamples,
                std::vector<GNode>& testingSamples) { // {{{
  galois::TimeAccumulator accumTimer;
  accumTimer.start();
  // galois::runtime::getThreadPool().burnPower(numThreads);

  DiffractedCollection<double, UT> dstate(NUM_SAMPLES);

  galois::StatTimer glmnetTime("GLMNET_Time");
  galois::StatTimer cdTime("CD_Time");
  galois::StatTimer FirstTime("First_Time");
  galois::StatTimer SecondTime("Second_Time");
  galois::StatTimer ThirdTime("Third_Time");
  galois::StatTimer ActiveSetTime("ActiveSet_Time");

  unsigned max_newton_iter = fixedIterations ? fixedIterations : maxIterations;
  unsigned max_cd_iter     = 50;
  unsigned max_num_linesearch = 20;
  double nu                   = 1e-12;
  double inner_eps            = 0.01;
  double sigma                = 0.01;

  double C[3] = {creg, 0, creg};

  std::vector<GNode> variables(g_train.begin() + NUM_SAMPLES, g_train.end());

  // initialization {{{
  galois::do_all(trainingSamples.begin(), trainingSamples.end(),
                 [&](GNode inst_node) {
                   auto& self   = g_train.getData(inst_node);
                   self.y       = self.y > 0 ? 1 : -1;
                   self.exp_wTx = 0.0;
                 });
  double w_norm = 0;
  galois::do_all(variables.begin(), variables.end(), [&](GNode feat_j) {
    auto& j_data        = g_train.getData(feat_j);
    double& w_j         = j_data.w;
    double& wpd_j       = j_data.wpd;
    double& xjneg_sum_j = j_data.xjneg_sum;
    w_norm += fabs(w_j);
    wpd_j       = w_j;
    xjneg_sum_j = 0;
    for (auto edge :
         g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
      auto x_ij  = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
      auto& self = g_train.getData(g_train.getEdgeDst(edge),
                                   galois::MethodFlag::UNPROTECTED);
      self.exp_wTx += w_j * x_ij;
      if (self.y == -1)
        xjneg_sum_j += creg * x_ij;
    }
  });
  galois::GAccumulator<double> xx;
  galois::do_all(variables.begin(), variables.end(), [&](GNode feat_j) {
    xx += g_train.getData(feat_j).xjneg_sum;
  });
  double cc = creg;
  printf("creg %lf init xx %lf\n", cc, xx.reduce());

  galois::do_all(
      trainingSamples.begin(), trainingSamples.end(), [&](GNode inst_node) {
        auto& self =
            g_train.getData(inst_node, galois::MethodFlag::UNPROTECTED);
        self.exp_wTx   = exp(self.exp_wTx);
        double tau_tmp = 1.0 / (1.0 + self.exp_wTx);
        self.tau       = creg * tau_tmp;
        self.D         = creg * self.exp_wTx * tau_tmp * tau_tmp;
      }); //}}}

  int newton_iter = 0;
  Bag cur_bag; // used for outerlevel active set
  std::vector<GNode> active_set;
  GLMNET_parameters params;
  params.Gmax_old   = std::numeric_limits<double>::max();
  size_t nr_samples = trainingSamples.size();

  while (newton_iter < max_newton_iter) {
    glmnetTime.start();

    cur_bag.clear();
    active_set.clear();
    for (auto& feat_j : variables)
      active_set.push_back(feat_j);
    params.Gmax_new.reset();
    params.Gnorm1_new.reset();

    // if(shuffleSamples) std::shuffle(active_set.begin(), active_set.end(),
    // gen);

    FirstTime.start();

    // Compute Newton direction -- Hessian and Gradient
    auto ln = galois::loopname("GLMENT-QPconstruction");
    auto wl = galois::wl<galois::worklists::PerSocketChunkFIFO<32>>();
    galois::for_each(active_set.begin(), active_set.end(),
                     glmnet_qp_construct(g_train, params, cur_bag, nr_samples),
                     ln, wl);

    double tmp_Gnorm1_new = params.Gnorm1_new.reduce();
    if (newton_iter == 0)
      params.Gnorm1_init = tmp_Gnorm1_new;
    params.Gmax_old = params.Gmax_new.reduce();
    FirstTime.stop();

    ActiveSetTime.start();

    // Compute Newton direction -- Coordinate Descet for QP
    cdTime.start();
    params.QP_Gmax_old = std::numeric_limits<double>::max();
    galois::do_all(
        trainingSamples.begin(), trainingSamples.end(), [&](GNode& inst_node) {
          g_train.getData(inst_node, galois::MethodFlag::UNPROTECTED).xTd = 0.0;
        });
    auto init_original_active_set = [&] {
      active_set.clear();
      for (auto& feat_j : cur_bag)
        active_set.push_back(feat_j);
    };
    init_original_active_set();

    ActiveSetTime.stop();

    int original_active_size = active_set.size();
    int cd_iter              = 0;
    Bag cd_bag;

    double grad_norm = 0, H_norm = 0, xx = 0;
    SecondTime.start();
    while (cd_iter < max_cd_iter) { //{{{
      params.QP_Gmax_new.reset();
      params.QP_Gnorm1_new.reset();

      if (shuffleSamples)
        std::shuffle(active_set.begin(), active_set.end(), gen);
      auto ln = galois::loopname("GLMENT-CDiteration");
#if 1
      auto wl = galois::wl<galois::worklists::PerSocketChunkFIFO<32>>();
      galois::for_each(
          active_set.begin(), active_set.end(),
          glmnet_cd<UT>(g_train, dstate, params, cd_bag, nr_samples), ln, wl);
      dstate.merge([&g_train](ptrdiff_t x) -> double& {
        return g_train.getData(x).xTd;
      });
#else
      {
        auto wl = galois::wl<galois::worklists::StableIterator<>>();
        galois::GAccumulator<double> Gaccum;
        double nu = 1e-12;
        for (auto feat_j : active_set) {
          auto& j_data =
              g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
          auto& H      = j_data.Hdiag;
          auto& Grad_j = j_data.Grad;
          auto& wpd_j  = j_data.wpd;
          auto& w_j    = j_data.w;
          double G     = Grad_j + (wpd_j - w_j) * nu;
          galois::for_each(
              g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)
                  .begin(),
              g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED).end(),
              [&](typename Graph::edge_iterator edge,
                  galois::UserContext<typename Graph::edge_iterator>&) {
                auto& x_ij = g_train.getEdgeData(edge);
                auto dst   = g_train.getEdgeDst(edge);
                auto& ddata =
                    g_train.getData(dst, galois::MethodFlag::UNPROTECTED);
                Gaccum += x_ij * ddata.D * ddata.xTd;
              },
              ln, wl);
          G += Gaccum.reduce();
          Gaccum.reset();
          double Gp        = G + 1;
          double Gn        = G - 1;
          double violation = 0;
          if (wpd_j == 0) {
            if (Gp < 0)
              violation = -Gp;
            else if (Gn > 0)
              violation = Gn;
            else if (Gp > params.QP_Gmax_old / nr_samples &&
                     Gn < -params.QP_Gmax_old / nr_samples) {
              continue;
            }
          } else if (wpd_j > 0)
            violation = fabs(Gp);
          else
            violation = fabs(Gn);
          cd_bag.push(feat_j);
          params.QP_Gmax_new.update(violation);
          params.QP_Gnorm1_new.update(violation);
          double z = 0;
          if (Gp < H * wpd_j)
            z = -Gp / H;
          else if (Gn > H * wpd_j)
            z = -Gn / H;
          else
            z = -wpd_j;
          if (fabs(z) < 1.0e-12)
            continue;
          z = std::min(std::max(z, -10.0), 10.0);
          wpd_j += z;
          galois::for_each(
              g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)
                  .begin(),
              g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED).end(),
              [&](typename Graph::edge_iterator edge,
                  galois::UserContext<typename Graph::edge_iterator>&) {
                auto& x_ij = g_train.getEdgeData(edge);
                auto dst   = g_train.getEdgeDst(edge);
                auto& ddata =
                    g_train.getData(dst, galois::MethodFlag::UNPROTECTED);
                ddata.xTd += x_ij * z;
              },
              ln, wl);
        }
      }
#endif
      cd_iter++;
      double tmp_QP_Gmax_new   = params.QP_Gmax_new.reduce();
      double tmp_QP_Gnorm1_new = params.QP_Gnorm1_new.reduce();
      active_set.clear();
      for (auto& feat_j : cd_bag)
        active_set.push_back(feat_j);
      cd_bag.clear();
      if (tmp_QP_Gmax_new <= inner_eps * params.Gnorm1_init) {
        // inner stopping
        if (active_set.size() == original_active_size)
          break;
        // active set reactivation
        else {
          init_original_active_set();
          params.QP_Gmax_old = std::numeric_limits<double>::max();
        }
      } else {
        params.QP_Gmax_old = tmp_QP_Gmax_new;
      }
    } //}}}
    cdTime.stop();
    SecondTime.stop();

    ThirdTime.start();
    // Perform Line Search
    // {{{
    galois::GAccumulator<double> delta_acc, w_norm_acc;
    galois::do_all(variables.begin(), variables.end(), //{{{
                   [&](GNode& feat_j) {
                     auto& self = g_train.getData(
                         feat_j, galois::MethodFlag::UNPROTECTED);
                     delta_acc.update(self.Grad * (self.wpd - self.w));
                     if (self.wpd != 0)
                       w_norm_acc.update(fabs(self.wpd));
                   }); //}}}
    double w_norm_new = w_norm_acc.reduce();
    double delta      = delta_acc.reduce() + (w_norm_new - w_norm);

    galois::GAccumulator<double> tmp_acc;
    galois::do_all(
        trainingSamples.begin(), trainingSamples.end(), [&](GNode& inst_node) {
          auto& self =
              g_train.getData(inst_node, galois::MethodFlag::UNPROTECTED);
          if (self.y == -1)
            tmp_acc.update(creg * self.xTd);
        });
    double negsum_xTd = tmp_acc.reduce();

    int num_linesearch = 0;
    for (num_linesearch = 0; num_linesearch < max_num_linesearch;
         num_linesearch++) {
      double cond = w_norm_new - w_norm + negsum_xTd - sigma * delta;
      tmp_acc.reset();
      galois::do_all(trainingSamples.begin(), trainingSamples.end(),
                     [&](GNode& inst_node) {
                       auto& self = g_train.getData(
                           inst_node, galois::MethodFlag::UNPROTECTED);
                       double exp_xTd   = exp(self.xTd);
                       self.exp_wTx_new = self.exp_wTx * exp_xTd;
                       tmp_acc.update(creg * log((1 + self.exp_wTx_new) /
                                                 (exp_xTd + self.exp_wTx_new)));
                     });
      cond += tmp_acc.reduce();
      if (cond <= 0.0) {
        w_norm = w_norm_new;
        galois::do_all(variables.begin(), variables.end(), [&](GNode& feat_j) {
          auto& self = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
          self.w     = self.wpd;
        });
        galois::do_all(
            trainingSamples.begin(), trainingSamples.end(), [&](GNode& inst_i) {
              auto& self =
                  g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED);
              self.exp_wTx   = self.exp_wTx_new;
              double tau_tmp = 1 / (1 + self.exp_wTx);
              self.tau       = creg * tau_tmp;
              self.D         = creg * self.exp_wTx * tau_tmp * tau_tmp;
            });
        break;
      } else {
        w_norm_new = 0;
        tmp_acc.reset();
        galois::do_all(variables.begin(), variables.end(), [&](GNode& feat_j) {
          auto& self = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
          self.wpd   = (self.w + self.wpd) * 0.5;
          if (self.wpd != 0)
            tmp_acc.update(fabs(self.wpd));
        });
        w_norm_new = tmp_acc.reduce();
        delta *= 0.5;
        negsum_xTd *= 0.5;
        galois::do_all(
            trainingSamples.begin(), trainingSamples.end(), [&](GNode& inst_i) {
              g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED).xTd *=
                  0.5;
            });
      }
    }
    if (num_linesearch >= max_num_linesearch) {
      galois::do_all(
          trainingSamples.begin(), trainingSamples.end(), [&](GNode& inst_i) {
            g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED).exp_wTx =
                0;
          });

      galois::do_all(variables.begin(), variables.end(), [&](GNode& feat_j) {
        auto& self = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
        if (self.w != 0) {
          for (auto& edge : g_train.out_edges(feat_j)) {
            auto& x_ij =
                g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
            g_train
                .getData(g_train.getEdgeDst(edge),
                         galois::MethodFlag::UNPROTECTED)
                .exp_wTx += self.w * x_ij;
          }
        }
      });
      galois::do_all(
          trainingSamples.begin(), trainingSamples.end(), [&](GNode& inst_i) {
            auto& exp_wTx =
                g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED)
                    .exp_wTx;
            exp_wTx = exp(exp_wTx);
          });
    }
    //}}} // end of line search

    ThirdTime.stop();
    if (cd_iter == 1)
      inner_eps *= 0.25;

    newton_iter++;
    glmnetTime.stop();
    accumTimer.stop();

    printf("iter %d walltime %.1f ittime %.1f cdtime %.2f cd-iters %d "
           "firsttime %.2f secondtime %.2f thirdtime %.2f",
           newton_iter, accumTimer.get() / 1e3, glmnetTime.get() / 1e3,
           cdTime.get() / 1e3, cd_iter, FirstTime.get() / 1e3,
           SecondTime.get() / 1e3, ThirdTime.get() / 1e3);
    if (printObjective) {
      printf(" f %.6f", getPrimalObjective(g_train, trainingSamples));
    }
    if (printAccuracy) {
      printf(" accuracy %.6f", getNumCorrect(g_test, testingSamples, g_train) /
                                   (double)testingSamples.size());
    }
    printf("\n");
    accumTimer.start();
  }
  // galois::runtime::getThreadPool().beKind();
} // }}}

void runGLMNET(Graph& g_train, Graph& g_test, std::mt19937& gen,
               std::vector<GNode>& trainingSamples,
               std::vector<GNode>& testingSamples) {
  switch (updateType) {
  case UpdateType::Wild:
    return runGLMNET_<UpdateType::Wild>(g_train, g_test, gen, trainingSamples,
                                        testingSamples);
  case UpdateType::ReplicateBySocket:
    return runGLMNET_<UpdateType::ReplicateBySocket>(
        g_train, g_test, gen, trainingSamples, testingSamples);
  case UpdateType::CycleBySocket:
    return runGLMNET_<UpdateType::CycleBySocket>(
        g_train, g_test, gen, trainingSamples, testingSamples);
  case UpdateType::ReplicateByThread:
    return runGLMNET_<UpdateType::ReplicateByThread>(
        g_train, g_test, gen, trainingSamples, testingSamples);
  case UpdateType::Staleness:
    return runGLMNET_<UpdateType::Staleness>(g_train, g_test, gen,
                                             trainingSamples, testingSamples);
  default:
    abort();
  }
}

#ifdef HAS_EIGEN
void runLeastSquares(Graph& g, std::mt19937& gen,
                     std::vector<GNode>& trainingSamples,
                     std::vector<GNode>& testingSamples) {
  Eigen::SparseMatrix<double> A(NUM_SAMPLES, NUM_VARIABLES);
  {
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> triplets{g.sizeEdges()};
    {
      auto it = triplets.begin();
      for (auto n : g) {
        for (auto edge : g.out_edges(n)) {
          *it++ =
              Triplet(n, g.getEdgeDst(edge) - NUM_SAMPLES, g.getEdgeData(edge));
        }
      }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
  }

  Eigen::VectorXd b(NUM_SAMPLES);
  size_t cur = 0;
  for (auto ii = g.begin(), ei = g.begin() + NUM_SAMPLES; ii != ei; ++ii) {
    b(cur++) = g.getData(*ii).field;
  }

  // Least-squares problem: minimize ||Ax - b||_2
  // normal equation: A^T A x = A^T b
  Eigen::VectorXd x;
  // Cholesky is almost always faster but QR is more numerically stable
  if (g.sizeEdges() > 10000) {
    // Solve normal equation directly with Cholesky
    Eigen::SparseMatrix<double> AT  = A.transpose();
    Eigen::SparseMatrix<double> ATA = AT * A;
    for (int i = 0; i < ATA.rows(); ++i)
      ATA.coeffRef(i, i) += creg;
    Eigen::VectorXd ATb = AT * b;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(ATA);
    x = solver.solve(ATb);
    std::cout << "cg iterations: " << solver.iterations() << "\n";
    std::cout << "cg est error: " << solver.error() << "\n";
  } else {
    // TODO add L2 regularizer
    // Decompose normal equation
    // A = QR
    // A^T A x = A^T b
    // R^T Q^T Q R x = R^T Q^T b => ... => R x = Q^T b
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
      GALOIS_DIE("factorization failed");
    }
    Eigen::VectorXd QTb = solver.matrixQ().transpose() * b;
    int r               = solver.rank();
    Eigen::VectorXd out;
    out.resize(A.cols());
    out.topRows(r) = solver.matrixR()
                         .topLeftCorner(r, r)
                         .triangularView<Eigen::Upper>()
                         .solve(QTb.topRows(r));
    out.bottomRows(out.rows() - r).setZero();
    x = solver.colsPermutation() * out;
  }

  // Verify
  {
    for (size_t i = 0; i < NUM_VARIABLES; ++i) {
      g.getData(i + NUM_SAMPLES).w = x(i);
    }
    std::vector<GNode> allSamples(g.begin(), g.begin() + NUM_SAMPLES);
    //    size_t n = getNumCorrect(g, allSamples);
    //    std::cout << "All: " << n / (double) NUM_SAMPLES << " (" << n << "/"
    //    << NUM_SAMPLES << ")\n"; n = getNumCorrect(g, testingSamples);
    //    std::cout << "Testing: " << n / (double) testingSamples.size() << " ("
    //    << n << "/" << testingSamples.size() << ")\n";
  }
}
#endif

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  Graph g_train, g_test;
  // Load Training Data
  galois::graphs::readGraph(g_train, inputTrainGraphFilename);
  if (algoType == AlgoType::CDLasso)
    NUM_SAMPLES = loadb(g_train, inputTrainLabelFilename);
  else {
    NUM_SAMPLES = loadLabels(g_train, inputTrainLabelFilename);
  }
  if (algoType != AlgoType::CDLasso and algoType != AlgoType::GLMNETL1RLR)
    initializeVariableCounts(g_train);
  NUM_VARIABLES = g_train.size() - NUM_SAMPLES;
  assert(NUM_SAMPLES > 0 && NUM_VARIABLES > 0);

  // Load Testing Data
  galois::graphs::readGraph(g_test, inputTestGraphFilename);
  if (algoType == AlgoType::CDLasso)
    NUM_TEST_SAMPLES = loadb(g_test, inputTestLabelFilename);
  else
    NUM_TEST_SAMPLES = loadLabels(g_test, inputTestLabelFilename);
  //  NUM_TEST_SAMPLES = loadLabels(g_test, inputTestLabelFilename);
  if (algoType != AlgoType::CDLasso and algoType != AlgoType::GLMNETL1RLR)
    initializeVariableCounts(g_test);
  NUM_TEST_VARIABLES = g_test.size() - NUM_TEST_SAMPLES;
  assert(NUM_TEST_SAMPLES > 0);

  // put samples in a list and shuffle them
  std::random_device rd;
  std::mt19937 gen(SEED == ~0U ? rd() : SEED);

  std::vector<GNode> trainingSamples(g_train.begin(),
                                     g_train.begin() + NUM_SAMPLES);
  std::vector<GNode> testingSamples(g_test.begin(),
                                    g_test.begin() + NUM_TEST_SAMPLES);

  printParameters(trainingSamples, testingSamples);
  if (printAccuracy) {
    std::cout << "Initial";
    if (printAccuracy) {
      std::cout << " Accuracy: "
                << getNumCorrect(g_test, testingSamples, g_train) /
                       (double)testingSamples.size();
    }
    std::cout << "\n";
  }

  galois::StatTimer timer;
  timer.start();
  switch (algoType) {
  case AlgoType::SGDL1:
  case AlgoType::SGDL2:
  case AlgoType::SGDLR:
    runPrimalSgd(g_train, g_test, gen, trainingSamples, testingSamples);
    break;
  case AlgoType::DCDL1:
  case AlgoType::DCDL2:
  case AlgoType::DCDLR:
  case AlgoType::CDLasso:
    runCD(g_train, g_test, gen, trainingSamples, testingSamples);
    break;
  case AlgoType::GLMNETL1RLR:
    runGLMNET(g_train, g_test, gen, trainingSamples, testingSamples);
    break;
#ifdef HAS_EIGEN
//    case AlgoType::LeastSquares: runLeastSquares(g, gen, trainingSamples,
//    testingSamples); break;
#endif
  default:
    abort();
  }
  timer.stop();

  return 0;
}

// vim: set noexpandtab:

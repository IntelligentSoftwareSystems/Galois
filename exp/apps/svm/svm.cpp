/** SVM with SGD -*- C++ -*-
 * @file
 * @section License
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Stochastic gradient descent for solving linear SVM, implemented with Galois.
 *
 * @author Prad Nelluru <pradn@cs.utexas.edu>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Accumulator.h"
#include "Galois/ParallelSTL.h"
#include "Galois/Substrate/PaddedLock.h"
#include "Lonestar/BoilerPlate.h"

#ifdef HAS_EIGEN
#include <Eigen/Sparse>
#endif

#include <random>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <vector>

/**           CONFIG           **/

static const char* const name = "Stochastic Gradient Descent for Linear Support Vector Machines";
static const char* const desc = "Implements a linear support vector machine using stochastic gradient descent";
static const char* const url = "sgdsvm";

enum class UpdateType {
  Wild,
  WildOrig,
  ReplicateByThread,
  ReplicateByPackage,
  Staleness
};

enum class AlgoType {
  PrimalStochasticGradientDescent,
  LeastSquares,
  DualCoordinateDescentL1Loss,
  DualCoordinateDescentL2Loss
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputGraphFilename(cll::Positional, cll::desc("<graph input file>"), cll::Required);
static cll::opt<std::string> inputLabelFilename(cll::Positional, cll::desc("<label input file>"), cll::Required);
static cll::opt<double> creg("creg", cll::desc("the regularization parameter C"), cll::init(1.0));
static cll::opt<bool> shuffleSamples("shuffle", cll::desc("shuffle samples between iterations"), cll::init(false));
static cll::opt<unsigned> SEED("seed", cll::desc("random seed"), cll::init(~0U));
static cll::opt<bool> printObjective("printObjective", cll::desc("print objective value"), cll::init(false));
static cll::opt<bool> printAccuracy("printAccuracy", cll::desc("print accuracy value"), cll::init(true));
static cll::opt<double> fractionTraining("fractionTraining", cll::desc("fraction of samples to use for training"), cll::init(0.8));
static cll::opt<size_t> numberTraining("numberTraining", cll::desc("number of samples to use for training"), cll::init(0));
static cll::opt<double> tol("tol", cll::desc("convergence tolerance"), cll::init(0.1));
static cll::opt<unsigned> maxIterations("maxIterations", cll::desc("maximum number of iterations"), cll::init(1000));
static cll::opt<unsigned> fixedIterations("fixedIterations", cll::desc("run specific number of iterations, ignoring convergence"), cll::init(0));
static cll::opt<UpdateType> updateType("update", cll::desc("Update type:"),
  cll::values(
    clEnumValN(UpdateType::Wild, "wild", "unsynchronized (default)"),
    clEnumValN(UpdateType::WildOrig, "wildorig", "unsynchronized"),
    clEnumValN(UpdateType::ReplicateByThread, "replicateByThread", "thread replication"),
    clEnumValN(UpdateType::ReplicateByPackage, "replicateByPackage", "package replication"),
    clEnumValN(UpdateType::Staleness, "staleness", "stale reads"),
    clEnumValEnd), cll::init(UpdateType::Wild));
static cll::opt<AlgoType> algoType("algo", cll::desc("Algorithm:"),
    cll::values(
      clEnumValN(AlgoType::PrimalStochasticGradientDescent, "psgd", "primal stochastic gradient descent (default)"),
      clEnumValN(AlgoType::DualCoordinateDescentL1Loss, "dcdl1", "Dual coordinate descent L1 loss"),
      clEnumValN(AlgoType::DualCoordinateDescentL2Loss, "dcdl2", "Dual coordinate descent L2 loss"),
#ifdef HAS_EIGEN
      clEnumValN(AlgoType::LeastSquares, "ls", "minimize l2 norm of residual as least squares problem"),
#endif
      clEnumValEnd), cll::init(AlgoType::PrimalStochasticGradientDescent));

/**          DATA TYPES        **/

typedef struct Node {
  double w; //weight - relevant for variable nodes
  int field; //variable nodes - variable count, sample nodes - label
  Node(): w(0.0), field(0) { }
} Node;

using Graph = Galois::Graph::LC_CSR_Graph<Node, double>::with_out_of_line_lockable<true>::type;
using GNode = Graph::GraphNode;

/**         CONSTANTS AND PARAMETERS       **/
unsigned NUM_SAMPLES = 0;
unsigned NUM_VARIABLES = 0;

unsigned variableNodeToId(GNode variable_node) {
  return ((unsigned) variable_node) - NUM_SAMPLES;
}

//undef to test specialization for dense feature space
//#define DENSE
//#define DENSE_NUM_FEATURES 500

template<typename T, UpdateType UT>
class DiffractedCollection {
  Galois::Substrate::PerThreadStorage<T*> thread;
  Galois::Substrate::PerPackageStorage<T*> package;
  Galois::LargeArray<T> old;
  size_t size;
  unsigned num_threads;
  unsigned num_packages;

  template<typename GetFn>
  void doMerge(const GetFn& getFn) {
    bool byThread = UT == UpdateType::ReplicateByThread || UT == UpdateType::Staleness;
    double *local = byThread ? *thread.getLocal() : *package.getLocal();
    Galois::do_all(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(size), [&](unsigned i) {
      unsigned n = byThread ? num_threads : num_packages;
      for (unsigned j = 1; j < n; j++) {
        double o = byThread ?
          (*thread.getRemote(j))[i] :
          (*package.getRemoteByPkg(j))[i];
        local[i] += o;
      }
      local[i] /= n;
      auto& v = getFn(i);
      if (UT == UpdateType::Staleness)
        v = (old[i] = local[i]);
      else
        v = local[i];
    });
    Galois::on_each([&](unsigned tid, unsigned total) {
      switch (UT) {
        case UpdateType::Staleness:
        case UpdateType::ReplicateByThread:
          if (tid)
            std::copy(local, local + size, *thread.getLocal());
          break;
        case UpdateType::ReplicateByPackage:
          if (tid && Galois::Substrate::getThreadPool().isLeader(tid))
            std::copy(local, local + size, *package.getLocal());
          break;
        default: abort();
      }
    });
  }

public:
  DiffractedCollection(size_t n): size(n) {
    num_threads = Galois::getActiveThreads();
    num_packages = Galois::Substrate::getThreadPool().getCumulativeMaxPackage(num_threads-1) + 1;

    if (UT == UpdateType::Staleness)
      old.create(n);
    switch (UT) {
      case UpdateType::ReplicateByThread:
      case UpdateType::Staleness:
        Galois::on_each([n, this](unsigned tid, unsigned total) {
          T *p = new T[n];
          *thread.getLocal() = p;
          std::fill(p, p + n, 0);
        });
      case UpdateType::ReplicateByPackage:
        Galois::on_each([n, this](unsigned tid, unsigned total) {
            if (Galois::Substrate::getThreadPool().isLeader(tid)) {
            T *p = new T[n];
            *package.getLocal() = p;
            std::fill(p, p + n, 0);
          }
        });
      case UpdateType::Wild:
      case UpdateType::WildOrig:
        break;
      default: abort();
    }
  }

  struct Accessor {
    T* rptr;
    T* wptr;
    T* bptr;

    Accessor(T* p): rptr(p), wptr(p), bptr(nullptr) { }
    Accessor(T* p1, T* p2): rptr(p1), wptr(p2), bptr(nullptr) { }
    Accessor(T* p1, T* p2, T* p3): rptr(p1), wptr(p2), bptr(p3) { }

    T& read(T& addr, ptrdiff_t x) {
      switch (UT) {
        case UpdateType::WildOrig:
        case UpdateType::Wild: return addr;
        default: return rptr[x];
      }
    }
    T& write(T& addr, ptrdiff_t x) {
      switch (UT) {
        case UpdateType::WildOrig:
        case UpdateType::Wild: return addr;
        default: return wptr[x];
      }
    }
    void writeBig(T& addr, const T& value) {
      if (!bptr) return;
      assert(rptr == wptr);
      bptr[std::distance(wptr, &addr)] = value;
    }
  };

  Accessor get() {
    if (num_packages > 1 && UT == UpdateType::ReplicateByPackage) {
      unsigned tid = Galois::Substrate::ThreadPool::getTID();
      unsigned my_package = Galois::Substrate::ThreadPool::getPackage();
      unsigned next = (my_package + 1) % num_packages;
      return Accessor { *package.getLocal(), *package.getLocal(), *package.getRemoteByPkg(next) };
    }
    
    switch (UT) {
      case UpdateType::Wild:
      case UpdateType::WildOrig:
      case UpdateType::Staleness:
        return Accessor { &old[0], *thread.getLocal() };
      case UpdateType::ReplicateByPackage:
        return Accessor { *package.getLocal() };
      case UpdateType::ReplicateByThread:
        return Accessor { *thread.getLocal() };
      default:
        abort();
    }
  }

  template<typename GetFn>
  void merge(const GetFn& getFn) {
    switch (UT) {
      case UpdateType::Wild:
      case UpdateType::WildOrig:
        return;
      case UpdateType::Staleness:
      case UpdateType::ReplicateByPackage:
      case UpdateType::ReplicateByThread:
        return doMerge(getFn);
      default:
        abort();
    }
  }
};

template<UpdateType UT>
struct LinearSVM {
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  DiffractedCollection<double, UT>& dstate;
  Galois::GAccumulator<size_t>& bigUpdates;
  double learningRate;

#ifdef DENSE
  Node* baseNodeData;
  ptrdiff_t edgeOffset;
  double* baseEdgeData;
#endif

  LinearSVM(Graph& _g, DiffractedCollection<double, UT>& d, double _lr, Galois::GAccumulator<size_t>& b):
    g(_g), dstate(d), bigUpdates(b), learningRate(_lr) {
#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }
  
  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {  
    Galois::PerIterAllocTy& alloc = ctx.getPerIterAlloc();

    // Store edge data in iteration-local temporary to reduce cache misses
#ifdef DENSE
    const ptrdiff_t size = DENSE_NUM_FEATURES;
#else
    ptrdiff_t size = std::distance(g.edge_begin(n, Galois::MethodFlag::UNPROTECTED), g.edge_end(n, Galois::MethodFlag::UNPROTECTED));
#endif
    // regularized factors
    double* rfactors = (double*) alloc.allocate(sizeof(double) * size);
    // document weights
    double* dweights = (double*) alloc.allocate(sizeof(double) * size);
    // model weights
    double* mweights = (double*) alloc.allocate(sizeof(double) * size);
    // write destinations
    double** wptrs = (double**) alloc.allocate(sizeof(double*) * size);

    // Gather
    size_t cur = 0;
    double dot = 0.0;
    auto d = dstate.get();
#ifdef DENSE
    double* myEdgeData = &baseEdgeData[size * n];
    for (cur = 0; cur < size; ) {
      int varCount = baseNodeData[cur].field;
#else
    for (auto edge_it : g.out_edges(n)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node);
      int varCount = var_data.field;
#endif

      double weight;
#ifdef DENSE
      weight = d.read(baseNodeData[cur].w, cur+edgeOffset);
#else
      wptrs[cur] = &d.write(var_data.w, variableNodeToId(variable_node));
      weight = d.read(var_data.w, variableNodeToId(variable_node));
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
    int label = sample_data.field;

    bool bigUpdate = label * dot < 1;
    if (bigUpdate)
      bigUpdates += size;
    for (cur = 0; cur < size; ++cur) {
      double delta;
      if (UT == UpdateType::WildOrig) {
        if (bigUpdate)
          delta = learningRate * (*wptrs[cur]/rfactors[cur] - label * dweights[cur]);
        else
          delta = *wptrs[cur]/rfactors[cur];
      } else {
        if (bigUpdate)
          delta = learningRate * (rfactors[cur] - label * dweights[cur]);
        else
          delta = rfactors[cur];
      }
#ifdef DENSE
      d.write(baseNodeData[cur].w, cur + edgeOffset) = mweights[cur] - delta;
#else
      if (UT == UpdateType::WildOrig) {
        double v = *wptrs[cur] - delta;
        *wptrs[cur] = v;
        if (bigUpdate)
          d.writeBig(*wptrs[cur], v);
      } else {
        double v = mweights[cur] - delta;
        *wptrs[cur] = v;
        if (bigUpdate)
          d.writeBig(*wptrs[cur], v);
      }
#endif
    }
  }
};

void printParameters(const std::vector<GNode>& trainingSamples, const std::vector<GNode>& testingSamples) {
  std::cout << "Input graph file: " << inputGraphFilename << "\n";
  std::cout << "Input label file: " << inputLabelFilename << "\n";
  std::cout << "Threads: " << Galois::getActiveThreads() << "\n";
  std::cout << "Samples: " << NUM_SAMPLES << "\n";
  std::cout << "Variables: " << NUM_VARIABLES << "\n";
  std::cout << "Training samples: " << trainingSamples.size() << "\n";
  std::cout << "Testing samples: " << testingSamples.size() << "\n";
  std::cout << "Algo type: ";
  switch (algoType) {
    case AlgoType::PrimalStochasticGradientDescent: std::cout << "primal stochastic gradient descent"; break;
    case AlgoType::DualCoordinateDescentL1Loss: std::cout << "dual coordinate descent L1 Loss"; break;
    case AlgoType::DualCoordinateDescentL2Loss: std::cout << "dual coordinate descent L2 Loss"; break;
    case AlgoType::LeastSquares: std::cout << "least squares"; break;
    default: abort();
  }
  std::cout << "\n";

  std::cout << "Update type: ";
  switch (updateType) {
    case UpdateType::Wild: std::cout << "wild"; break;
    case UpdateType::WildOrig: std::cout << "wild orig"; break;
    case UpdateType::ReplicateByThread: std::cout << "replicate by thread"; break;
    case UpdateType::ReplicateByPackage: std::cout << "replicate by package"; break;
    case UpdateType::Staleness: std::cout << "stale reads"; break;
    default: abort();
  }
  std::cout << "\n";
}

void initializeVariableCounts(Graph& g) {
  for (auto n : g) {
    for (auto edge_it : g.out_edges(n)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& data = g.getData(variable_node);
      data.field++; //increase count of variable occurrences
    }
  }
}

unsigned loadLabels(Graph& g, std::string filename) {
  std::ifstream infile(filename);

  unsigned sample_id;
  int label;
  int num_labels = 0;
  while (infile >> sample_id >> label) {
    g.getData(sample_id).field = label;
    ++num_labels;
  }
  
  return num_labels;
}

size_t getNumCorrect(Graph& g, std::vector<GNode>& testing_samples) {
  Galois::GAccumulator<size_t> correct;

  Galois::do_all(testing_samples.begin(), testing_samples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g.getData(n);
    int label = data.field;
    for (auto edge_it : g.out_edges(n)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& data = g.getData(variable_node);
      double weight = g.getEdgeData(edge_it);
      sum += data.w * weight;
    }

    if (sum <= 0.0 && label == -1) {
      correct += 1;
    } else if (sum > 0.0 && label == 1) {
      correct += 1;
    }
  });
  
  return correct.reduce();
}

double getPrimalObjective(Graph& g, const std::vector<GNode>& trainingSamples) {
  // 0.5 * w^Tw + C * sum_i [max(0, 1 - y_i * w^T * x_i)]^2
  Galois::GAccumulator<double> objective;

  Galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g.getData(n);
    int label = data.field;
    for (auto edge_it : g.out_edges(n)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& data = g.getData(variable_node);
      double weight = g.getEdgeData(edge_it);
      sum += data.w * weight;
    }

    double o = std::max(0.0, 1 - label * sum);
    objective += o * o;
  });
  
  Galois::GAccumulator<double> norm;
  Galois::do_all(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(NUM_VARIABLES), [&](size_t i) {
    double v = g.getData(i + NUM_SAMPLES).w;
    norm += v * v;
  });
  return objective.reduce() * creg + 0.5 * norm.reduce();
}

template<UpdateType UT>
void runPrimalSgd(Graph& g, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples) {
  Galois::TimeAccumulator accumTimer;
  accumTimer.start();

  DiffractedCollection<double, UT> dstate(NUM_VARIABLES);

  Galois::StatTimer sgdTime("SgdTime");
  
  unsigned iterations = maxIterations;
  double minObj = std::numeric_limits<double>::max();
  if (fixedIterations)
    iterations = fixedIterations;

  for (unsigned iter = 1; iter <= iterations; ++iter) {
    sgdTime.start();
    
    //include shuffling time in the time taken per iteration
    //also: not parallel
    if (shuffleSamples)
      std::shuffle(trainingSamples.begin(), trainingSamples.end(), gen);
    
    double learning_rate = 30/(100.0 + iter);
    auto ts_begin = trainingSamples.begin();
    auto ts_end = trainingSamples.end();
    auto ln = Galois::loopname("LinearSVM");
    auto wl = Galois::wl<Galois::WorkList::dChunkedFIFO<32>>();
    Galois::GAccumulator<size_t> bigUpdates;

    Galois::Timer flopTimer;
    flopTimer.start();

    Galois::for_each(ts_begin, ts_end, LinearSVM<UT>(g, dstate, learning_rate, bigUpdates), ln, wl);

    flopTimer.stop();
    sgdTime.stop();

    size_t numBigUpdates = bigUpdates.reduce();
    double flop = 4*g.sizeEdges() + 2 + 3*numBigUpdates + g.sizeEdges();
    size_t millis = flopTimer.get();
    double gflops = 0;
    if (millis)
      gflops = flop / millis / 1e6;

    dstate.merge([&g](ptrdiff_t x) -> double& { return g.getData(x + NUM_SAMPLES).w; });

    accumTimer.stop();
    std::cout 
      << iter << " GFLOP/s " << gflops << " "
      << "(" << millis / 1e3 << " s)"
      << " AccumTime " << accumTimer.get() / 1e3;
    accumTimer.start();
    if (printAccuracy) {
      std::cout << " Accuracy: " << getNumCorrect(g, testingSamples)/ (double) testingSamples.size();
    }
    double obj = 0;
    if (!fixedIterations) {
      obj = getPrimalObjective(g, trainingSamples);
    }
    if (printObjective) {
      std::cout << " Obj: " << obj;
    }
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

double getDualObjective(Graph& g, const std::vector<GNode>& trainingSamples, double* diag, const std::vector<double>& alpha) {
  // 0.5 * w^Tw + C * sum_i [max(0, 1 - y_i * w^T * x_i)]^2
  Galois::GAccumulator<double> objective;

  Galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode n) {
    //Node& data = g.getData(n);
    int label = g.getData(n).field;
    objective += alpha[n] * (alpha[n] * diag[label + 1] - 2);
  });
  
  Galois::do_all(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(NUM_VARIABLES), [&](size_t i) {
    double v = g.getData(i + NUM_SAMPLES).w;
    objective += v * v;
  });
  return objective.reduce();
}

//TODO(ddn): Parallelize
// See Algorithm 3 of Hsieh et al., ICML 2008
void runDualCoordinateDescent(Graph& g, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples, bool useL1Loss) {
  Galois::TimeAccumulator accumTimer;
  accumTimer.start();
  
  std::vector<double> alpha(NUM_SAMPLES);
  std::vector<double> QD(NUM_SAMPLES);

  double diag[] = { 0.5/creg, 0, 0.5/creg };
  double ub[] = { std::numeric_limits<double>::max(), 0, std::numeric_limits<double>::max() };
  if (useL1Loss) {
    diag[0] = 0;
    diag[2] = 0;
    ub[0] = creg;
    ub[2] = creg;
  }

  for (auto ii = g.begin(), ei = g.begin() + NUM_SAMPLES; ii != ei; ++ii) {
    int& label = g.getData(*ii).field;
    if (label != 1 && label != -1) {
      label = label <= 0 ? -1 : 1;
    }

    QD[*ii] = diag[label + 1];
    for (auto edge : g.out_edges(*ii)) {
      double val = g.getEdgeData(edge);
      QD[*ii] += val * val;
    }
  }

  Galois::StatTimer cdTime("CDTime");
  double maxPG = std::numeric_limits<double>::max();
  double minPG = std::numeric_limits<double>::lowest();
  std::vector<GNode> active = trainingSamples;
  std::vector<GNode> activeNew;
  unsigned iter;
  for (iter = 1; iter <= maxIterations; ++iter) {
    cdTime.start();
    
    double maxPGNew = std::numeric_limits<double>::lowest();
    double minPGNew = std::numeric_limits<double>::max();
    if (iter != 1 && shuffleSamples) {
      std::shuffle(active.begin(), active.end(), gen);
    }

    Galois::Timer flopTimer;
    flopTimer.start();
    size_t flop = 0;
    for (GNode n : active) {
      double G = 0;
      int label = g.getData(n).field;
      flop += 2 * std::distance(g.edge_begin(n), g.edge_end(n));
      for (auto edge : g.out_edges(n)) {
        G += g.getData(g.getEdgeDst(edge)).w * g.getEdgeData(edge);
      }
      G = G * label - 1;
      G += alpha[n] * diag[label+1];
      flop += 3;

      double C = ub[label+1];
      double PG = 0;
      if (alpha[n] == 0) {
        if (G > maxPG) {
          continue;
        } else if (G < 0) {
          PG = G;
        }
      } else if (alpha[n] == C) {
        if (G < minPG) {
          continue;
        } else if (G > 0) {
          PG = G;
        }
      } else {
        PG = G;
      }
      maxPGNew = std::max(maxPGNew, PG);
      minPGNew = std::min(minPGNew, PG);

      if (std::fabs(PG) > 1.0e-12) {
        double a = alpha[n];
        alpha[n] = std::min(std::max(alpha[n] - G/QD[n], 0.0), C);
        double d = (alpha[n] - a) * label;
        flop += 3;
        flop += 2 * std::distance(g.edge_begin(n), g.edge_end(n));
        for (auto edge : g.out_edges(n)) {
          double& w = g.getData(g.getEdgeDst(edge)).w;
          w += d * g.getEdgeData(edge);
        }
      }
      activeNew.push_back(n);
    }
    flopTimer.stop();
    cdTime.stop();

    accumTimer.stop();
    size_t millis = flopTimer.get();
    double gflops = 0;
    if (millis)
      gflops = flop / millis / 1e6;
    std::cout 
      << iter << " GFLOP/s " << gflops << " "
      << "(" << millis / 1e3 << " s)"
      << " AccumTime " << accumTimer.get() / 1e3
      << " ActiveSet " << active.size();
    accumTimer.start();
    if (printAccuracy) {
      double accuracy = getNumCorrect(g, testingSamples) / (double) testingSamples.size();
      std::cout << " Accuracy: " << accuracy;
    } 
    if (printObjective) {
      std::cout << " Obj: " << getDualObjective(g, trainingSamples, diag, alpha);
    }
    std::cout << "\n";

    active.clear();
    std::swap(active, activeNew);

    if (maxPGNew - minPGNew <= tol) {
      if (active.size() == trainingSamples.size()) {
        std::cout << "Converged in " << iter << " iterations\n";
        return;
      } else {
        active = trainingSamples;
        maxPG = std::numeric_limits<double>::max();
        minPG = std::numeric_limits<double>::lowest();
      }
    } else {
      maxPG = maxPGNew;
      minPG = minPGNew;
      if (maxPG <= 0)
        maxPG = std::numeric_limits<double>::max();
      if (minPG >= 0)
        minPG = std::numeric_limits<double>::lowest();
    }
  }

  std::cout << "Failed to converge\n";
}

#ifdef HAS_EIGEN
void runLeastSquares(Graph& g, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples) {
  Eigen::SparseMatrix<double> A(NUM_SAMPLES, NUM_VARIABLES);
  {
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> triplets { g.sizeEdges() };
    {
      auto it = triplets.begin();
      for (auto n : g) {
        for (auto edge : g.out_edges(n)) {
          *it++ = Triplet(n, g.getEdgeDst(edge) - NUM_SAMPLES, g.getEdgeData(edge));
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
    Eigen::SparseMatrix<double> AT = A.transpose();
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
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
      GALOIS_DIE("factorization failed");
    }
    Eigen::VectorXd QTb = solver.matrixQ().transpose() * b;
    int r = solver.rank();
    Eigen::VectorXd out;
    out.resize(A.cols());
    out.topRows(r) = solver.matrixR().topLeftCorner(r, r).triangularView<Eigen::Upper>().solve(QTb.topRows(r));
    out.bottomRows(out.rows() - r).setZero();
    x = solver.colsPermutation() * out;
  }

  // Verify
  {
    for (size_t i = 0; i < NUM_VARIABLES; ++i) {
      g.getData(i + NUM_SAMPLES).w = x(i);
    }
    std::vector<GNode> allSamples(g.begin(), g.begin() + NUM_SAMPLES);
    size_t n = getNumCorrect(g, allSamples);
    std::cout << "All: " << n / (double) NUM_SAMPLES << " (" << n << "/" << NUM_SAMPLES << ")\n";
    n = getNumCorrect(g, testingSamples);
    std::cout << "Testing: " << n / (double) testingSamples.size() << " (" << n << "/" << testingSamples.size() << ")\n";
  }
}
#endif

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;
  
  Graph g;
  Galois::Graph::readGraph(g, inputGraphFilename);
  NUM_SAMPLES = loadLabels(g, inputLabelFilename);
  initializeVariableCounts(g);
  NUM_VARIABLES = g.size() - NUM_SAMPLES;
  assert(NUM_SAMPLES > 0 && NUM_VARIABLES > 0);

  //put samples in a list and shuffle them
  std::random_device rd;
  std::mt19937 gen(SEED == ~0U ? rd() : SEED);

  std::vector<GNode> allSamples(g.begin(), g.begin() + NUM_SAMPLES);
  std::shuffle(allSamples.begin(), allSamples.end(), gen);

  //copy a fraction of the samples to the training samples list
  unsigned numTraining = numberTraining;
  if (numTraining == 0 || numTraining >= NUM_SAMPLES) 
    numTraining = std::min(static_cast<unsigned>(NUM_SAMPLES * fractionTraining), NUM_SAMPLES);
  std::vector<GNode> trainingSamples(allSamples.begin(), allSamples.begin() + numTraining);
  //the remainder of samples go into the testing samples list
  std::vector<GNode> testingSamples(allSamples.begin() + numTraining, allSamples.end());
  
  printParameters(trainingSamples, testingSamples);
  if (printAccuracy) {
    std::cout << "Initial";
    if (printAccuracy) {
      std::cout << " Accuracy: " << getNumCorrect(g, testingSamples) / (double) testingSamples.size();
    }
    std::cout << "\n";
  }

  Galois::StatTimer timer;
  timer.start();
  switch (algoType) {
    case AlgoType::PrimalStochasticGradientDescent:
      switch (updateType) {
        case UpdateType::Wild: runPrimalSgd<UpdateType::Wild>(g, gen, trainingSamples, testingSamples); break;
        case UpdateType::WildOrig: runPrimalSgd<UpdateType::WildOrig>(g, gen, trainingSamples, testingSamples); break;
        case UpdateType::ReplicateByPackage: runPrimalSgd<UpdateType::ReplicateByPackage>(g, gen, trainingSamples, testingSamples); break;
        case UpdateType::ReplicateByThread: runPrimalSgd<UpdateType::ReplicateByThread>(g, gen, trainingSamples, testingSamples); break;
        case UpdateType::Staleness: runPrimalSgd<UpdateType::Staleness>(g, gen, trainingSamples, testingSamples); break;
        default: abort();
      }
      break;
    case AlgoType::DualCoordinateDescentL1Loss: runDualCoordinateDescent(g, gen, trainingSamples, testingSamples, true); break;
    case AlgoType::DualCoordinateDescentL2Loss: runDualCoordinateDescent(g, gen, trainingSamples, testingSamples, false); break;
#ifdef HAS_EIGEN
    case AlgoType::LeastSquares: runLeastSquares(g, gen, trainingSamples, testingSamples); break;
#endif
    default: abort();
  }
  timer.stop();

  return 0;
}

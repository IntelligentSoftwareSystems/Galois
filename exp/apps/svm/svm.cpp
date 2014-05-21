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
#include "Galois/Graph/Graph.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Accumulator.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Lonestar/BoilerPlate.h"

#ifdef HAS_EIGEN
#include <Eigen/Sparse>
#endif

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

Galois::Runtime::PerThreadStorage<double*> thread_weights;
Galois::Runtime::PerPackageStorage<double*> package_weights;
Galois::LargeArray<double> old_weights;

//undef to test specialization for dense feature space
//#define DENSE
//#define DENSE_NUM_FEATURES 500

template<UpdateType UT>
struct linearSVM {  
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  double learningRate;
  Galois::GAccumulator<size_t>& bigUpdates;
  bool has_other;

#ifdef DENSE
  Node* baseNodeData;
  ptrdiff_t edgeOffset;
  double* baseEdgeData;
#endif
  linearSVM(Graph& _g, double _lr, Galois::GAccumulator<size_t>& b) : g(_g), learningRate(_lr), bigUpdates(b) {
    has_other = Galois::Runtime::LL::getMaxPackageForThread(Galois::getActiveThreads() - 1) > 1;
#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }
  
  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {  
    double *packagew = *package_weights.getLocal();
    double *threadw = *thread_weights.getLocal();
    double *otherw = NULL;
    Galois::PerIterAllocTy& alloc = ctx.getPerIterAlloc();

    if (has_other) {
      unsigned tid = Galois::Runtime::LL::getTID();
      unsigned my_package = Galois::Runtime::LL::getPackageForSelf(tid);
      unsigned next = my_package + 1;
      if (next >= 4)
        next -= 4;
      otherw = *package_weights.getRemoteByPkg(next);
    }
    // Store edge data in iteration-local temporary to reduce cache misses
#ifdef DENSE
    const ptrdiff_t size = DENSE_NUM_FEATURES;
#else
    ptrdiff_t size = std::distance(g.edge_begin(n), g.edge_end(n));
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
      switch (UT) {
        default:
        case UpdateType::WildOrig:
        case UpdateType::Wild:
          weight = baseNodeData[cur].w;
          break;
        case UpdateType::ReplicateByThread:
          weight = threadw[cur+edgeOffset];
          break;
        case UpdateType::ReplicateByPackage:
          weight = packagew[cur+edgeOffset];
          break;
        case UpdateType::Staleness:
          weight = old_weights[cur+edgeOffset]; 
          break;
      }
#else
      switch (UT) {
        default:
        case UpdateType::WildOrig:
        case UpdateType::Wild:
          wptrs[cur] = &var_data.w;
          weight = *wptrs[cur];
          break;
        case UpdateType::ReplicateByThread:
          wptrs[cur] = &threadw[variableNodeToId(variable_node)];
          weight = *wptrs[cur];
          break;
        case UpdateType::ReplicateByPackage:
          wptrs[cur] = &packagew[variableNodeToId(variable_node)];
          weight = *wptrs[cur];
          break;
        case UpdateType::Staleness:
          wptrs[cur] = &threadw[variableNodeToId(variable_node)];
          weight = old_weights[variableNodeToId(variable_node)]; 
          break;
      }
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
      switch (UT) {
        default:
        case UpdateType::WildOrig:
        case UpdateType::Wild:
          baseNodeData[cur].w = mweights[cur] - delta;
          break;
        case UpdateType::ReplicateByThread:
          threadw[cur+edgeOffset] = mweights[cur] - delta;
          break;
        case UpdateType::ReplicateByPackage:
          packagew[cur+edgeOffset] = mweights[cur] - delta;
          break;
        case UpdateType::Staleness:
          threadw[cur+edgeOffset] = mweights[cur] - delta;
          break;
      }
#else
      if (UT == UpdateType::WildOrig) {
        *wptrs[cur] -= delta;
      } else {
        *wptrs[cur] = mweights[cur] - delta;
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
    case AlgoType::PrimalStochasticGradientDescent: std::cout << "primal stocahstic gradient descent"; break;
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

void runPrimalSgd(Graph& g, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples) {
  Galois::TimeAccumulator accumTimer;
  accumTimer.start();

  //allocate storage for weights from previous iteration
  old_weights.create(NUM_VARIABLES);
  if (updateType == UpdateType::ReplicateByThread || updateType == UpdateType::Staleness) {
    Galois::on_each([](unsigned tid, unsigned total) {
      double *p = new double[NUM_VARIABLES];
      *thread_weights.getLocal() = p;
      std::fill(p, p + NUM_VARIABLES, 0);
    });
  }
  if (updateType == UpdateType::ReplicateByPackage) {
    Galois::on_each([](unsigned tid, unsigned total) {
      if (Galois::Runtime::LL::isPackageLeader(tid)) {
        double *p = new double[NUM_VARIABLES];
        *package_weights.getLocal() = p;
        std::fill(p, p + NUM_VARIABLES, 0);
      }
    });
  }

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

    UpdateType type = updateType;
    switch (type) {
      case UpdateType::Wild:
        Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::Wild>(g, learning_rate, bigUpdates), ln, wl);
        break;
      case UpdateType::WildOrig:
        Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::WildOrig>(g, learning_rate, bigUpdates), ln, wl);
        break;
      case UpdateType::ReplicateByPackage:
        Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::ReplicateByPackage>(g, learning_rate, bigUpdates), ln, wl);
        break;
      case UpdateType::ReplicateByThread:
        Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::ReplicateByThread>(g, learning_rate, bigUpdates), ln, wl);
        break;
      case UpdateType::Staleness:
        Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::Staleness>(g, learning_rate, bigUpdates), ln, wl);
        break;
      default: abort();
    }
    flopTimer.stop();
    sgdTime.stop();

    size_t numBigUpdates = bigUpdates.reduce();
    double flop = 4*g.sizeEdges() + 2 + 3*numBigUpdates + g.sizeEdges();
    if (type == UpdateType::ReplicateByPackage)
      flop += numBigUpdates;
    size_t millis = flopTimer.get();
    double gflops = 0;
    if (millis)
      gflops = flop / millis / 1e6;

    //swap weights from past iteration and this iteration
    if (type != UpdateType::Wild && type != UpdateType::WildOrig) {
      bool byThread = type == UpdateType::ReplicateByThread || type == UpdateType::Staleness;
      double *localw = byThread ? *thread_weights.getLocal() : *package_weights.getLocal();
      unsigned num_threads = Galois::getActiveThreads();
      unsigned num_packages = Galois::Runtime::LL::getMaxPackageForThread(num_threads-1) + 1;
      Galois::do_all(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(NUM_VARIABLES), [&](unsigned i) {
        unsigned n = byThread ? num_threads : num_packages;
        for (unsigned j = 1; j < n; j++) {
          double o = byThread ?
            (*thread_weights.getRemote(j))[i] :
            (*package_weights.getRemoteByPkg(j))[i];
          localw[i] += o;
        }
        localw[i] /=  n;
        GNode variable_node = (GNode) (i + NUM_SAMPLES);
        Node& var_data = g.getData(variable_node, Galois::NONE);
        var_data.w = localw[i];
        old_weights[i] = var_data.w;
      });
      Galois::on_each([&](unsigned tid, unsigned total) {
        switch (type) {
          case UpdateType::Staleness:
          case UpdateType::ReplicateByThread:
            if (tid)
              std::copy(localw, localw + NUM_VARIABLES, *thread_weights.getLocal());
          break;
          case UpdateType::ReplicateByPackage:
            if (tid && Galois::Runtime::LL::isPackageLeader(tid))
              std::copy(localw, localw + NUM_VARIABLES, *package_weights.getLocal());
          break;
          default: abort();
        }
      });
    }

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
    Node& data = g.getData(n);
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
    case AlgoType::PrimalStochasticGradientDescent: runPrimalSgd(g, gen, trainingSamples, testingSamples); break;
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

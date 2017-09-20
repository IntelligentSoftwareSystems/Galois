/** SVM with SGD/DCD/CD/newGLMENT -*- C++ -*-
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
 * Dual Coordinate Descent for solving L2 regularization + (hinge, squared hinge, 
 *                                                  and logsitic) loss functions
 * Coordinate Descent for solving L1 regularization + squared-L2 loss
 * newGLMENT for solving L1 regularization + Logsitic loss
 *
 * @author Prad Nelluru <pradn@cs.utexas.edu>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Cho-Jui Hiseh <cjhsieh@cs.utexas.edu>
 * @author Hsiang-Fu Yu <rofuyu@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Accumulator.h"
#include "Galois/ParallelSTL.h"
#include "Galois/Substrate/PaddedLock.h"
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
  SGDL1,
  SGDL2,
  SGDLR, 
  DCDL1,
  DCDL2,
  DCDLR,
  CDLasso, 
  GLMNETL1RLR,
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputTrainGraphFilename(cll::Positional, cll::desc("<training graph input file>"), cll::Required);
static cll::opt<std::string> inputTrainLabelFilename(cll::Positional, cll::desc("<training label input file>"), cll::Required);
static cll::opt<std::string> inputTestGraphFilename(cll::Positional, cll::desc("<testing graph input file>"), cll::Required);
static cll::opt<std::string> inputTestLabelFilename(cll::Positional, cll::desc("<testing label input file>"), cll::Required);
static cll::opt<double> creg("creg", cll::desc("the regularization parameter C"), cll::init(1.0));
static cll::opt<bool> shuffleSamples("shuffle", cll::desc("shuffle samples between iterations"), cll::init(true));
static cll::opt<unsigned> SEED("seed", cll::desc("random seed"), cll::init(~0U));
static cll::opt<bool> printObjective("printObjective", cll::desc("print objective value"), cll::init(true));
static cll::opt<bool> printAccuracy("printAccuracy", cll::desc("print accuracy value"), cll::init(true));
static cll::opt<double> tol("tol", cll::desc("convergence tolerance"), cll::init(0.1));
static cll::opt<unsigned> maxIterations("maxIterations", cll::desc("maximum number of iterations"), cll::init(1000));
static cll::opt<bool> useshrink("useshrink", cll::desc("use rhinking strategy for coordinate descent"), cll::init(true));
static cll::opt<unsigned> fixedIterations("fixedIterations", cll::desc("run specific number of iterations, ignoring convergence"), cll::init(10));
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
//      clEnumValN(AlgoType::SGDL1, "sgdl1", "primal stochastic gradient descent hinge loss (default)"),
//      clEnumValN(AlgoType::SGDL2, "sgdl2", "primal stochastic gradient descent square-hinge loss"),
//      clEnumValN(AlgoType::SGDLR, "sgdlr", "primal stochastic gradient descent logistic regression"),
      clEnumValN(AlgoType::DCDL1, "dcdl1", "Dual coordinate descent hinge loss"),
      clEnumValN(AlgoType::DCDL2, "dcdl2", "Dual coordinate descent square-hinge loss"),
      clEnumValN(AlgoType::DCDLR, "dcdlr", "Dual coordinate descent logistic regression"),
	  clEnumValN(AlgoType::CDLasso, "cdlasso", "Coordinate descent Lasso"), 
	  clEnumValN(AlgoType::GLMNETL1RLR, "l1rlr", "new GLMENT for L1-regularized Logistic Regression"), 
      clEnumValEnd), cll::init(AlgoType::DCDL1));

/**          DATA TYPES        **/

typedef struct Node {
  //double w; //weight - relevant for variable nodes
  union {
	  int field; //variable nodes - (1/variable count), sample nodes - label
	  int y; // sample node label
  };

  union {
    double alpha; // sample node
    double w; // variable node
	double exp_wTx; // sample node: newGLMNET;
  };

  union {
	  double QD;
	  double xTx;
	  double xTd; // sample node: newGLMNET
	  double Hdiag; // variable node: newGLMNET
  };

  union {
	  double alpha2;
	  double b; 
	  double exp_wTx_new; // sample node: newGLMNET;
	  double Grad; // variable node: newGLMNET
  };

  union {
	  double D; // sample node: newGLMNET
	  double wpd; // variable node: newGLMNET
  };

  union {
	  double tau; // sample node: used for newGLMNET
	  double xjneg_sum; // variable node: newGLMNET
  };

  Node(): w(0.0), field(0), QD(0.0), alpha2(0.0), D(0.0), tau(0.0) { }
} Node;


using Graph = galois::graphs::LC_CSR_Graph<Node, double>::with_out_of_line_lockable<true>::type;
using GNode = Graph::GraphNode;
typedef galois::InsertBag<GNode> Bag;

/**         CONSTANTS AND PARAMETERS       **/
unsigned NUM_SAMPLES = 0;
unsigned NUM_VARIABLES = 0;
unsigned NUM_TEST_SAMPLES = 0;
unsigned NUM_TEST_VARIABLES = 0;

unsigned variableNodeToId(GNode variable_node) {
  return ((unsigned) variable_node) - NUM_SAMPLES;
}

galois::Substrate::PerThreadStorage<double*> thread_weights;
galois::Substrate::PerPackageStorage<double*> package_weights;
galois::LargeArray<double> old_weights;

//undef to test specialization for dense feature space
//#define DENSE
//#define DENSE_NUM_FEATURES 500



struct LogisticRegression {  
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  bool has_other;

  AlgoType alg_type;
  double innereps;
  size_t *newton_iter;

  double diag;
  double C;

  LogisticRegression(Graph& _g, double _innereps, size_t *_newton_iter) : g(_g), innereps(_innereps), newton_iter(_newton_iter)  {
    has_other = galois::Substrate::getThreadPool().getCumulativeMaxPackage(galois::getActiveThreads() - 1) > 1;
	C = creg;
	alg_type = AlgoType::DCDLR;
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {  
	Node& sample_data = g.getData(n);

    // Gather
    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      double weight = var_data.w;
      dot += weight * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
    }
    
    int label = sample_data.field;
	double d=0.0;

	// For Coordinate Descent
	if (alg_type == AlgoType::DCDLR) {  // Added by Rofu
		int yi = label > 0 ? 1:-1;
		double ywTx = dot*yi, xisq = sample_data.xTx;
		double alpha[2] = {sample_data.alpha, sample_data.alpha2};
		//double &alpha = sample_data.alpha, &alpha2 = sample_data.alpha2;
		double a = xisq, b = ywTx;

		// Decide to minimize g_1(z) or g_2(z)
		int ind1 = 0, ind2 = 1, sign = 1;
		if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
		{
			ind1 = 1;
			ind2 = 0;
			sign = -1;
		}

		//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
		double alpha_old = alpha[ind1];
		double z = alpha_old;
		if(C - z < 0.5 * C)
			z = 0.1*z;
		double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));

		// Newton method on the sub-problem
		const double eta = 0.1; // xi in the paper
		const int max_inner_iter = 100;
		int inner_iter = 0;
		while (inner_iter <= max_inner_iter)
		{
			if(fabs(gp) < innereps)
				break;
			double gpp = a + C/(C-z)/z;
			double tmpz = z - gp/gpp;
			if(tmpz <= 0)
				z *= eta;
			else // tmpz in (0, C)
				z = tmpz;
			gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			inner_iter++;
		}
		*newton_iter += inner_iter;
		alpha[ind1] = z;
		alpha[ind2] = C-z;
		sample_data.alpha = alpha[0];
		sample_data.alpha2 = alpha[1];
		d = sign*(z - alpha_old)*yi;
		if(d == 0) return;
	}

    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node, galois::MethodFlag::UNPROTECTED);

      double delta = -d*g.getEdgeData(edge_it);
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
  //double PGmax_new;
  //double PGmin_new;
  size_t active_size;
  galois::GReduceMax<double> PGmax_new;
  galois::GReduceMin<double> PGmin_new;

  std::vector<bool> isactive;
} DCD_parameters;

struct linearSVM_DCD {  
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  Bag* next_bag;
  bool has_other;

  double diag;
  double C;
  DCD_parameters *params;

  linearSVM_DCD(Graph& _g, DCD_parameters *_params, Bag *_next_bag=NULL) : g(_g), diag(_params->diag), C(_params->C), params(_params), next_bag(_next_bag) {
    has_other = galois::Substrate::getThreadPool().getCumulativeMaxPackage(galois::getActiveThreads() - 1) > 1;

  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {  
    Node& sample_data = g.getData(n);

    // Gather
    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      double weight = var_data.w;
      dot += weight * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
    }
    
    int label = sample_data.field;
    double d=0.0;

    double &nowalpha = sample_data.alpha;
    double a = nowalpha;
    double G = dot*label - 1 + nowalpha*diag;
    double PG = 0;
    
    if ( useshrink == true) {
      if ( a == 0) {
        if ( G > params->PGmax_old) {
          return;
        } else if ( G < 0 ) {
          PG = G;
        }
      } else if ( a == C ) {
        if ( G < params->PGmin_old) {
          return;
        } else if ( G>0 ) {
          PG = G;
        }
      } else {
        PG = G;
      }
	  next_bag->push(n);

	  params->PGmax_new.update(PG);
	  params->PGmin_new.update(PG);
    }
    
    nowalpha = std::min(std::max(a-G/sample_data.QD, 0.0), C);
    d = (nowalpha - a)*label;
    if ( d == 0.0 )
      return;

    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node, galois::MethodFlag::UNPROTECTED);

      double delta = -d*g.getEdgeData(edge_it);
      var_data.w -= delta;
    }
  }
};

// SGD for linearSVM and logistic regression -- only for wild update 
template<UpdateType UT>
struct linearSGD_Wild {  

  Graph& g;
  double learningRate;
  bool has_other;

#ifdef DENSE
  Node* baseNodeData;
  ptrdiff_t edgeOffset;
  double* baseEdgeData;
#endif
  linearSGD_Wild(Graph& _g, double _lr) : g(_g), learningRate(_lr) {
    has_other = galois::Substrate::getThreadPool().getCumulativeMaxPackage(galois::getActiveThreads() - 1) > 1;
#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {  
    Node& sample_data = g.getData(n);
   	double invcreg = 1.0/creg; 
	// Gather
    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
      double weight = var_data.w;
	  dot += weight * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
    }
    
    int label = sample_data.field;

	bool bigUpdate = true;
	if ( (algoType == AlgoType::SGDL1) || (algoType == AlgoType::SGDL2))
		bigUpdate = label * dot < 1;

	double d = 0.0;
	if ( algoType == AlgoType::SGDL1 )
		d = 1.0;
	else if ( algoType == AlgoType::SGDL2 )
		d = 2*(1-label*dot);
	else if ( algoType == AlgoType::SGDLR )
		d = 1/(1+exp(dot*label));

    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
	  int varCount = var_data.field;
	  double rfactor = var_data.QD;
	  double delta = 0; 
	  if ( bigUpdate == true)
		  delta = learningRate * ( var_data.w*rfactor - d * label *  g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED));
	  else
		  delta = learningRate * ( var_data.w*rfactor);
      var_data.w -= delta;
    }
  }
};



// SGD for linearSVM and logistic regression 
template<UpdateType UT>
struct linearSGD {  
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_aborts;

  Graph& g;
  double learningRate;
  galois::GAccumulator<size_t>& bigUpdates;
  bool has_other;

#ifdef DENSE
  Node* baseNodeData;
  ptrdiff_t edgeOffset;
  double* baseEdgeData;
#endif
  linearSGD(Graph& _g, double _lr, galois::GAccumulator<size_t>& b) : g(_g), learningRate(_lr), bigUpdates(b) {
    has_other = galois::Substrate::getThreadPool().getCumulativeMaxPackage(galois::getActiveThreads() - 1) > 1;
#ifdef DENSE
    baseNodeData = &g.getData(g.getEdgeDst(g.edge_begin(0)));
    edgeOffset = std::distance(&g.getData(NUM_SAMPLES), baseNodeData);
    baseEdgeData = &g.getEdgeData(g.edge_begin(0));
#endif
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {  
    double *packagew = *package_weights.getLocal();
    double *threadw = *thread_weights.getLocal();
    double *otherw = NULL;
    galois::PerIterAllocTy& alloc = ctx.getPerIterAlloc();

    if (has_other) {
      unsigned tid = galois::Substrate::ThreadPool::getTID();
      unsigned my_package = galois::Substrate::ThreadPool::getPackage();
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
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& var_data = g.getData(variable_node, galois::MethodFlag::UNPROTECTED);
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
    if (bigUpdate || (algoType == AlgoType::SGDLR))
      bigUpdates += size;

	double d = 0.0;
	if ( algoType == AlgoType::SGDL1 )
		d = 1.0;
	else if ( algoType == AlgoType::SGDL2 )
		d = 2*(1-label*dot);
	else if ( algoType == AlgoType::SGDLR )
		d = 1/(1+exp(dot*label));

    for (cur = 0; cur < size; ++cur) {
      double delta;
	  if ( algoType == AlgoType::SGDLR )
	  {
      	if (UT == UpdateType::WildOrig) {
      	    delta = learningRate * (*wptrs[cur]/rfactors[cur] - d * label * dweights[cur]);
     	 } else {
     	     delta = learningRate * (rfactors[cur] - d * label * dweights[cur]);
     	 }
	  }
	  else
	  {
    	if (UT == UpdateType::WildOrig) {
      	  if (bigUpdate)
      	    delta = learningRate * (*wptrs[cur]/rfactors[cur] - d * label * dweights[cur]);
     	   else
        	  delta = *wptrs[cur]/rfactors[cur];
     	 } else {
     	   if (bigUpdate)
     	     delta = learningRate * (rfactors[cur] - d * label * dweights[cur]);
     	   else
     	     delta = rfactors[cur];
     	 }
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

typedef struct {
	double Gmax_old;
	double Gnorm1_init;
  galois::GReduceMax<double> Gmax_new;
  galois::GAccumulator<double> Gnorm1_new;
} CD_parameters;

// Primal CD for Lasso 
struct Lasso_CD {  
  Graph& g;
  double lambda;
  CD_parameters *params;
  Bag* next_bag;

  Lasso_CD(Graph& _g, CD_parameters *_params, Bag *_next_bag=NULL) : g(_g), params(_params), next_bag(_next_bag) {
	  lambda = 0.5/creg;
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {  
    Node& var_data = g.getData(n);
	double &w = var_data.w;

	if ( var_data.xTx == 0.0 )
		return;
	double wold = w;
	double ainv = var_data.xTx;

//	if ( n>10+NUM_SAMPLES )
//		return;
	// Gather
    double dot = 0.0;
    for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
      GNode sample_node = g.getEdgeDst(edge_it);
      Node& sample_data = g.getData(sample_node, galois::MethodFlag::UNPROTECTED);
	  double r = sample_data.alpha;
	  dot += r * g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
//	  printf("%d-%d %lf   ", n-NUM_SAMPLES, sample_node, g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED));
    }

	double violation = 0;
	if ( useshrink )
	{
	double G = dot*2*creg;
	double Gp = G+1;
	double Gn = G-1;
	if ( wold == 0 )
	{
		if ( Gp < 0 )
			violation = -Gp;
		else if ( Gn > 0 )
			violation = Gn;
		else if (Gp>(params->Gmax_old/NUM_SAMPLES) && Gn < -(params->Gmax_old/NUM_SAMPLES))
			return;
	} else if (wold >0) {
		violation = std::fabs(Gp);
	} else {
		violation = std::fabs(Gn);
	}

	params->Gmax_new.update(violation);
	params->Gnorm1_new += violation;
	next_bag->push(n);
	}
	double z = wold - dot*ainv;
	double lambda1 = lambda*ainv;

//	double z = wold - dot/a;
//	double lambda1 = 0.5/creg/a;

	double wnew = std::max(std::fabs(z)-lambda1, 0.0);
	if ( z<0)
		wnew =-wnew;
	double delta = wnew - wold;
//	if ( n-NUM_SAMPLES > 1)
//	printf("%d: %lf %lf %lf\n", n-NUM_SAMPLES-2, wold, wnew, delta);
	if ( std::fabs(delta) > 1e-12 )
	{
    	for (auto edge_it : g.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
			GNode sample_node = g.getEdgeDst(edge_it);
			Node& sample_data = g.getData(sample_node, galois::MethodFlag::UNPROTECTED);
			sample_data.alpha += delta*g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);
		}
		w = wnew;
    }
  }
};


double getDualObjective(Graph& g, const std::vector<GNode>& trainingSamples, double* diag, const std::vector<double>& alpha) {
  // 0.5 * w^Tw + C * sum_i [max(0, 1 - y_i * w^T * x_i)]^2
  galois::GAccumulator<double> objective;

  galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode n) {
    Node& data = g.getData(n);
    int label = g.getData(n).field;
    objective += alpha[n] * (alpha[n] * diag[label + 1] - 2);
  });
  
  galois::do_all(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(NUM_VARIABLES), [&](size_t i) {
    double v = g.getData(i + NUM_SAMPLES).w;
    objective += v * v;
  });
  return objective.reduce();
}




void printParameters(const std::vector<GNode>& trainingSamples, const std::vector<GNode>& testingSamples) {
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
    case AlgoType::SGDL1: std::cout << "primal stocahstic gradient descent for hinge Loss"; break;
    case AlgoType::SGDL2: std::cout << "primal stocahstic gradient descent for square-hinge Loss"; break;
    case AlgoType::SGDLR: std::cout << "primal stocahstic gradient descent for logistic Loss"; break;
	case AlgoType::DCDL1: std::cout << "dual coordinate descent hinge loss parallel"; break;
	case AlgoType::DCDL2: std::cout << "dual coordinate descent square-hinge loss parallel"; break;
	case AlgoType::DCDLR: std::cout << "dual coordinate descent logsitic regression"; break;
	case AlgoType::CDLasso: std::cout << "coordinate descent lasso"; break;
	case AlgoType::GLMNETL1RLR: std::cout << "new GLMNET l1r-lr"; break;
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
  for ( auto n : g)
  {
	  if ( n >= NUM_SAMPLES){
	  Node& data = g.getData(n);
	  if ( data.field != 0)
		  data.QD = 1.0/(creg*data.field);
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
	  if ( label >0 )
		  g.getData(sample_id).field = 1;
	  else
		  g.getData(sample_id).field = -1;
    ++num_labels;
  }
  
  return num_labels;
}

size_t getNumCorrect(Graph& g_test, std::vector<GNode>& testing_samples, Graph& g_train) {
  galois::GAccumulator<size_t> correct;

  std::vector<double> w_vec(NUM_VARIABLES);
  galois::do_all(g_train.begin()+NUM_SAMPLES, g_train.end(), [&](GNode n) {
		  Node& data = g_train.getData(n);
		  w_vec[n-NUM_SAMPLES] = data.w;
  });

  galois::do_all(testing_samples.begin(), testing_samples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g_test.getData(n);
    int label = data.field;
    for (auto edge_it : g_test.out_edges(n)) {
      GNode variable_node = g_test.getEdgeDst(edge_it);
	  if ( (variable_node-NUM_TEST_SAMPLES) < NUM_VARIABLES )
	  {
      	double weight = g_test.getEdgeData(edge_it);
	  	sum += w_vec[variable_node-NUM_TEST_SAMPLES]*weight;
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

double getTestRMSE(Graph& g_test, std::vector<GNode>& testing_samples, Graph& g_train) {
  galois::GAccumulator<double> square_err;

  std::vector<double> w_vec(NUM_VARIABLES);
  galois::do_all(g_train.begin()+NUM_SAMPLES, g_train.end(), [&](GNode n) {
		  Node& data = g_train.getData(n);
		  w_vec[n-NUM_SAMPLES] = data.w;
  });

  double wnorm = 0;
  for ( auto i: w_vec)
	  wnorm += i*i;
//  printf("wnorm: %lf, umvariables: %d\n", wnorm, NUM_VARIABLES);

  galois::do_all(testing_samples.begin(), testing_samples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g_test.getData(n);
    double b = data.b;
    for (auto edge_it : g_test.out_edges(n)) {
      GNode variable_node = g_test.getEdgeDst(edge_it);
	  if ( (variable_node-NUM_TEST_SAMPLES) < NUM_VARIABLES )
	  {
      	double weight = g_test.getEdgeData(edge_it);
	  	sum += w_vec[variable_node-NUM_TEST_SAMPLES]*weight;
	  }
    }
	square_err += (sum-b)*(sum-b);
  });

  double err = square_err.reduce();
//  printf("err: %lf\n", err);
  return sqrt(err/NUM_TEST_SAMPLES);
//  return correct.reduce();
}


double getPrimalObjective(Graph& g, const std::vector<GNode>& trainingSamples) {
  // 0.5 * w^Tw + C * sum_i loss(w^T * x_i, y_i)
  //// 0.5 * w^Tw + C * sum_i [max(0, 1 - y_i * w^T * x_i)]^2
  galois::GAccumulator<double> objective;

  galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode n) {
    double sum = 0.0;
    Node& data = g.getData(n);
    int label = data.field;
	double b;
	if ( algoType == AlgoType::CDLasso )
		b = data.b;
    for (auto edge_it : g.out_edges(n)) {
      GNode variable_node = g.getEdgeDst(edge_it);
      Node& data = g.getData(variable_node);
      double weight = g.getEdgeData(edge_it);
      sum += data.w * weight;
    }

	double o;
	if ( (algoType == AlgoType::DCDL2) || (algoType == AlgoType::SGDL2))
	{
		o = std::max(0.0, 1-label*sum);
		o = o*o;
	} 
	else if ( (algoType == AlgoType::DCDLR) || (algoType == AlgoType::SGDLR) || (algoType == AlgoType::GLMNETL1RLR) )
		o = log(1+exp(-label*sum));
	else if ( (algoType == AlgoType::DCDL1) || (algoType == AlgoType::SGDL1) )
		o = std::max(0.0, 1-label*sum);
	else if ( algoType == AlgoType::CDLasso ) 
		o = (sum - b)*(sum-b);
    objective += o;
  });
  
  galois::GAccumulator<double> norm;
  galois::do_all(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(NUM_VARIABLES), [&](size_t i) {
    double v = g.getData(i + NUM_SAMPLES).w;
	if ( algoType == AlgoType::CDLasso || algoType == AlgoType::GLMNETL1RLR)
		norm += std::fabs(v);
	else
	    norm += 0.5*v * v;
  });
  return objective.reduce() * creg + norm.reduce();
}


void runDCD(Graph& g_train, Graph& g_test, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples) {
	galois::TimeAccumulator accumTimer;
	accumTimer.start();
	galois::StatTimer DcdTime("DcdTime");

	// Initialization for DCD
	double diag[] = { 0.5/creg, 0, 0.5/creg };
	double ub[] = { std::numeric_limits<double>::max(), 0, std::numeric_limits<double>::max() };
	if (algoType==AlgoType::DCDL1 or algoType == AlgoType::DCDLR) {
		diag[0] = 0;
		diag[2] = 0;
		ub[0] = creg;
		ub[2] = creg;
	}

	DCD_parameters params;
	params.C = ub[0];
	params.diag = diag[0];
    params.PGmax_old = std::numeric_limits<double>::max();
	params.PGmin_old = std::numeric_limits<double>::lowest();
	params.active_size = NUM_SAMPLES;

	Bag bags[2];
	Bag *cur_bag = &bags[0], *next_bag = &bags[1];

	// For LR
	double innereps = 1e-2;
	double innereps_min = 1e-8;//min(1e-8, eps);
	galois::StatTimer QDTime("QdTime");
	QDTime.start();
	auto ts_begin = trainingSamples.begin();
	auto ts_end = trainingSamples.end();
	for (auto ii = ts_begin, ei = ts_end; ii != ei; ++ii) {
		int& yi = g_train.getData(*ii).y;
		auto& nodedata = g_train.getData(*ii);
		cur_bag->push(*ii);
		nodedata.alpha = 0;
		yi = yi >= 0 ? +1: -1;
		nodedata.xTx = 0;

		if(algoType == AlgoType::DCDLR) {
			nodedata.alpha = std::min(0.001*ub[yi+1], 1e-8);
			nodedata.alpha2 = ub[yi+1] - nodedata.alpha;
		} else {
			nodedata.alpha = 0;
		}
		for (auto edge : g_train.out_edges(*ii)) {
			double val = g_train.getEdgeData(edge);
			nodedata.xTx += val*val;
			auto variable_node = g_train.getEdgeDst(edge);
			Node &data = g_train.getData(variable_node);
			data.w += yi*nodedata.alpha*val;
		}
		nodedata.QD = nodedata.xTx + diag[yi+1];
	}
	QDTime.stop();

	unsigned iterations = maxIterations;
	double minObj = std::numeric_limits<double>::max();
	iterations = fixedIterations;

	bool is_terminate = false;
	std::vector<GNode> active_set;

	for (unsigned iter = 1; iter <= iterations && is_terminate == false; ++iter) {
		DcdTime.start();

		params.PGmax_new.reset();
		params.PGmin_new.reset();

		//include shuffling time in the time taken per iteration
		//also: not parallel

		if(useshrink) {
			active_set.clear();
			for(auto &gg: *cur_bag)
				active_set.push_back(gg);
		}
		if (shuffleSamples)
		{
			if(useshrink) {
				std::shuffle(active_set.begin(), active_set.end(), gen);
			} else {
				std::shuffle(trainingSamples.begin(), trainingSamples.end(), gen);
			}
		}
	
		size_t newton_iter = 0; // used by dcd for LR
		auto ts_begin = trainingSamples.begin();
		auto ts_end = trainingSamples.end();
		auto ln = galois::loopname("LinearSVM");
		auto wl = galois::wl<galois::WorkList::dChunkedFIFO<32>>();
		galois::GAccumulator<size_t> bigUpdates;


		if(algoType == AlgoType::DCDLR) {
			galois::for_each(ts_begin, ts_end, LogisticRegression(g_train, innereps, &newton_iter), ln, wl);
		} else if (useshrink){
			cur_bag->clear();
			printf("active set size: %d\n", active_set.size());
			galois::for_each(active_set.begin(), active_set.end(), linearSVM_DCD(g_train, &params, cur_bag), ln, wl);
		} else {
			galois::for_each(ts_begin, ts_end, linearSVM_DCD(g_train, &params), ln, wl);
		}

		if ( useshrink == true)
		{
			size_t active_size = std::distance(cur_bag->begin(), cur_bag->end());
			double PGmax_local = params.PGmax_new.reduce();
			double PGmin_local = params.PGmin_new.reduce();
			//if ( params.PGmax_new - params.PGmin_new <= tol )
			if ( PGmax_local - PGmin_local <= tol )
			{
				if ( active_size == NUM_SAMPLES)
					is_terminate = true;
				else
				{
					cur_bag->clear();
					for(auto ii = ts_begin; ii != ts_end; ii++)
						cur_bag->push(*ii);
					params.PGmax_old = std::numeric_limits<double>::max();
					params.PGmin_old = std::numeric_limits<double>::lowest();
				}
			}
			else
			{
				params.PGmax_old = PGmax_local;
				params.PGmin_old = PGmin_local;
				if ( params.PGmax_old <= 1e-300 )
				{
					params.PGmax_old = std::numeric_limits<double>::max();
				}
				if ( params.PGmin_old >= 0 )
					params.PGmin_old = std::numeric_limits<double>::lowest();
			}
		}

		DcdTime.stop();

		if(algoType == AlgoType::DCDLR)
			if(newton_iter <= trainingSamples.size()/10)
				innereps = std::max(1e-8, 0.1*innereps);
		size_t numBigUpdates = bigUpdates.reduce();

		accumTimer.stop();

		std::cout << "iter " <<  iter << " walltime " <<  DcdTime.get()/1e3;
		std::cout << " f " << getPrimalObjective(g_train, trainingSamples);
		std::cout << " accuracy " << getNumCorrect(g_test, testingSamples, g_train)/ (double) testingSamples.size();

		std::cout << "\n";
	}
}



void runPrimalSgd(Graph& g_train, Graph& g_test, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples) {
  galois::TimeAccumulator accumTimer;
  accumTimer.start();

  //allocate storage for weights from previous iteration
  old_weights.create(NUM_VARIABLES);
  if (updateType == UpdateType::ReplicateByThread || updateType == UpdateType::Staleness) {
    galois::on_each([](unsigned tid, unsigned total) {
      double *p = new double[NUM_VARIABLES];
      *thread_weights.getLocal() = p;
      std::fill(p, p + NUM_VARIABLES, 0);
    });
  }
  if (updateType == UpdateType::ReplicateByPackage) {
    galois::on_each([](unsigned tid, unsigned total) {
        if (galois::Substrate::getThreadPool().isLeader(tid)) {
        double *p = new double[NUM_VARIABLES];
        *package_weights.getLocal() = p;
        std::fill(p, p + NUM_VARIABLES, 0);
      }
    });
  }

  galois::StatTimer sgdTime("SgdTime");
  
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
    auto ln = galois::loopname("LinearSVM");
    auto wl = galois::wl<galois::WorkList::dChunkedFIFO<32>>();
    galois::GAccumulator<size_t> bigUpdates;

    galois::Timer flopTimer;
    flopTimer.start();

    UpdateType type = updateType;

    switch (type) {
      case UpdateType::Wild:
	        galois::for_each(ts_begin, ts_end, linearSGD_Wild<UpdateType::Wild>(g_train, learning_rate), ln, wl);
        break;
      case UpdateType::WildOrig:
        galois::for_each(ts_begin, ts_end, linearSGD<UpdateType::WildOrig>(g_train, learning_rate, bigUpdates), ln, wl);
        break;
      case UpdateType::ReplicateByPackage:
        galois::for_each(ts_begin, ts_end, linearSGD<UpdateType::ReplicateByPackage>(g_train, learning_rate, bigUpdates), ln, wl);
        break;
      case UpdateType::ReplicateByThread:
        galois::for_each(ts_begin, ts_end, linearSGD<UpdateType::ReplicateByThread>(g_train, learning_rate, bigUpdates), ln, wl);
        break;
      case UpdateType::Staleness:
        galois::for_each(ts_begin, ts_end, linearSGD<UpdateType::Staleness>(g_train, learning_rate, bigUpdates), ln, wl);
        break;
      default: abort();
    }
	
    flopTimer.stop();
    sgdTime.stop();

    size_t numBigUpdates = bigUpdates.reduce();
    double flop = 4*g_train.sizeEdges() + 2 + 3*numBigUpdates + g_train.sizeEdges();
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
      unsigned num_threads = galois::getActiveThreads();
      unsigned num_packages = galois::Substrate::getThreadPool().getCumulativeMaxPackage(num_threads-1) + 1;
      galois::do_all(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(NUM_VARIABLES), [&](unsigned i) {
        unsigned n = byThread ? num_threads : num_packages;
        for (unsigned j = 1; j < n; j++) {
          double o = byThread ?
            (*thread_weights.getRemote(j))[i] :
            (*package_weights.getRemoteByPkg(j))[i];
          localw[i] += o;
        }
        localw[i] /=  n;
        GNode variable_node = (GNode) (i + NUM_SAMPLES);
        Node& var_data = g_train.getData(variable_node, galois::MethodFlag::UNPROTECTED);
        var_data.w = localw[i];
        old_weights[i] = var_data.w;
      });
      galois::on_each([&](unsigned tid, unsigned total) {
        switch (type) {
          case UpdateType::Staleness:
          case UpdateType::ReplicateByThread:
            if (tid)
              std::copy(localw, localw + NUM_VARIABLES, *thread_weights.getLocal());
          break;
          case UpdateType::ReplicateByPackage:
            if (tid && galois::Substrate::getThreadPool().isLeader(tid))
              std::copy(localw, localw + NUM_VARIABLES, *package_weights.getLocal());
          break;
          default: abort();
        }
      });
    }

    accumTimer.stop();

	std::cout << "iter " <<  iter << " walltime " <<  sgdTime.get()/1e3;

	double obj = getPrimalObjective(g_train, trainingSamples) ;
	if ( printObjective )
		std::cout << " f " << obj;
	if ( printAccuracy )
		std::cout << " accuracy " << getNumCorrect(g_test, testingSamples, g_train)/ (double) testingSamples.size();

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

void runCD(Graph& g_train, Graph& g_test, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples) {
	galois::TimeAccumulator accumTimer;
	accumTimer.start();
	galois::StatTimer CdTime("CdTime");

	unsigned iterations = maxIterations;
	if (fixedIterations)
		iterations = fixedIterations;

	bool is_terminate = false;

	std::vector<GNode> variables(g_train.begin()+NUM_SAMPLES, g_train.end());

	for ( auto ii = variables.begin(), ei = variables.end(); ii!=ei ; ii++ ) {
		auto& nodedata = g_train.getData(*ii);
		nodedata.w = 0;
		nodedata.xTx = 0;

		for (auto edge : g_train.out_edges(*ii)) {
			double val = g_train.getEdgeData(edge);
			nodedata.xTx += val*val;
		}
		nodedata.xTx = 1.0/nodedata.xTx;
	}

	for ( auto ii = trainingSamples.begin(), ei = trainingSamples.end() ; ii!=ei ; ii++ ) {
		auto& nodedata = g_train.getData(*ii);
		auto& label = g_train.getData(*ii).b;
		nodedata.alpha = label*(-1);
	}

	CD_parameters params;
	params.Gmax_old = std::numeric_limits<double>::max();

	Bag cur_bag; 
	std::vector<GNode> active_set;
	for ( auto ii = variables.begin(), ei = variables.end(); ii!=ei ; ii++ )
	{
		cur_bag.push(*ii);
	}

	for (unsigned iter = 1; iter <= iterations && is_terminate == false; ++iter) {
		CdTime.start();
	
		params.Gmax_new.reset();
		params.Gnorm1_new.reset();

		if (useshrink) {
			active_set.clear();
			for ( auto &gg: cur_bag)
			{
				active_set.push_back(gg);
			}
		}
		if (shuffleSamples)
		{
			if (useshrink) {
				std::shuffle(active_set.begin(), active_set.end(), gen);
			} else {
				std::shuffle(variables.begin(), variables.end(), gen);
			}
		}

		auto ln = galois::loopname("PrimalCD");
		auto wl = galois::wl<galois::WorkList::dChunkedFIFO<32>>();

		if (useshrink) {
			cur_bag.clear();
			printf("active set size: %d\n", active_set.size());
			galois::for_each(active_set.begin(), active_set.end(), Lasso_CD(g_train, &params, &cur_bag), ln, wl);
		} else {
			galois::for_each(variables.begin(), variables.end(), Lasso_CD(g_train, &params), ln, wl);
		}

		if (useshrink == true )
		{
			size_t active_size = std::distance(cur_bag.begin(), cur_bag.end());
			double Gmax_local = params.Gmax_new.reduce();
			double Gnorm1_local = params.Gnorm1_new.reduce();
			if ( iter == 1 )
				params.Gnorm1_init = Gnorm1_local;
			printf("gnorm1: %lf, Gmax_new: %lf\n", Gnorm1_local, Gmax_local);
			if ( Gnorm1_local <= tol*(params.Gnorm1_init))
			{
				cur_bag.clear();
				for ( auto ii = variables.begin(), ei = variables.end(); ii!=ei ; ii++ ) 
					cur_bag.push(*ii);

				params.Gmax_old = std::numeric_limits<double>::max();
				tol = tol*0.1;
			}
			else
				params.Gmax_old = Gmax_local;
		}

		CdTime.stop();
		accumTimer.stop();

		std::cout << "iter " <<  iter << " walltime " <<  CdTime.get()/1e3;

		std::cout.precision(10);
		if ( printObjective )
			std::cout << " f " << getPrimalObjective(g_train, trainingSamples);
		if ( printAccuracy )
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

//#define CACHING
// cd for the subproblem of glmenet for L1R-LR
struct glmnet_cd { // {{{
#ifdef CACHING
	typedef int tt_needs_per_iter_alloc;
#endif
	typedef int tt_does_not_need_aborts;
	typedef GLMNET_parameters param_t;
	Graph& g_train;
	GLMNET_parameters &params;
	Bag& cd_bag;
	size_t nr_samples;
	double nu;
	glmnet_cd(Graph &_g, param_t &_p, Bag &bag, size_t _nr_samples): g_train(_g), params(_p), cd_bag(bag), nr_samples(_nr_samples), nu(1e-12){}
	void operator()(GNode feat_j, galois::UserContext<GNode>& ctx){
		auto &j_data = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
		auto &H = j_data.Hdiag;
		auto &Grad_j = j_data.Grad;
		auto &wpd_j = j_data.wpd;
		auto &w_j = j_data.w;
		double G = Grad_j + (wpd_j-w_j)*nu;
#ifdef CACHING
		galois::PerIterAllocTy& alloc = ctx.getPerIterAlloc();
                
                ptrdiff_t size = std::distance(g_train.edge_begin(feat_j, galois::MethodFlag::UNPROTECTED), g_train.edge_end(feat_j, galois::MethodFlag::UNPROTECTED));
                double** xTdAddrs = (double**) alloc.allocate(sizeof(*xTdAddrs) * size);
                double* xTds = (double*) alloc.allocate(sizeof(*xTds) * size);
                double* xijs = (double*) alloc.allocate(sizeof(*xijs) * size);
#endif
                int ii = 0;
		for(auto &edge : g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
			auto &x_ij = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
#ifdef CACHING
                        xijs[ii] = x_ij;
#endif
			auto &self = g_train.getData(g_train.getEdgeDst(edge), galois::MethodFlag::UNPROTECTED);
#ifdef CACHING
                        xTdAddrs[ii] = &self.xTd;
                        xTds[ii] = *xTdAddrs[ii];
			G += xijs[ii]*self.D*xTds[ii];
#else
			G += x_ij*self.D*self.xTd;
#endif
                        ii += 1;
		}
		double Gp = G+1;
		double Gn = G-1;
		double violation = 0;
		if(wpd_j == 0) {
			if(Gp < 0) violation = -Gp;
			else if(Gn > 0) violation = Gn;
			else if(Gp>params.QP_Gmax_old/nr_samples && Gn < -params.QP_Gmax_old/nr_samples) {
				return;
			} 
		} else if(wpd_j > 0) 
			violation = fabs(Gp);
		else 
			violation = fabs(Gn);
		cd_bag.push(feat_j);
		params.QP_Gmax_new.update(violation);
		params.QP_Gnorm1_new.update(violation);
		double z = 0;
		if(Gp < H*wpd_j) z = -Gp/H;
		else if(Gn > H*wpd_j) z = -Gn/H;
		else z = -wpd_j;
		if(fabs(z) < 1.0e-12)
			return;
		z = std::min(std::max(z,-10.0),10.0);
		wpd_j += z;
                ii = 0;
		for(auto &edge : g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
#ifdef CACHING
                        auto x_ij = xijs[ii];
#else
			auto x_ij = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
#endif
#ifdef CACHING
			*xTdAddrs[ii] = xTds[ii] + x_ij*z;
#else
			g_train.getData(g_train.getEdgeDst(edge), galois::MethodFlag::UNPROTECTED).xTd += x_ij*z;
#endif
                        ii += 1;
		}
	}
}; // }}}

struct glmnet_qp_construct { // {{{
	typedef GLMNET_parameters param_t;
	Graph& g_train;
	GLMNET_parameters &params;
	Bag& cd_bag;
	size_t nr_samples;
	double nu;
	glmnet_qp_construct(Graph &_g, param_t &_p, Bag& bag, size_t _nr_samples): g_train(_g), params(_p), cd_bag(bag), nr_samples(_nr_samples), nu(1e-12){}
	void operator()(GNode feat_j, galois::UserContext<GNode>& ctx){
		auto &j_data = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
		auto &w_j = j_data.w;
		auto &Hdiag_j = j_data.Hdiag;
		auto &Grad_j = j_data.Grad;
		auto &xjneg_sum_j = j_data.xjneg_sum;
		Hdiag_j = nu; Grad_j = 0;
		double tmp = 0;
		for(auto &edge: g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
			auto x_ij = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
			auto &self = g_train.getData(g_train.getEdgeDst(edge), galois::MethodFlag::UNPROTECTED);
			Hdiag_j += x_ij*x_ij*self.D;
			tmp += x_ij*self.tau;
		}
		Grad_j = -tmp + xjneg_sum_j;

		double Gp = Grad_j+1;
		double Gn = Grad_j-1;
		double violation = 0;
		if(w_j == 0) {
			if(Gp < 0) violation = -Gp;
			else if(Gn > 0) violation = Gn;
			//outer-level shrinking
			else if(Gp > params.Gmax_old/nr_samples && Gn <-params.Gmax_old/nr_samples) {
				return;
			}

		}
		else if(w_j > 0)
			violation = fabs(Gp);
		else
			violation = fabs(Gn);
		cd_bag.push(feat_j);
		params.Gmax_new.update(violation);
		params.Gnorm1_new.update(violation);
	}
}; // }}}
void run_newGLMNET(Graph& g_train, Graph& g_test, std::mt19937& gen, std::vector<GNode>& trainingSamples, std::vector<GNode>& testingSamples) { // {{{
	galois::TimeAccumulator accumTimer;
	accumTimer.start();
	galois::StatTimer glmnetTime("GLMENT Time");
	galois::StatTimer cdTime("CD Time");
	galois::StatTimer FirstTime("First_Time");
	galois::StatTimer SecondTime("Second_Time");
	galois::StatTimer ThirdTime("Third_Time");

	galois::StatTimer ActiveSetTime("Third_Time");

	unsigned max_newton_iter = fixedIterations? fixedIterations: maxIterations;
	unsigned max_cd_iter = 1000;
	unsigned max_num_linesearch = 20;
	double nu = 1e-12;
	double inner_eps = 0.01;
	double sigma = 0.01;

	double C[3] = {creg,0,creg};

	std::vector<GNode> variables(g_train.begin()+NUM_SAMPLES, g_train.end());

	// initialization {{{
	for(auto &inst_node: trainingSamples) {
		auto &self = g_train.getData(inst_node);
		self.y = self.y > 0 ? 1: -1;
		self.exp_wTx = 0.0;
	}
	double w_norm = 0;
	for(auto &feat_j: variables) {
		auto &j_data = g_train.getData(feat_j);
		double &w_j = j_data.w;
		double &wpd_j = j_data.wpd;
		double &xjneg_sum_j = j_data.xjneg_sum;
		w_norm += fabs(w_j);
		wpd_j = w_j;
		xjneg_sum_j = 0;
		for(auto edge: g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
			auto x_ij = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
			auto &self = g_train.getData(g_train.getEdgeDst(edge), galois::MethodFlag::UNPROTECTED);
			self.exp_wTx += w_j*x_ij;
			if(self.y == -1) 
				xjneg_sum_j += creg*x_ij;
		}
	}
	double xx = 0;
	for(auto &feat_j: variables)
		xx += g_train.getData(feat_j).xjneg_sum;
	double cc = creg;
	printf("creg %lf init xx %lf\n", cc, xx);

	for(auto &inst_node: trainingSamples) {
		auto &self = g_train.getData(inst_node, galois::MethodFlag::UNPROTECTED);
		self.exp_wTx = exp(self.exp_wTx);
		double tau_tmp = 1.0/(1.0+self.exp_wTx);
		self.tau = creg*tau_tmp;
		self.D = creg*self.exp_wTx*tau_tmp*tau_tmp;
	} //}}}

	int newton_iter = 0;
	Bag cur_bag; // used for outerlevel active set
	std::vector<GNode> active_set;
	GLMNET_parameters params;
	params.Gmax_old = std::numeric_limits<double>::max();
	size_t nr_samples = trainingSamples.size();

	while(newton_iter < max_newton_iter)
	{
		glmnetTime.start();

		cur_bag.clear();
		active_set.clear();
		for(auto &feat_j : variables)
			active_set.push_back(feat_j);
		params.Gmax_new.reset();
		params.Gnorm1_new.reset();

		//if(shuffleSamples) std::shuffle(active_set.begin(), active_set.end(), gen);

		FirstTime.start();

		// Compute Newton direction -- Hessian and Gradient
		auto ln = galois::loopname("GLMENT-QPconstruction");
		auto wl = galois::wl<galois::WorkList::dChunkedFIFO<32>>();
		galois::for_each(active_set.begin(), active_set.end(), glmnet_qp_construct(g_train, params, cur_bag, nr_samples), ln, wl);
		/*
		galois::do_all(active_set.begin(), active_set.end(),
		[&](GNode feat_j) { // {{{
			auto &j_data = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
			auto &w_j = j_data.w;
			auto &Hdiag_j = j_data.Hdiag;
			auto &Grad_j = j_data.Grad;
			auto &xjneg_sum_j = j_data.xjneg_sum;
			Hdiag_j = nu; Grad_j = 0;
			double tmp = 0;
			for(auto &edge: g_train.out_edges(feat_j, galois::MethodFlag::UNPROTECTED)) {
				auto x_ij = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
				auto &self = g_train.getData(g_train.getEdgeDst(edge), galois::MethodFlag::UNPROTECTED);
				Hdiag_j += x_ij*x_ij*self.D;
				tmp += x_ij*self.tau;
			}
			Grad_j = -tmp + xjneg_sum_j;

			double Gp = Grad_j+1;
			double Gn = Grad_j-1;
			double violation = 0;
			if(w_j == 0) {
				if(Gp < 0) violation = -Gp;
				else if(Gn > 0) violation = Gn;
				//outer-level shrinking
				else if(Gp > params.Gmax_old/nr_samples && Gn <-params.Gmax_old/nr_samples) {
					return;
				}

			}
			else if(w_j > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);
			cur_bag.push(feat_j);
			params.Gmax_new.update(violation);
			params.Gnorm1_new.update(violation);
		}//}}}
		);
		*/



		double tmp_Gnorm1_new = params.Gnorm1_new.reduce();
		if(newton_iter == 0)
			params.Gnorm1_init = tmp_Gnorm1_new;
		params.Gmax_old = params.Gmax_new.reduce();
		FirstTime.stop();
	
		ActiveSetTime.start();

		// Compute Newton direction -- Coordinate Descet for QP
		cdTime.start();
		params.QP_Gmax_old = std::numeric_limits<double>::max();
		galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode &inst_node) {g_train.getData(inst_node, galois::MethodFlag::UNPROTECTED).xTd = 0.0;});
		auto init_original_active_set = [&]{
			active_set.clear();
			for(auto &feat_j : cur_bag)
				active_set.push_back(feat_j);
		};
		init_original_active_set();
		ActiveSetTime.stop();


		int original_active_size = active_set.size();
		int cd_iter = 0;
		Bag cd_bags[2];
		std::copy(active_set.begin(), active_set.end(), std::back_inserter(cd_bags[0]));

		double grad_norm = 0, H_norm = 0, xx = 0;
		SecondTime.start();
		while(cd_iter < max_cd_iter) { //{{{
			params.QP_Gmax_new.reset();
			params.QP_Gnorm1_new.reset();
                        //XXX
			//if(shuffleSamples)
			//	std::shuffle(active_set.begin(), active_set.end(), gen);
			auto ln = galois::loopname("GLMENT-CDiteration");
			auto wl = galois::wl<galois::WorkList::dChunkedFIFO<32>>();
			//galois::for_each(active_set.begin(), active_set.end(), glmnet_cd(g_train, params, cd_bag, nr_samples), ln, wl);
			galois::for_each_local(cd_bags[0], glmnet_cd(g_train, params, cd_bags[1], nr_samples), ln, wl);

			cd_iter++;
			double tmp_QP_Gmax_new = params.QP_Gmax_new.reduce();
			double tmp_QP_Gnorm1_new = params.QP_Gnorm1_new.reduce();
			//active_set.clear();
			//for(auto &feat_j: cd_bag)
			//	active_set.push_back(feat_j);
			//cd_bag.clear();
			if(tmp_QP_Gmax_new <= inner_eps*params.Gnorm1_init){
				if (std::distance(cd_bags[1].begin(), cd_bags[1].end()) == original_active_size)
					break;
				//inner stopping
				if(active_set.size() == original_active_size)
					break;
				//active set reactivation
				else { 
					//init_original_active_set();
					std::copy(active_set.begin(), active_set.end(), std::back_inserter(cd_bags[0]));
					params.QP_Gmax_old = std::numeric_limits<double>::max();
				}
			} else {
				params.QP_Gmax_old = tmp_QP_Gmax_new;
			}
			cd_bags[0].clear();
			std::swap(cd_bags[0], cd_bags[1]);
		}//}}}
		cdTime.stop();
		SecondTime.stop();

		ThirdTime.start();
		// Perform Line Search 
		// {{{
		galois::GAccumulator<double> delta_acc, w_norm_acc;
		galois::do_all(variables.begin(), variables.end(), //{{{
				[&](GNode &feat_j){
				auto &self = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
				delta_acc.update(self.Grad*(self.wpd-self.w));
				if(self.wpd != 0) w_norm_acc.update(fabs(self.wpd));
				});//}}}
		double w_norm_new = w_norm_acc.reduce();
		double delta = delta_acc.reduce()+(w_norm_new-w_norm);

		galois::GAccumulator<double> tmp_acc;
		galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode &inst_node) {
			auto &self = g_train.getData(inst_node, galois::MethodFlag::UNPROTECTED);
			if(self.y == -1) tmp_acc.update(creg*self.xTd);
		});
		double negsum_xTd = tmp_acc.reduce();

		int num_linesearch = 0;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++ ){
			double cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;
			tmp_acc.reset();
			galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode &inst_node){
				auto &self = g_train.getData(inst_node, galois::MethodFlag::UNPROTECTED);
				double exp_xTd = exp(self.xTd);
				self.exp_wTx_new = self.exp_wTx*exp_xTd;
				tmp_acc.update(creg*log((1+self.exp_wTx_new)/(exp_xTd+self.exp_wTx_new)));
			});
			cond += tmp_acc.reduce();
			if(cond <= 0.0) {
				w_norm = w_norm_new;
				galois::do_all(variables.begin(), variables.end(), [&](GNode &feat_j){
						auto &self = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
						self.w = self.wpd;
					});
				galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode &inst_i){
						auto &self = g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED);
						self.exp_wTx = self.exp_wTx_new;
						double tau_tmp = 1/(1+self.exp_wTx);
						self.tau = creg*tau_tmp;
						self.D = creg*self.exp_wTx*tau_tmp*tau_tmp;
					});
				break;
			} else {
				w_norm_new = 0;
				tmp_acc.reset();
				galois::do_all(variables.begin(), variables.end(), [&](GNode &feat_j){
						auto &self = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
						self.wpd = (self.w+self.wpd)*0.5;
						if(self.wpd != 0) tmp_acc.update(fabs(self.wpd));
					});
				w_norm_new = tmp_acc.reduce();
				delta *= 0.5;
				negsum_xTd *= 0.5;
				galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode &inst_i){ g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED).xTd *=0.5;});
			}
		}
		if(num_linesearch >= max_num_linesearch) {
			galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode &inst_i){ g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED).exp_wTx = 0;});

			galois::do_all(variables.begin(), variables.end(), [&](GNode &feat_j){
					auto &self = g_train.getData(feat_j, galois::MethodFlag::UNPROTECTED);
					if(self.w != 0) {
						for(auto &edge: g_train.out_edges(feat_j)) {
						auto &x_ij = g_train.getEdgeData(edge, galois::MethodFlag::UNPROTECTED);
						g_train.getData(g_train.getEdgeDst(edge), galois::MethodFlag::UNPROTECTED).exp_wTx += self.w*x_ij;
						}
					}
				});
			galois::do_all(trainingSamples.begin(), trainingSamples.end(), [&](GNode &inst_i){ auto &exp_wTx = g_train.getData(inst_i, galois::MethodFlag::UNPROTECTED).exp_wTx; exp_wTx = exp(exp_wTx);});
		}
		//}}} // end of line search

		ThirdTime.stop();
		if(cd_iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		glmnetTime.stop();
		accumTimer.stop();

		printf("iter %d walltime %.6f cdtime %.6f cd-iters %d firsttime %.6f secondtime %.6f thirdtime %.6f", newton_iter, glmnetTime.get()/1e3, cdTime.get()/1e3, cd_iter, FirstTime.get()/1e3, SecondTime.get()/1e3, ThirdTime.get()/1e3);
		if(printObjective) {
			printf(" f %.6f", getPrimalObjective(g_train, trainingSamples));
		}
		if(printAccuracy) {
			printf(" accuracy %.6f", getNumCorrect(g_test, testingSamples, g_train)/(double)testingSamples.size());
		}
		puts("");
	}

} // }}}



int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;
 
  Graph g_train, g_test;
  // Load Training Data
  galois::graphs::readGraph(g_train, inputTrainGraphFilename);
  if ( algoType == AlgoType::CDLasso )
	  NUM_SAMPLES = loadb(g_train, inputTrainLabelFilename);
  else {
	  NUM_SAMPLES = loadLabels(g_train, inputTrainLabelFilename);
  }
  if ( algoType != AlgoType::CDLasso and algoType != AlgoType::GLMNETL1RLR)
	  initializeVariableCounts(g_train);
  NUM_VARIABLES = g_train.size() - NUM_SAMPLES;
  assert(NUM_SAMPLES > 0 && NUM_VARIABLES > 0);

  // Load Testing Data
  galois::graphs::readGraph(g_test, inputTestGraphFilename);
  if ( algoType == AlgoType::CDLasso )
	  NUM_TEST_SAMPLES = loadb(g_test, inputTestLabelFilename);
  else
	  NUM_TEST_SAMPLES = loadLabels(g_test, inputTestLabelFilename);
//  NUM_TEST_SAMPLES = loadLabels(g_test, inputTestLabelFilename);
  if(algoType != AlgoType::CDLasso and algoType != AlgoType::GLMNETL1RLR)
	  initializeVariableCounts(g_test);
  NUM_TEST_VARIABLES = g_test.size() - NUM_TEST_SAMPLES;
  assert(NUM_TEST_SAMPLES > 0);

  //put samples in a list and shuffle them
  std::random_device rd;
  std::mt19937 gen(SEED == ~0U ? rd() : SEED);

  std::vector<GNode> trainingSamples(g_train.begin(), g_train.begin() + NUM_SAMPLES);
  std::vector<GNode> testingSamples(g_test.begin(), g_test.begin()+NUM_TEST_SAMPLES);

  printParameters(trainingSamples, testingSamples);
  if (printAccuracy) {
    std::cout << "Initial";
    if (printAccuracy) {
      std::cout << " Accuracy: " << getNumCorrect(g_test, testingSamples, g_train) / (double) testingSamples.size();
    }
    std::cout << "\n";
  }
  
  galois::preAlloc(numThreads * 10);
  galois::reportPageAlloc("MeminfoPre");
  galois::StatTimer timer;
  timer.start();
  switch (algoType) {
/*    case AlgoType::SGDL1: runPrimalSgd(g_train, g_test, gen, trainingSamples, testingSamples); break;
	case AlgoType::SGDL2: runPrimalSgd(g_train, g_test, gen, trainingSamples, testingSamples); break;
	case AlgoType::SGDLR: runPrimalSgd(g_train, g_test, gen, trainingSamples, testingSamples); break;*/
    case AlgoType::DCDL1: runDCD(g_train, g_test, gen, trainingSamples, testingSamples); break;
	case AlgoType::DCDL2: runDCD(g_train, g_test, gen, trainingSamples, testingSamples); break;
	case AlgoType::DCDLR: runDCD(g_train, g_test, gen, trainingSamples, testingSamples); break;
	case AlgoType::CDLasso: runCD(g_train, g_test, gen, trainingSamples, testingSamples); break;
	case AlgoType::GLMNETL1RLR: run_newGLMNET(g_train, g_test, gen, trainingSamples, testingSamples); break;
    default: abort();
  }
  timer.stop();
  galois::reportPageAlloc("MeminfoPost");

  return 0;
}

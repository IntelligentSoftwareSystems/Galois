#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>

namespace cll = llvm::cl;

static const char* name = "Belief propagation";
static const char* desc = "Performs belief propagation on Ising Grids";
static const char* url  = 0;

static cll::opt<int> algo("algo", cll::desc("Algorithm"), cll::init(1));
static cll::opt<int> N(cll::Positional, cll::desc("<N>"), cll::Required);
static cll::opt<int> hardness(cll::Positional, cll::desc("<hardness>"),
                              cll::Required);
static cll::opt<int> seed(cll::Positional, cll::desc("<seed>"), cll::Required);
static cll::opt<int> MaxIterations(cll::Positional,
                                   cll::desc("<max iterations>"),
                                   cll::Required);

static const double DAMPING = 0.2;
// static const double TOL = 1e-10;

template <typename NodeTy, typename EdgeTy>
struct BipartiteGraph
    : public galois::graphs::MorphGraph<NodeTy, EdgeTy, true> {
  typedef galois::graphs::MorphGraph<NodeTy, EdgeTy, true> Super;
  typedef std::vector<typename Super::GraphNode> NodeList;
  NodeList factors;
  NodeList variables;

  void addNode(const typename Super::GraphNode& n, bool isFactor,
               galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
    if (isFactor) {
      factors.push_back(n);
    } else {
      variables.push_back(n);
    }
    Super::addNode(n, mflag);
  }

  void addNode(const typename Super::GraphNode& n,
               galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
    assert(0 && "Not supported in Bipartite Graph");
    abort();
  }

  void removeNode(typename Super::GraphNode n,
                  galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
    assert(0 && "Not supported in Bipartite Graph");
    abort();
  }
};

enum { BP_SEQ = 0, BP_SEQ_RAND, BP_PAR, BP_MAX_RES };

//! Probabilities specialized for pairwise binary case and consider unary case
//! as another special case
struct Prob {
  double v[4];

  Prob() {
    for (int i = 0; i < 4; ++i)
      v[i] = 1;
  }

  Prob(double p00, double p01, double p10, double p11) {
    v[0] = p00;
    v[1] = p01;
    v[2] = p10;
    v[3] = p11;
  }

  Prob(double p0, double p1) {
    v[0] = p0;
    v[1] = p1;
    v[2] = 1;
    v[3] = 1;
  }

  void operator*=(const Prob& other) {
    for (int i = 0; i < 4; ++i)
      v[i] *= other.v[i];
  }

  void pow(const double& other) {
    for (int i = 0; i < 4; ++i)
      v[i] = std::pow(v[i], other);
  }
};

template <int Type>
struct BP {
  struct Node {
    //! Pairwise potentials
    Prob prob;
    Node() {}
    Node(double p00, double p01, double p10, double p11)
        : prob(p00, p01, p10, p11) {}
    Node(double p0, double p1) : prob(p0, p1) {}
  };

  struct Edge {
    Prob message;
    Prob message_new;
    Edge() {} // For Graph
  };

  typedef BipartiteGraph<Node, Edge> Graph;
  typedef typename Graph::GraphNode GraphNode;
  typedef std::vector<std::pair<GraphNode, GraphNode>> SequentialSchedule;

  Graph& graph;
  SequentialSchedule m_sequential_schedule;

  BP(Graph& g) : graph(g) {}

  // initialize messages to 1
  // initialize residual to 0

  Prob calculateIncomingMessageProduct(const GraphNode& factor,
                                       const GraphNode& var_i) {
    Prob prod(graph.getData(factor).prob);

    for (typename Graph::edge_iterator ii = graph.edge_begin(factor),
                                       ei = graph.edge_end(factor);
         ii != ei; ++ii) {
      GraphNode var = graph.getEdgeDst(ii);

      if (var == var_i)
        continue;

      Prob prod_j;
      for (typename Graph::edge_iterator jj = graph.edge_begin(var),
                                         ej = graph.edge_end(var);
           jj != ej; ++jj) {
        GraphNode factor_j = graph.getEdgeDst(jj);
        if (factor_j == factor)
          continue;
        prod_j *= graph.getEdgeData(jj).message;
      }

      // XXX
      prod *= prod_j;
    }

    return prod;
  }

  void calculateMessage(const GraphNode& var, const GraphNode& factor) {
    Prob marg;

    if (std::distance(graph.edge_begin(var), graph.edge_end(var)) == 1) {
      marg = graph.getData(factor).prob;
    } else {
      Prob& prob = graph.getData(factor).prob;
      prob       = calculateIncomingMessageProduct(factor, var);

      // XXX compute marginal and normalize
    }

    graph.getEdgeData(graph.findEdge(var, factor)).message_new = marg;
  }

  void updateMessage(const GraphNode& var, const GraphNode& factor) {
    Edge& edge = graph.getEdgeData(graph.findEdge(var, factor));
    if (DAMPING) {
      Prob m = edge.message_new;
      m.pow(1 - DAMPING);
      edge.message.pow(DAMPING);
      edge.message *= m;
    } else {
      edge.message = edge.message_new;
    }
  }

  void maxResidualScheduling() {}

  void parallelScheduling() {
    for (typename Graph::NodeList::iterator ii = graph.variables.begin(),
                                            ei = graph.variables.end();
         ii != ei; ++ii) {
      for (typename Graph::edge_iterator jj = graph.edge_begin(*ii),
                                         ej = graph.edge_end(*ii);
           jj != ej; ++jj) {
        calculateMessage(*ii, graph.getEdgeDst(jj));
      }
    }

    for (typename Graph::NodeList::iterator ii = graph.variables.begin(),
                                            ei = graph.variables.end();
         ii != ei; ++ii) {
      for (typename Graph::edge_iterator jj = graph.edge_begin(*ii),
                                         ej = graph.edge_end(*ii);
           jj != ej; ++jj) {
        updateMessage(*ii, graph.getEdgeDst(jj));
      }
    }
  }

  void generateSequentialSchedule() {
    m_sequential_schedule.clear();

    for (typename Graph::NodeList::iterator ii = graph.factors.begin(),
                                            ei = graph.factors.end();
         ii != ei; ++ii) {
      for (typename Graph::edge_iterator jj = graph.edge_begin(*ii),
                                         ej = graph.edge_end(*ii);
           jj != ej; ++jj) {
        m_sequential_schedule.push_back(
            std::make_pair(graph.getEdgeDst(jj), *ii));
      }
    }
  }

  void sequentialScheduling(bool randomize) {
    if (randomize)
      std::random_shuffle(m_sequential_schedule.begin(),
                          m_sequential_schedule.end());

    for (typename SequentialSchedule::iterator
             ii = m_sequential_schedule.begin(),
             ei = m_sequential_schedule.end();
         ii != ei; ++ii) {
      calculateMessage(ii->first, ii->second);
      updateMessage(ii->first, ii->second);
    }
  }

  bool isConverged() { return false; }

  void operator()() {
    if (Type == BP_SEQ || Type == BP_SEQ_RAND) {
      generateSequentialSchedule();
    }

    for (int iteration = 0; iteration < MaxIterations; ++iteration) {
      switch (Type) {
      case BP_MAX_RES:
        maxResidualScheduling();
        break;
      case BP_PAR:
        parallelScheduling();
        break;
      case BP_SEQ_RAND:
        sequentialScheduling(true);
        break;
      case BP_SEQ:
        sequentialScheduling(false);
        break;
      default:
        abort();
      }

      if (isConverged()) {
        break;
      }
    }

    std::cout << "Did not converge\n";
  }
};

struct Exact {};

#if 0
/**
 * GRAPHLAB implementation of Gaussiabn Belief Propagation Code See
 * algrithm description and explanation in: Danny Bickson, Gaussian
 * Belief Propagation: Theory and Application. Ph.D. Thesis. The
 * Hebrew University of Jerusalem. Submitted October 2008.
 * http://arxiv.org/abs/0811.2518 By Danny Bickson, CMU. Send any bug
 * fixes/reports to bickson@cs.cmu.edu Code adapted to GraphLab by
 * Joey Gonzalez, CMU July 2010
 *
 * Functionality: The code solves the linear system Ax = b using
 * Gaussian Belief Propagation. (A is either square matrix or
 * skinny). A assumed to be full column rank.  Algorithm is described
 * in Algorithm 1, page 14 of the above Phd Thesis.
 *
 * If you are using this code, you should cite the above reference. Thanks!
 */
struct GBP {
  struct Node: public BaseNode {
    double prev_x;
    double prior_mean;
    double prev_prec;
    double cur_prec;
    Node(double b, double a, double w): BaseNode(b, a, w),
      prior_mean(b), prev_prec(w), cur_prec(0) { }
  };

  struct Edge {
    double weight;
    double mean;
    double prec;
    Edge() { } // Graph requires this
    Edge(double w): weight(w), mean(0), prec(0) { }
  };
  
  typedef galois::graphs::MorphGraph<Node,Edge,true> Graph;

  Graph& graph;

  GBP(Graph& g): graph(g) { }

  void operator()(const Graph::GraphNode& src) {
    Node& node = src.getData(galois::MethodFlag::UNPROTECTED);

    node.prev_x = node.x;
    node.prev_prec = node.cur_prec;

    double mu_i = node.prior_mean;
    double J_i = node.weight;
    assert(J_i != 0);

    // Incoming edges
    for (Graph::neighbor_iterator dst = graph.neighbor_begin(src, galois::MethodFlag::WRITE),
        edst = graph.neighbor_end(src, galois::MethodFlag::WRITE); dst != edst; ++dst) {
      const Edge& edge = graph.getEdgeData(*dst, src, galois::MethodFlag::UNPROTECTED);
      mu_i += edge.mean;
      J_i += edge.prec;
    }

    assert(J_i != 0);
    node.x = mu_i / J_i;
    assert(!isnan(node.x));
    node.cur_prec = J_i;

    for (Graph::neighbor_iterator dst = graph.neighbor_begin(src, galois::MethodFlag::UNPROTECTED),
        edst = graph.neighbor_end(src, galois::MethodFlag::UNPROTECTED); dst != edst; ++dst) {
      Edge& inEdge = graph.getEdgeData(*dst, src, galois::MethodFlag::UNPROTECTED);
      Edge& outEdge = graph.getEdgeData(src, dst, galois::MethodFlag::UNPROTECTED);

      double mu_i_j = mu_i - inEdge.mean;
      double J_i_j = J_i - inEdge.prec;

      outEdge.mean = -(outEdge.weight * mu_i_j / J_i_j);
      outEdge.prec = -((outEdge.weight * outEdge.weight) / J_i_j);

      //double priority = fabs(node.cur_prec) + 1e-5;

      //ctx.push(*dst);
    }
  }

  void operator()() {
    for (int i = 0; i < MaxIterations; ++i) {
      std::for_each(graph.active_begin(), graph.active_end(), *this);
      double r = relativeResidual(graph);
      std::cout << "RE " << r << "\n";
      if (r < TOL)
        return;
    }
    std::cout << "Did not converge\n";
  }
};
// From asynch_GBP.m in gabp-src.zip at
//  http://www.cs.cmu.edu/~bickson/gabp/index.html
struct GBP {
  struct Node: public BaseNode {
    double x_prev;
    double mean; // h(i)
    double prec; // J(i)
    Node(double b, double a, double w): BaseNode(b, a, w),
       prec(0) { }
  };

  struct Edge {
    double weight; // A(i,i)
    double mean;   // Mh = zeros(m, m)
    double prec;   // MJ = zeros(m, m)
    Edge() { } // MorphGraph requires this
    Edge(double w): weight(w), mean(0), prec(0) { }
  };
  
  typedef galois::graphs::MorphGraph<Node,Edge,true> Graph;

  Graph& graph;

  GBP(Graph& g): graph(g) { }

  void operator()(const Graph::GraphNode& src) {
    Node& node = src.getData(galois::MethodFlag::UNPROTECTED);
    
    node.x_prev = node.x;

    node.mean = node.b;
    node.prec = node.weight;

    // Sum up all mean and percision values got from neighbors
    //  h(i) = b(i) + sum(Mh(:,i));  %(7)
    // Variance can not be zero (must be a diagonally dominant matrix)!
    //  assert(A(i,i) ~= 0);
    //  J(i) = A(i,i) + sum(MJ(:,i));
    for (Graph::neighbor_iterator dst = graph.neighbor_begin(src, galois::MethodFlag::WRITE),
        edst = graph.neighbor_end(src, galois::MethodFlag::WRITE); dst != edst; ++dst) {
      const Edge& edge = graph.getEdgeData(*dst, src, galois::MethodFlag::UNPROTECTED);
      node.mean += edge.mean;
      node.prec += edge.prec;
    }

    node.x = node.mean / node.prec;

    // Send message to all neighbors
    //  for j=1:m
    //    if (i ~= j && A(i,j) ~= 0)
    //      h_j = h(i) - Mh(j,i);
    //      J_j = J(i) - MJ(j,i);
    //      assert(A(i,j) == A(j,i));
    //      assert(J_j ~= 0);
    //      Mh(i,j) = (-A(j,i) / J_j)* h_j;
    //      MJ(i,j) = (-A(j,i) / J_j) * A(i,j);
    //    end
    //  end
    for (Graph::neighbor_iterator dst = graph.neighbor_begin(src, galois::MethodFlag::UNPROTECTED),
        edst = graph.neighbor_end(src, galois::MethodFlag::UNPROTECTED); dst != edst; ++dst) {
      Edge& inEdge = graph.getEdgeData(*dst, src, galois::MethodFlag::UNPROTECTED);
      Edge& outEdge = graph.getEdgeData(src, dst, galois::MethodFlag::UNPROTECTED);
      
      double mean_j = node.mean - inEdge.mean;
      double prec_j = node.prec - inEdge.prec;

      outEdge.mean = -inEdge.weight * mean_j / prec_j;
      outEdge.prec = -inEdge.weight * outEdge.weight / prec_j;
      assert(inEdge.weight == outEdge.weight);

      //double priority = fabs(node.cur_prec) + 1e-5;

      //ctx.push(*dst);
    }
  }

  void operator()() {
    std::vector<Graph::GraphNode> elements(graph.size());
    std::copy(graph.active_begin(), graph.active_end(), elements.begin());

    for (int i = 0; i < MaxIterations; ++i) {
      std::random_shuffle(elements.begin(), elements.end());
      std::for_each(elements.begin(), elements.end(), *this);
      double r = relativeResidual(graph);
      std::cout << "RE " << r << "\n";
      if (r < TOL)
        return;
    }
    std::cout << "Did not converge\n";
  }
};
#endif

//! Generate random Ising grid
//!  N*N discrete variables, X_i, \phi(X_i) in {0, 1} (spin)
//!  \phi_ij(X_i, X_j) = e^{\lambda*C} if x_i = x_j or e^{-\lambda*C} otherwise
//!  \lambda in [-0.5, 0.5]
template <typename Graph>
struct GenerateInput {
  typedef typename Graph::GraphNode GraphNode;
  typedef typename Graph::node_data_type node_data_type;
  typedef typename Graph::edge_data_type edge_data_type;

  Graph& graph;
  int hardness;

  double nextRand() { return rand() / (double)RAND_MAX; }

  //! Create a pairwise factor
  void addFactor(int var1, int var2) {
    GraphNode& n1 = graph.variables[var1];
    GraphNode& n2 = graph.variables[var2];

    double lambda = nextRand() - 0.5;
    edge_data_type edge;
    double p00      = exp(lambda * hardness);
    double p01      = exp(-lambda * hardness);
    GraphNode new_n = graph.createNode(node_data_type(p00, p01, p01, p00));
    graph.addNode(new_n, true);
    graph.getEdgeData(graph.addEdge(new_n, n1)) = edge;
    graph.getEdgeData(graph.addEdge(new_n, n2)) = edge;
    graph.getEdgeData(graph.addEdge(n1, new_n)) = edge;
    graph.getEdgeData(graph.addEdge(n2, new_n)) = edge;
  }

  //! Create a unary factor
  void addFactor(int var) {
    GraphNode& n = graph.variables[var];
    double h     = nextRand();
    edge_data_type edge;
    GraphNode new_n = graph.createNode(node_data_type(exp(-h), exp(h)));
    graph.addNode(new_n, true);
    graph.getEdgeData(graph.addEdge(new_n, n)) = edge;
    graph.getEdgeData(graph.addEdge(n, new_n)) = edge;
  }

  GenerateInput(Graph& g, int N, int h, int seed) : graph(g), hardness(h) {
    srand(seed);

    // Create variables
    for (int i = 0; i < N * N; ++i) {
      GraphNode n = graph.createNode(node_data_type());
      graph.addNode(n, false);
    }

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        if (i >= 1)
          addFactor(i * N + j, (i - 1) * N + j);
        if (j >= 1)
          addFactor(i * N + j, i * N + (j - 1));
        addFactor(i * N + j);
      }
    }
  }
};

template <typename Algo>
static void start(int N, int hardness, int seed) {
  typename Algo::Graph g;
  GenerateInput<typename Algo::Graph>(g, N, hardness, seed);

  galois::StatTimer T;
  T.start();
  Algo algo(g);
  algo();
  T.stop();
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);

  switch (algo) {
  case 3:
    std::cout << "Using BP-MAX-RES\n";
    start<BP<BP_MAX_RES>>(N, hardness, seed);
    break;
  case 2:
    std::cout << "Using BP-PAR\n";
    start<BP<BP_PAR>>(N, hardness, seed);
    break;
  case 1:
    std::cout << "Using BP-SEQ-RAND\n";
    start<BP<BP_SEQ_RAND>>(N, hardness, seed);
    break;
  case 0:
  default:
    std::cout << "Using BP-SEQ\n";
    start<BP<BP_SEQ>>(N, hardness, seed);
    break;
  }

  return 0;
}

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

#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"

#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <cstdlib>
#include <iostream>

#include <math.h>

using namespace std;
using namespace galois::worklists;

namespace cll = llvm::cl;

static const char* name = "Survey Propagation";
static const char* desc = "Solves SAT problems using survey propagation";
static const char* url  = "survey_propagation";

static cll::opt<int> seed(cll::Positional, cll::desc("<seed>"), cll::Required);
static cll::opt<int> M(cll::Positional, cll::desc("<num variables>"),
                       cll::Required);
static cll::opt<int> N(cll::Positional, cll::desc("<num clauses>"),
                       cll::Required);
static cll::opt<int> K(cll::Positional, cll::desc("<variables per clause>"),
                       cll::Required);

// SAT problem:
// variables Xi E {0,1}, i E {1 .. N), M constraints
// constraints are or clauses of variables or negation of variables
// clause a has variables i1...iK, J^a_ir E {-+1}
// zi1 = J^a_ir * xir

// Graph form:
// N variables each get a variable node (circles in paper) (SET X, i,j,k...)
// M clauses get a function node (square in paper) (set A, a,b,c...)
// edge between xi and a if xi appears in a, weighted by J^a_i
// V(i) function nodes a... to which variable node i is connected
// n_i = |V(i)| = degree of variable node
// V+(i) positive edges, V-(i) negative edges (per J^a_i) (V(i) = V+(i) + V-(i))
// V(i)\b set V(i) without b
// given connected Fnode a and Vnode j, V^u_a(j) and V^s_a(j) are neighbors
// which cause j sat or unsat a:
// if (J^a_j = 1): V^u_a(j) = V+(j); V^s_a(j) = V-(j)\a
// if (J^a_j = -1): V^u_a(j) = V-(j); V^s_a(j) = V+(j)\a

// Graph+data:
// survey n_a->i E [0,1]

// implementation
// As a graph
// nodes have:
//  a name
//  a eta product
// edges have:
//  a double survey
//  a bool for sign (inversion of variable)
//  a pi product
// Graph is undirected (for now)

struct SPEdge {
  double eta;
  bool isNegative;

  SPEdge() {}
  SPEdge(bool isNeg) : isNegative(isNeg) {
    eta = (double)rand() / (double)RAND_MAX;
  }
};

struct SPNode {
  bool isClause;
  int name;
  bool solved;
  bool value;
  int t;

  double Bias;

  SPNode(int n, bool b)
      : isClause(b), name(n), solved(false), value(false), t(0) {}
};

typedef galois::graphs::MorphGraph<SPNode, SPEdge, false> Graph;
typedef galois::graphs::MorphGraph<SPNode, SPEdge, false>::GraphNode GNode;

// interesting parameters:
static const double epsilon = 0.000001;
// static const int tmax = 100;
// static int tlimit = 0;

void initialize_random_formula(Graph& graph, int M, int N, int K,
                               std::vector<GNode>& literalsN,
                               std::vector<std::pair<GNode, int>>& clauses) {
  // M clauses
  // N variables
  // K vars per clause

  // build up clauses and literals
  clauses.resize(M);
  literalsN.resize(N);

  for (int m = 0; m < M; ++m) {
    GNode node = graph.createNode(SPNode(m, true));
    graph.addNode(node, galois::MethodFlag::UNPROTECTED);
    clauses[m] = std::make_pair(node, 0);
  }
  for (int n = 0; n < N; ++n) {
    GNode node = graph.createNode(SPNode(n, false));
    graph.addNode(node, galois::MethodFlag::UNPROTECTED);
    literalsN[n] = node;
  }

  for (int m = 0; m < M; ++m) {
    // Generate K unique values
    std::vector<int> touse;
    while (touse.size() != (unsigned)K) {
      // extra complex to generate uniform rand value
      int newK = (int)(((double)rand() / ((double)RAND_MAX + 1)) * (double)(N));
      if (std::find(touse.begin(), touse.end(), newK) == touse.end()) {
        touse.push_back(newK);
        graph.getEdgeData(graph.addEdge(clauses[m].first, literalsN[newK],
                                        galois::MethodFlag::UNPROTECTED)) =
            SPEdge((bool)(rand() % 2));
      }
    }
  }

  // std::random_shuffle(literals.begin(), literals.end());
  // std::random_shuffle(clauses.begin(), clauses.end());
}

void print_formula(Graph& graph, std::vector<std::pair<GNode, int>>& clauses) {
  for (unsigned m = 0; m < clauses.size(); ++m) {
    if (m != 0)
      std::cout << " & ";
    std::cout << "c" << m << "( ";
    GNode N = clauses[m].first;
    for (auto ii : graph.edges(N, galois::MethodFlag::UNPROTECTED)) {
      if (ii != graph.edge_begin(N, galois::MethodFlag::UNPROTECTED))
        std::cout << " | ";
      SPEdge& E = graph.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
      if (E.isNegative)
        std::cout << "-";

      SPNode& V =
          graph.getData(graph.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED);
      std::cout << "v" << V.name;
      if (V.solved)
        std::cout << "[" << (V.value ? 1 : 0) << "]";
      std::cout << "{" << E.eta << "," << V.Bias << "," << (V.value ? 1 : 0)
                << "}";
      std::cout << " ";
    }
    std::cout << " )";
  }
  std::cout << "\n";
}

void print_fixed(Graph& graph, std::vector<GNode>& literalsN) {
  for (unsigned n = 0; n < literalsN.size(); ++n) {
    GNode N   = literalsN[n];
    SPNode& V = graph.getData(N, galois::MethodFlag::UNPROTECTED);
    if (V.solved)
      std::cout << V.name << "[" << (V.value ? 1 : 0) << "]\n";
  }
  std::cout << "\n";
}

int count_fixed(Graph& graph, std::vector<GNode>& literalsN) {
  int retval = 0;
  for (unsigned n = 0; n < literalsN.size(); ++n) {
    GNode N   = literalsN[n];
    SPNode& V = graph.getData(N, galois::MethodFlag::UNPROTECTED);
    if (V.solved)
      ++retval;
  }
  return retval;
}

double eta_for_a_i(Graph& graph, GNode a, GNode i) {
  double etaNew = 1.0;
  // for each j
  for (auto jii : graph.edges(a, galois::MethodFlag::UNPROTECTED)) {
    GNode j = graph.getEdgeDst(jii);
    if (j != i) {
      bool ajNegative =
          graph.getEdgeData(jii, galois::MethodFlag::UNPROTECTED).isNegative;
      double prodP = 1.0;
      double prodN = 1.0;
      double prod0 = 1.0;
      // for each b
      for (auto bii : graph.edges(j, galois::MethodFlag::UNPROTECTED)) {
        GNode b    = graph.getEdgeDst(bii);
        SPEdge Ebj = graph.getEdgeData(bii, galois::MethodFlag::UNPROTECTED);
        if (b != a)
          prod0 *= (1.0 - Ebj.eta);
        if (Ebj.isNegative)
          prodN *= (1.0 - Ebj.eta);
        else
          prodP *= (1.0 - Ebj.eta);
      }
      double PIu, PIs;
      if (ajNegative) {
        PIu = (1.0 - prodN) * prodP;
        PIs = (1.0 - prodP) * prodN;
      } else {
        PIs = (1.0 - prodN) * prodP;
        PIu = (1.0 - prodP) * prodN;
      }
      double PI0 = prod0;
      etaNew *= (PIu / (PIu + PIs + PI0));
    }
  }
  return etaNew;
}

struct EIndexer : public std::unary_function<std::pair<GNode, int>, int> {
  int operator()(const std::pair<GNode, int>& v) { return v.second; }
};

struct ELess {
  bool operator()(const std::pair<GNode, int>& lhs,
                  const std::pair<GNode, int>& rhs) {
    if (lhs.second < rhs.second)
      return true;
    if (lhs.second > rhs.second)
      return false;
    return lhs < rhs;
  }
};
struct EGreater {
  bool operator()(const std::pair<GNode, int>& lhs,
                  const std::pair<GNode, int>& rhs) {
    if (lhs.second > rhs.second)
      return true;
    if (lhs.second < rhs.second)
      return false;
    return lhs > rhs;
  }
};

// return true if converged
void SP_algorithm(Graph& graph, galois::GAccumulator<unsigned int>& nontrivial,
                  galois::GReduceMax<double>& maxBias,
                  galois::GAccumulator<int>& numBias,
                  galois::GAccumulator<double>& sumBias,
                  std::vector<GNode>& literalsN,
                  std::vector<std::pair<GNode, int>>& clauses) {
  // 0) at t = 0, for every edge a->i, randomly initialize the message sigma
  // a->i(t=0) in [0,1] 1) for t = 1 to tmax: 1.1) sweep the set of edges in a
  // random order, and update sequentially the warnings on all the edges of the
  // graph, generating the values sigma a->i (t) using SP_update 1.2) if (|sigma
  // a->i(t) - sigma a->i (t-1) < E on all the edges, the iteration has
  // converged and generated sigma* a->i = sigma a->i(t), goto 2 2) if t = tmax
  // return un-converged.  if (t < tmax) then return the set of fixed point
  // warnings sigma* a->i = sigma a->i (t)

  //  tlimit += tmax;

  using WL = galois::worklists::PerSocketChunkFIFO<1024>;

  galois::reportPageAlloc("MeminfoPre: SP_algorithm for_each");
  galois::for_each(
      galois::iterate(clauses),
      [&](const std::pair<GNode, int>& p, auto& ctx) {
        GNode a = p.first;
        // std::cerr << graph.getData(a).t << " ";
        // if (graph.getData(a, galois::MethodFlag::UNPROTECTED).t >= tlimit)
        //   return;

        //    for (Graph::neighbor_iterator iii = graph.neighbor_begin(a),
        //      	   iee = graph.neighbor_end(a); iii != iee; ++iii)
        //   for (Graph::neighbor_iterator bii = graph.neighbor_begin(*iii),
        // 	     bee = graph.neighbor_end(*iii); bii != bee; ++bii)
        // 	//	for (Graph::neighbor_iterator jii = graph.neighbor_begin(*bii),
        // 	//     	       jee = graph.neighbor_end(*bii);
        // 	//     	     jii != jee; ++jii)
        //{}

        auto a_data = graph.getData(a);
        ++(a_data.t);

        // for each i
        for (auto iii : graph.edges(a, galois::MethodFlag::UNPROTECTED)) {
          GNode i  = graph.getEdgeDst(iii);
          double e = eta_for_a_i(graph, a, i);
          double olde =
              graph.getEdgeData(iii, galois::MethodFlag::UNPROTECTED).eta;
          graph.getEdgeData(iii).eta = e;
          // std::cout << olde << ',' << e << " ";
          if (fabs(olde - e) > epsilon) {
            for (auto bii : graph.edges(i, galois::MethodFlag::UNPROTECTED)) {
              GNode b = graph.getEdgeDst(bii);
              if (a != b) // && graph.getData(b,
                          // galois::MethodFlag::UNPROTECTED).t < tlimit)
                ctx.push(std::make_pair(b, 100 - (int)(100.0 * (olde - e))));
            }
          }
        }
      },
      galois::loopname("update_eta"), galois::wl<WL>());
  galois::reportPageAlloc("MeminfoPost: SP_algorithm for_each");

  maxBias.reset();
  numBias.reset();
  sumBias.reset();
  nontrivial.reset();

  galois::reportPageAlloc("MeminfoPre: SP_algorithm do_all");
  // update_biases
  galois::do_all(
      galois::iterate(literalsN),
      [&](GNode i) {
        SPNode& idata = graph.getData(i, galois::MethodFlag::UNPROTECTED);
        if (idata.solved)
          return;

        double pp1 = 1.0;
        double pp2 = 1.0;
        double pn1 = 1.0;
        double pn2 = 1.0;
        double p0  = 1.0;

        // for each function a
        for (auto aii : graph.edges(i, galois::MethodFlag::UNPROTECTED)) {
          SPEdge& aie = graph.getEdgeData(aii, galois::MethodFlag::UNPROTECTED);

          double etaai = aie.eta;
          if (etaai > epsilon)
            nontrivial += 1;
          if (aie.isNegative) {
            pp2 *= (1.0 - etaai);
            pn1 *= (1.0 - etaai);
          } else {
            pp1 *= (1.0 - etaai);
            pn2 *= (1.0 - etaai);
          }
          p0 *= (1.0 - etaai);
        }
        double pp = (1.0 - pp1) * pp2;
        double pn = (1.0 - pn1) * pn2;

        double BiasP = pp / (pp + pn + p0);
        double BiasN = pn / (pp + pn + p0);
        //    double Bias0 = 1.0 - BiasP - BiasN;

        double d = BiasP - BiasN;
        if (d < 0.0)
          d = BiasN - BiasP;
        idata.Bias  = d;
        idata.value = (BiasP > BiasN);

        assert(!std::isnan(d) && !std::isnan(-d));
        maxBias.update(d);
        numBias += 1;
        sumBias += d;
      },
      galois::loopname("update_biases"));
  galois::reportPageAlloc("MeminfoPost: SP_algorithm do_all");
}

void decimate(Graph& graph, galois::GAccumulator<unsigned int>& nontrivial,
              galois::GReduceMax<double>& maxBias,
              galois::GAccumulator<int>& numBias,
              galois::GAccumulator<double>& sumBias,
              std::vector<GNode>& literalsN) {
  double m       = maxBias.reduce();
  double n       = nontrivial.reduce();
  int num        = numBias.reduce();
  double average = num > 0 ? sumBias.reduce() / num : 0.0;
  std::cout << "NonTrivial " << n << " MaxBias " << m << " Average Bias "
            << average << "\n";
  double d = ((m - average) * 0.25) + average;

  const double limit = d;

  galois::reportPageAlloc("MeminfoPre: decimate");
  // fix_variables
  galois::do_all(
      galois::iterate(literalsN),
      [&](GNode i) {
        SPNode& idata = graph.getData(i);
        if (idata.solved)
          return;
        if (idata.Bias > limit) {
          idata.solved = true;
          // TODO: simplify graph
          // for each b
          for (auto bii : graph.edges(i)) {
            graph.getData(graph.getEdgeDst(bii)).solved = true;
            graph.getData(graph.getEdgeDst(bii)).value  = true;
          }
          graph.removeNode(i);
        }
      },
      galois::loopname("fix_variables"));
  galois::reportPageAlloc("MeminfoPost: decimate");
}

bool survey_inspired_decimation(Graph& graph,
                                galois::GAccumulator<unsigned int>& nontrivial,
                                galois::GReduceMax<double>& maxBias,
                                galois::GAccumulator<int>& numBias,
                                galois::GAccumulator<double>& sumBias,
                                std::vector<GNode>& literalsN,
                                std::vector<std::pair<GNode, int>>& clauses) {
  // 0) Randomize initial conditions for the surveys
  // 1) run SP
  //   if (SP does not converge, return SP UNCONVEREGED and stop
  //   if SP convereges, use fixed-point surveys n*a->i to
  // 2) decimate
  // 2.1) if non-trivial surveys (n != 0) are found, then:
  //   a) compute biases (W+,W-,W0) from PI+,PI-,PI0
  //   b) fix largest |W+ - W-| to x =  W+ > W-
  //   c) clean the graph
  // 2.2) if all surveys are trivial(n = 0), output simplified subformula
  // 4) if solved, output SAT, if no contradiction, continue at 1, if
  // contridiction, stop
  galois::preAlloc(numThreads +
                   100 * (graph.size()) / galois::runtime::pagePoolSize());
  do {
    SP_algorithm(graph, nontrivial, maxBias, numBias, sumBias, literalsN,
                 clauses);
    if (nontrivial.reduce()) {
      std::cout << "DECIMATED\n";
      decimate(graph, nontrivial, maxBias, numBias, sumBias, literalsN);
    } else {
      std::cout << "SIMPLIFIED\n";
      return false;
    }
  } while (true); // while (true);
  return true;
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  srand(seed);
  Graph graph;
  galois::GAccumulator<unsigned int> nontrivial;
  galois::GReduceMax<double> maxBias;
  galois::GAccumulator<int> numBias;
  galois::GAccumulator<double> sumBias;

  std::vector<GNode> literalsN;
  std::vector<std::pair<GNode, int>> clauses;

  initialize_random_formula(graph, M, N, K, literalsN, clauses);
  // print_formula(graph, clauses);
  // build_graph();
  // print_graph();

  std::cout << "Starting...\n";

  galois::StatTimer T;
  T.start();
  survey_inspired_decimation(graph, nontrivial, maxBias, numBias, sumBias,
                             literalsN, clauses);
  T.stop();

  // print_formula(graph, clauses);
  // print_fixed(graph, literalsN);

  std::cout << "Fixed " << count_fixed(graph, literalsN) << " variables\n";

  return 0;
}

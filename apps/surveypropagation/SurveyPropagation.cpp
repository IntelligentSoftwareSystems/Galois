/** Survey propagation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * Single source shortest paths.
 *
 * @author Martin Burtscher <burtscher@txstate.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/util/Accumulator.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <cstdlib>
#include <iostream>

using namespace std;

static const char* name = "Survey Propagation";
static const char* description = "Solves SAT problems using survey propagation\n";
static const char* url = "survey_propagation";
static const char* help = "<seed> <num clauses> <num variables> <k>";


//SAT problem:
//variables Xi E {0,1}, i E {1 .. N), M constraints
//constraints are or clauses of variables or negation of variables
//clause a has variables i1...iK, J^a_ir E {-+1}
//zi1 = J^a_ir * xir

//Graph form:
//N variables each get a variable node (circles in paper) (SET X, i,j,k...)
//M clauses get a function node (square in paper) (set A, a,b,c...)
// edge between xi and a if xi appears in a, weighted by J^a_i
//V(i) function nodes a... to which variable node i is connected
//n_i = |V(i)| = degree of variable node
//V+(i) positive edges, V-(i) negative edges (per J^a_i) (V(i) = V+(i) + V-(i))
//V(i)\b set V(i) without b
//given connected Fnode a and Vnode j, V^u_a(j) and V^s_a(j) are neighbors which cause j sat or unsat a:
// if (J^a_j = 1): V^u_a(j) = V+(j); V^s_a(j) = V-(j)\a
// if (J^a_j = -1): V^u_a(j) = V-(j); V^s_a(j) = V+(j)\a

//Graph+data:
//survey n_a->i E [0,1]



//implementation
//As a graph
//nodes have:
//  a name
//  a eta product
//edges have:
//  a double survey
//  a bool for sign (inversion of variable)
//  a pi product
//Graph is undirected (for now)

struct SPEdge {
  double eta;
  bool isNegative;

  SPEdge() {}
  SPEdge(bool isNeg) :isNegative(isNeg) {
    eta = (double)rand() / (double)RAND_MAX;
  }
};

struct SPNode {
  bool isClause;
  int name;
  bool solved;
  bool value;
  
  double Bias;

  SPNode(int n, bool b) :isClause(b), name(n), solved(false), value(false) {}
};

typedef Galois::Graph::FirstGraph<SPNode, SPEdge, false> Graph;
typedef Galois::Graph::FirstGraph<SPNode, SPEdge, false>::GraphNode GNode;

static Graph graph;

static std::vector<GNode> literals;
static std::vector<GNode> clauses;

static Galois::GAccumulator<unsigned int> nontrivial;

static Galois::GReduceMax<double> maxBias;
static Galois::GReduceAverage<double> averageBias;

//interesting parameters:
static const double epsilon = 0.000001;

void initalize_random_formula(int M, int N, int K) {
  //M clauses
  //N variables
  //K vars per clause

  //build up clauses and literals
  clauses.resize(M);
  literals.resize(N);

  for (int m = 0; m < M; ++m) {
    GNode N = graph.createNode(SPNode(m, true));
    graph.addNode(N, Galois::NONE);
    clauses[m] = N;
  }
  for (int n = 0; n < N; ++n) {
    GNode N = graph.createNode(SPNode(n, false));
    graph.addNode(N, Galois::NONE);
    literals[n] = N;
  }

  for (int m = 0; m < M; ++m) {
    //Generate K unique values
    std::vector<int> touse;
    while (touse.size() != (unsigned)K) {
      //extra complex to generate uniform rand value
      int newK = (int)(((double)rand()/(double)RAND_MAX) * (double)N);
      if (std::find(touse.begin(), touse.end(), newK) 
	  == touse.end()) {
	touse.push_back(newK);
	graph.addEdge(clauses[m], literals[newK], SPEdge((bool)(rand() % 2)), Galois::NONE);
      }
    }
  }
}

void print_formula() {
  for (unsigned m = 0; m < clauses.size(); ++m) {
    if (m != 0)
      std::cout << " & ";
    std::cout << "c" << m << "( ";
    GNode N = clauses[m];
    for (Graph::neighbor_iterator ii = graph.neighbor_begin(N, Galois::NONE), ee = graph.neighbor_end( N, Galois::NONE); ii != ee; ++ii) {
      if (ii != graph.neighbor_begin(N, Galois::NONE))
	std::cout << " | ";
      SPEdge& E = graph.getEdgeData(N, ii, Galois::NONE);
      if (E.isNegative)
	std::cout << "-";
      
      SPNode& V = graph.getData(*ii, Galois::NONE);
      std::cout << "v" << V.name;
      if (V.solved)
	std::cout << "[" << (V.value ? 1 : 0) << "]";
      std::cout << "{" << E.eta << "," << V.Bias << "," << (V.value ? 1 : 0) << "}";
      std::cout << " ";
    }
    std::cout << " )";
  }
  std::cout << "\n";
}

void print_fixed() {
  for (unsigned n = 0; n < literals.size(); ++n) {
    GNode N = literals[n];
    SPNode& V = graph.getData(N, Galois::NONE);
    if (V.solved)
      std::cout << V.name << "[" << (V.value ? 1 : 0) << "] ";
  }
  std::cout << "\n";
}

int count_fixed() {
  int retval = 0;
  for (unsigned n = 0; n < literals.size(); ++n) {
    GNode N = literals[n];
    SPNode& V = graph.getData(N, Galois::NONE);
    if (V.solved)
      ++retval;
  }
  return retval;
}

struct update_eta {

  double eta_for_a_i(GNode a, GNode i) {
    double etaNew = 1.0;
    //for each j
    for (Graph::neighbor_iterator jii = graph.neighbor_begin(a), 
	   jee = graph.neighbor_end(a); jii != jee; ++jii) {
      GNode j = *jii;
      if (j != i) {
	bool ajNegative = graph.getEdgeData(a,jii).isNegative;
	double prodP = 1.0;
	double prodN = 1.0;
	double prod0 = 1.0;
	//for each b
	for (Graph::neighbor_iterator bii = graph.neighbor_begin(j), 
	       bee = graph.neighbor_end(j); 
	     bii != bee; ++bii) {
	  GNode b = *bii;
	  SPEdge& Ebj = graph.getEdgeData(j, bii);
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

  template<typename Context>
  void operator()(GNode a, Context& ctx) {
    //for each i
    for (Graph::neighbor_iterator iii = graph.neighbor_begin(a), 
	   iee = graph.neighbor_end(a); iii != iee; ++iii) {
      GNode i = *iii;
      double e = eta_for_a_i(a, i);
      double olde = graph.getEdgeData(a,iii).eta;
      graph.getEdgeData(a,iii).eta = e;
      //std::cout << olde << ',' << e << " ";
      if (fabs(olde - e) > epsilon) {
	for (Graph::neighbor_iterator bii = graph.neighbor_begin(i), 
	   bee = graph.neighbor_end(i); bii != bee; ++bii) {
	  GNode b = *bii;
	  if (a != b)
	    ctx.push(b);
	}
      }
    }
  }
};

//compute biases on each node
struct update_biases {
  template<typename Context>
  void operator()(GNode i, const Context& ctx) {
    SPNode& idata = i.getData(Galois::NONE);
    if (idata.solved) return;

    double pp1 = 1.0;
    double pp2 = 1.0;
    double pn1 = 1.0;
    double pn2 = 1.0;
    double p0 = 1.0;

    //for each function a
    for (Graph::neighbor_iterator aii = graph.neighbor_begin(i, Galois::NONE), aee = graph.neighbor_end(i, Galois::NONE); aii != aee; ++aii) {
      SPEdge& aie = graph.getEdgeData(i, *aii, Galois::NONE);

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
    idata.Bias = d;
    idata.value = (BiasP > BiasN);

    assert(d != NAN && -d != NAN);
    maxBias.update(d);
    averageBias.update(d);
  }
};

//return true if converged
void SP_algorithm() {
  //0) at t = 0, for every edge a->i, randomly initialize the message sigma a->i(t=0) in [0,1]
  //1) for t = 1 to tmax:
  //1.1) sweep the set of edges in a random order, and update sequentially the warnings on all the edges of the graph, generating the values sigma a->i (t) using SP_update
  //1.2) if (|sigma a->i(t) - sigma a->i (t-1) < E on all the edges, the iteration has converged and generated sigma* a->i = sigma a->i(t), goto 2
  //2) if t = tmax return un-converged.  if (t < tmax) then return the set of fixed point warnings sigma* a->i = sigma a->i (t)
  
  Galois::for_each(clauses.begin(), clauses.end(), update_eta(), "update_eta");
  maxBias.reset(0.0);
  averageBias.reset(0.0);
  nontrivial.reset(0);
  Galois::for_each(literals.begin(), literals.end(), update_biases(), "update_bias");
}

struct fix_variables {
  double limit;
  fix_variables(double d) :limit(d) {}
  template<typename Context>
  void operator()(GNode i, const Context& ctx) {
    SPNode& idata = i.getData();
    if (idata.solved) return;
    if (idata.Bias > limit) {
      idata.solved = true;
      //TODO: simplify graph
      //for each b
      for (Graph::neighbor_iterator bii = graph.neighbor_begin(i), 
	     bee = graph.neighbor_end(i); 
	   bii != bee; ++bii) {
	bii->getData().solved = true;
	bii->getData().value = true;
      }
      graph.removeNode(i);
    }
  }
};

void decimate() {
  std::cout << "NonTrivial " << nontrivial.get() << " MaxBias " << maxBias.get() << " Average Bias " << averageBias.get() << "\n";
  double d = ((maxBias.get() - averageBias.get()) * 0.25) + averageBias.get();
  Galois::for_each(literals.begin(), literals.end(), fix_variables(d), "fix_variables");
}

bool survey_inspired_decimation() {
  //0) Randomize initial conditions for the surveys
  //1) run SP
  //   if (SP does not converge, return SP UNCONVEREGED and stop
  //   if SP convereges, use fixed-point surveys n*a->i to
  //2) decimate
  //2.1) if non-trivial surveys (n != 0) are found, then:
  //   a) compute biases (W+,W-,W0) from PI+,PI-,PI0
  //   b) fix largest |W+ - W-| to x =  W+ > W-
  //   c) clean the graph
  //2.2) if all surveys are trivial(n = 0), output simplified subformula
  //4) if solved, output SAT, if no contradiction, continue at 1, if contridiction, stop
  do {
    SP_algorithm();
    if (nontrivial.get()) {
      std::cout << "DECIMATED\n";
      decimate();
    } else {
      std::cout << "SIMPLIFIED\n";
      return false;
    }
  } while (true); // while (true);
  return true;
}


int main(int argc, const char** argv) {
  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 4) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  int seed = atoi(args[0]);
  srand(seed);

  int M = atoi(args[1]);
  int N = atoi(args[2]);
  int K = atoi(args[3]);

  initalize_random_formula(M,N,K);
  //print_formula();
  //build_graph();
  //print_graph();

  Galois::StatTimer T;
  T.start();
  survey_inspired_decimation();
  T.stop();

  //print_formula();
  //print_fixed();

  std::cout << "Fixed " << count_fixed() << " variables\n";

  return 0;
}

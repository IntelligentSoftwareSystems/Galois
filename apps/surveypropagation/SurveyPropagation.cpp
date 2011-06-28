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
 * @author Martin Burtscher <burtscher@txstate.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Timer.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <cstdlib>

static const char* name = "Survey Propagation";
static const char* description = "Solves SAT problems using survey propagation\n";
static const char* url = 
  "http://iss.ices.utexas.edu/lonestar/surveypropagation.html";
static const char* help = "<seed> <num_clauses> <num_variables> <k>";


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

struct literal {
  int name;
  bool solved;
  bool value;
  std::vector<int> functions;
  literal() :name(0), solved(0), value(0) {}
};

struct clause {
  int name;
  //var num, negated
  std::vector<std::pair<int, bool> > variables;
  clause() :name(0) {}
};

std::vector<clause> clauses;
std::vector<literal> literals;

struct SPEdge {
  double PIu;
  double PIs;
  double PI0;
  double eta;
  bool isNegative;
  SPEdge() {}
  SPEdge(bool isNeg) :isNegative(isNeg) {
    eta = (double)rand() / (double)RAND_MAX;
  }
};

struct SPNode {
  literal* lit;
  clause* cla;
  double BiasP;
  double BiasN;
  double Bias0;
  double Bias;
  SPNode(literal* l) :lit(l), cla(0) {}
  SPNode(clause* c) :lit(0), cla(c) {}
};


typedef Galois::Graph::FirstGraph<SPNode, SPEdge, false> Graph;
typedef Galois::Graph::FirstGraph<SPNode, SPEdge, false>::GraphNode GNode;

static Graph graph;

static std::vector<GNode> gliterals;
static std::vector<GNode> gclauses;

static Galois::accumulator<unsigned int> unconverged;
static Galois::accumulator<unsigned int> nontrivial;

static Galois::reduce_max<double> maxBias;
static Galois::reduce_average<double> averageBias;

//interesting parameters:
static const double epsilon = 0.00001;
static const int tmax = 100;

void initalize_random_formula(int M, int N, int K) {
  //M clauses
  //N variables
  //K vars per clause

  //build up clauses and literals
  clauses.resize(M);
  literals.resize(N);

  for (int m = 0; m < M; ++m)
    clauses[m].name = m;
  for (int n = 0; n < N; ++n)
    literals[n].name = n;

  for (int m = 0; m < M; ++m) {
    //Generate K unique values
    std::vector<int> touse;
    while (touse.size() != (unsigned)K) {
      //extra complex to generate uniform rand value
      int newK = (int)(((double)rand()/(double)RAND_MAX) * (double)N);
      if (std::find(touse.begin(), touse.end(), newK) 
	  == touse.end()) {
	touse.push_back(newK);
	clauses[m].variables.push_back(std::make_pair(newK, (bool) (rand() % 2)));
	literals[newK].functions.push_back(m);
      }
    }
  }
}

void print_formula() {
  for (unsigned m = 0; m < clauses.size(); ++m) {
    if (m != 0)
      std::cout << " & ";
    std::cout << "c" << m << "( ";
    for (unsigned k = 0; k < clauses[m].variables.size(); ++k) {
      if (k != 0)
	std::cout << " | ";
      if (clauses[m].variables[k].second)
	std::cout << "-";
      int n = clauses[m].variables[k].first;
      std::cout << "v" << n;
      if (literals[n].solved)
	std::cout << "[" << (literals[n].value ? 1 : 0) << "]";
      std::cout << " ";
    }
    std::cout << " )";
  }
  std::cout << "\n";
}

//transform the clauses and literals into a graph
void build_graph() {

  gliterals.resize(literals.size());
  for (unsigned i = 0; i < literals.size(); ++i) {
    GNode N = graph.createNode(&literals[i]);
    graph.addNode(N, Galois::Graph::NONE);
    gliterals[i] = N;
  }

  gclauses.resize(clauses.size());
  for (unsigned i = 0; i < clauses.size(); ++i) {
    GNode N = graph.createNode(&clauses[i]);
    graph.addNode(N, Galois::Graph::NONE);
    gclauses[i] = N;
    for (unsigned j = 0; j < clauses[i].variables.size(); ++j)
      graph.addEdge(N, gliterals[clauses[i].variables[j].first],
          SPEdge(clauses[i].variables[j].second), Galois::Graph::NONE);
  }
}

void print_graph() {
  for (unsigned i = 0; i < clauses.size(); ++i) {
    GNode N = gclauses[i];
    for (Graph::neighbor_iterator
        ii = graph.neighbor_begin(N, Galois::Graph::NONE),
        ee = graph.neighbor_end(N, Galois::Graph::NONE); ii != ee; ++ii) {
      std::cout << "c"
	<< graph.getData(N, Galois::Graph::NONE).cla->name
	<< "->v"
	<< graph.getData(*ii, Galois::Graph::NONE).lit->name
	<< " eta: "
	<< graph.getEdgeData(N, *ii, Galois::Graph::NONE).eta
	<< " PI[u,s,0]: "
	<< graph.getEdgeData(N, *ii, Galois::Graph::NONE).PIu
	<< " "
	<< graph.getEdgeData(N, *ii, Galois::Graph::NONE).PIs
	<< " "
	<< graph.getEdgeData(N, *ii, Galois::Graph::NONE).PI0
	<< "\n";
    }
  }
  std::cout << "\n";
}

static bool isVu(bool JajNegative, bool JbjNegative) {
  if (JajNegative)
    return JbjNegative;
  else
    return !JbjNegative;
}

static bool isVs(bool JajNegative, bool JbjNegative) {
  if (JajNegative)
    return !JbjNegative;
  else
    return JbjNegative;
}

//Update all pi products on all edges of the variable j
struct update_pi {
  template<typename Context>
  void operator()(GNode j, const Context& ctx) {
    assert(j.getData(Galois::Graph::NONE).lit);
    
    //for each function a
    for (Graph::neighbor_iterator
        aii = graph.neighbor_begin(j, Galois::Graph::NONE),
        aee = graph.neighbor_end(j, Galois::Graph::NONE); aii != aee; ++aii) {
      SPEdge& jae = graph.getEdgeData(j, *aii, Galois::Graph::NONE);
      //Now we have j->a, compute each pi product
      double prodVuu = 1.0;
      double prodVus = 1.0;
      double prodVss = 1.0;
      double prodVsu = 1.0;
      double prodVa = 1.0;
      //for each b
      for (Graph::neighbor_iterator 
          bii = graph.neighbor_begin(j, Galois::Graph::NONE),
          bee = graph.neighbor_end(j, Galois::Graph::NONE); bii != bee; ++bii) {
	SPEdge& bje = graph.getEdgeData(j, *bii, Galois::Graph::NONE);
	double etabj = bje.eta;
	if (bii != aii) { //all of these terms ignore the source
	  if (isVu(jae.isNegative, bje.isNegative)) {
	    prodVuu *= (1.0 - etabj); //1st term
	    prodVsu *= (1.0 - etabj); //2nd term
	  } else {
	    prodVus *= (1.0 - etabj); //1st term
	    prodVss *= (1.0 - etabj); //2nd term
	  }
	  //3rd term
	  prodVa *= (1 - etabj);
	}
      }
      jae.PIu = (1.0 - prodVuu) * prodVus;
      jae.PIs = (1.0 - prodVss) * prodVsu;
      jae.PI0 = prodVa;
    }
  }
};

//update all eta products (surveys) on all edges of the function a
struct update_eta {
  template<typename Context>
  void operator()(GNode a, const Context& ctx) {
    assert(a.getData(Galois::Graph::NONE).cla);
    
    //for each i
    for (Graph::neighbor_iterator 
        iii = graph.neighbor_begin(a, Galois::Graph::NONE),
        iee = graph.neighbor_end(a, Galois::Graph::NONE); iii != iee; ++iii) {
      SPEdge& aie = graph.getEdgeData(a, *iii, Galois::Graph::NONE);
      double prod = 1.0;
      //for each j
      for (Graph::neighbor_iterator
          jii = graph.neighbor_begin(a, Galois::Graph::NONE),
          jee = graph.neighbor_end(a, Galois::Graph::NONE); jii != jee; ++jii) {
	if (jii != iii) { //ignore i
	  SPEdge& jae = graph.getEdgeData(a, *jii, Galois::Graph::NONE);
	  prod *= (jae.PIu / (jae.PIu + jae.PIs + jae.PI0));
	}
      }
      double delta = aie.eta < prod ? prod - aie.eta : aie.eta - prod;
      if (delta > epsilon)
	unconverged += 1;
      if (prod > epsilon)
	nontrivial += 1;
      aie.eta = prod;
    }
  }
};

//compute biases on each node
struct update_biases {
  template<typename Context>
  void operator()(GNode i, const Context& ctx) {
    SPNode& idata = i.getData(Galois::Graph::NONE);
    assert(idata.lit);
    
    double pp1 = 1.0;
    double pp2 = 1.0;
    double pn1 = 1.0;
    double pn2 = 1.0;
    double p0 = 1.0;

    //for each function a
    for (Graph::neighbor_iterator
        aii = graph.neighbor_begin(i, Galois::Graph::NONE),
        aee = graph.neighbor_end(i, Galois::Graph::NONE); aii != aee; ++aii) {
      SPEdge& aie = graph.getEdgeData(i, *aii, Galois::Graph::NONE);

      double etaai = aie.eta;
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
    
    idata.BiasP = pp / (pp + pn + p0);
    idata.BiasN = pn / (pp + pn + p0);
    idata.Bias0 = 1.0 - idata.BiasP - idata.BiasN;

    double d = idata.BiasP - idata.BiasN;
    if (d < 0.0)
      d = idata.BiasN - idata.BiasP;
    idata.Bias = d;

    maxBias.insert(d);
    averageBias.insert(d);
  }
};

//return true if converged
bool SP_algorithm() {
  //0) at t = 0, for every edge a->i, randomly initialize the
  //   message sigma a->i(t=0) in [0,1]
  //1) for t = 1 to tmax:
  //1.1) sweep the set of edges in a random order, and update sequentially
  //     the warnings on all the edges of the graph, generating the values
  //     sigma a->i (t) using SP_update
  //1.2) if (|sigma a->i(t) - sigma a->i (t-1) < E on all the edges, the
  //     iteration has converged and generated sigma* a->i = sigma a->i(t),
  //     goto 2
  //2) if t = tmax return un-converged.  if (t < tmax) then return the set
  //   of fixed point warnings sigma* a->i = sigma a->i (t)
  
  int x = tmax;
  do {
    unconverged.reset(0);
    nontrivial.reset(0);
    --x;

    // for (unsigned int i = 0; i < gliterals.size(); ++i)
    //   update_pi()(gliterals[i], 0);

    Galois::for_each(gliterals.begin(), gliterals.end(), update_pi());
    
    //print_graph();
    
    // for (unsigned int i = 0; i < gclauses.size(); ++i)
    //   update_eta()(gclauses[i], 0);

    Galois::for_each(gclauses.begin(), gclauses.end(), update_eta());
   
    //print_graph();

    std::cout << "U: " << unconverged.get() << " NT: " << nontrivial.get() << "\n";

  } while (x && unconverged.get());

  return unconverged.get() == 0;
}

struct fix_variables {
  double limit;
  fix_variables(double d) :limit(d) {}
  template<typename Context>
  void operator()(GNode i, const Context& ctx) {
    SPNode& idata = i.getData(Galois::Graph::NONE);
    if (idata.Bias > limit) {
      idata.lit->solved = true;
      idata.lit->value = (idata.BiasP > idata.BiasN);
      //TODO: simplify graph
    }
  }
};

void decimate() {
  maxBias.reset(0.0);
  averageBias.reset(0.0);
  Galois::for_each(gliterals.begin(), gliterals.end(), update_biases());
  std::cout << "\n" << maxBias.get() << " " << averageBias.get() << "\n";
  double d = ((maxBias.get() - averageBias.get()) * 0.25) + averageBias.get();
  Galois::for_each(gliterals.begin(), gliterals.end(), fix_variables(d));
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
  //4) if solved, output SAT, if no contradiction, continue at 1, if
  //   contridiction, stop
  do {
    if (SP_algorithm()) {
      if (nontrivial.get()) {
	decimate();
      } else {
	std::cout << "SIMPLIFIED\n";
	return false;
      }
    } else {
      std::cout << "UNCONVERGED\n";
      return false; // unconverged
    }
  } while (false); // while (true);
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
  build_graph();
  //print_graph();

  survey_inspired_decimation();

  //print_formula();

  return 0;
}

#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <cstdlib>

using namespace std;

static const char* name = "Survey Propagation";
static const char* description = "Solves SAT problems using survey propagation\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/surveypropagation.html";
static const char* help = "seed";


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
};

struct clause {
  int name;
  //var num, negated
  std::vector<std::pair<int, bool> > variables;
};

std::vector<clause> clauses;
std::vector<literal> literals;

struct SPEdge {
  double PIu;
  double PIs;
  double PI0;
  double eta;
  bool isNegative;
  SPEdge() {
    eta = (double)rand() / (double)RAND_MAX;
  }
};

struct SPNode {
  bool isVariableNode;
  bool isFunctionNode;
};


typedef Galois::Graph::FirstGraph<SPNode, SPEdge, false> Graph;
typedef Galois::Graph::FirstGraph<SPNode, SPEdge, false>::GraphNode GNode;

Graph graph;

void initalize_random_formula(int M, int N, int K) {
  //M clauses
  //N variables
  //K vars per clause

  //build up clauses and literals
  clauses.resize(M);
  literals.resize(N);

  for (int m = 0; m < M; ++m) {
    //Generate K unique values
    std::vector<int> touse;
    while (touse.size() != K) {
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
  for (int m = 0; m < clauses.size(); ++m) {
    if (m != 0)
      std::cout << " & ";
    std::cout << "( ";
    for (int k = 0; k < clauses[m].variables.size(); ++k) {
      if (k != 0)
	std::cout << " | ";
      if (clauses[m].variables[k].second)
	std::cout << "-";
      int n = clauses[m].variables[k].first;
      std::cout << n;
      if (literals[n].solved)
	std::cout << "[" << literals[n].value << "]";
      std::cout << " ";
    }
    std::cout << " )";
  }
  std::cout << "\n";
}


//Update all pi products on all edges of the variable j
struct update_pi {
  void operator()(GNode j) {
    assert(j.getData().isVariableNode);
    
    //for each function a
    for (Graph::neighbor_iterator aii = graph.neighbor_begin(j), aee = graph.neighbor_end(j); aii != aee; ++aii) {
      SPEdge& jae = graph.getEdgeData(j, *aii);
      //Now we have j->a, compute each pi product
      double prodVuu = 1.0;
      double prodVus = 1.0;
      double prodVss = 1.0;
      double prodVsu = 1.0;
      double prodVa = 1.0;
      //for each b
      for (Graph::neighbor_iterator bii = graph.neighbor_begin(j), bee = graph.neighbor_end(j); bii != bee; ++bii) {
	SPEdge& bje = graph.getEdgeData(j, *bii);
	double etabj = bje.eta;
	if (bii != aii) { //all of these terms ignore the source
	  if ((bje.isNegative && jae.isNegative) 
	      || (!bje.isNegative && !jae.isNegative)) {
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
  void operator()(GNode a) {
    assert(a.getData().isFunctionNode);

    //for each i
    for (Graph::neighbor_iterator iii = graph.neighbor_begin(a), iee = graph.neighbor_end(a); iii != iee; ++iii) {
      SPEdge& aie = graph.getEdgeData(a, *iii);
      double prod = 1.0;
      //for each j
      for (Graph::neighbor_iterator jii = graph.neighbor_begin(a), jee = graph.neighbor_end(a); jii != jee; ++jii) {
	if (jii != iii) { //ignore i
	  SPEdge& jae = graph.getEdgeData(a, *jii);
	  prod *= (jae.PIu / (jae.PIu + jae.PIs + jae.PI0));
	}
    }
      aie.eta = prod;
    }
  }
};

void SP_algorithm() {
  //0) at t = 0, for every edge a->i, randomly initialize the message sigma a->i(t=0) in [0,1]
  //1) for t = 1 to tmax:
  //1.1) sweep the set of edges in a random order, and update sequentially the warnings on all the edges of the graph, generating the values sigma a->i (t) using SP_update
  //1.2) if (|sigma a->i(t) - sigma a->i (t-1) < E on all the edges, the iteration has converged and generated sigma* a->i = sigma a->i(t), goto 2
  //2) if t = tmax return un-converged.  if (t < tmax) then return the set of fixed point warnings sigma* a->i = sigma a->i (t)
  
  

}

void survey_inspired_decimation() {
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
  print_formula();

  return 0;
}

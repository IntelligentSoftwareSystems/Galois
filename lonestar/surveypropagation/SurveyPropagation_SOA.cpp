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
 * Survey Propagation
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Gurbinder Gill <gill@cs.utexas.edu>
 */
#include "galois/Timer.h"
#include "galois/Atomic.h"
#include "galois/graphs/Graph.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "galois/runtime/Profile.h"

#include <cstdlib>
#include <iostream>

#include <math.h>

//using namespace std;
using namespace galois::worklists;

namespace cll = llvm::cl;

static const char* name = "Survey Propagation";
static const char* desc = "Solves SAT problems using survey propagation";
static const char* url = "survey_propagation";

static cll::opt<int> seed(cll::Positional, cll::desc("<seed>"), cll::Required);
static cll::opt<int> M(cll::Positional, cll::desc("<num clauses>"), cll::Required);
static cll::opt<int> N(cll::Positional, cll::desc("<num variables>"), cll::Required);
static cll::opt<int> K(cll::Positional, cll::desc("<variables per clause>"), cll::Required);

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

#if 0
struct SPNode {
  bool isClause;
  int name;
  bool solved;
  bool value;
  int t;
  
  double Bias;

  galois::GAtomic<bool> onWL;

  SPNode(int n, bool b) :isClause(b), name(n), solved(false), value(false), t(0), onWL(false) {}
};
#endif
struct SPNode {
  uint32_t id;
  SPNode(uint32_t _id) : id(_id) {}
  SPNode() {}
};

using Graph = galois::graphs::FirstGraph<SPNode, SPEdge, false>;
using GNode = galois::graphs::FirstGraph<SPNode, SPEdge, false>::GraphNode;

//interesting parameters:
static const double epsilon = 0.000001;
//static const int tmax = 100;
//static int tlimit = 0;

struct EIndexer: public std::unary_function<std::pair<GNode,int>,int> {
  int operator()(const std::pair<GNode,int>& v) {
    return v.second;
  }
};

struct ELess {
  bool operator()(const std::pair<GNode,int>& lhs,
		  const std::pair<GNode,int>& rhs) {
    if (lhs.second < rhs.second) return true;
    if (lhs.second > rhs.second) return false;
    return lhs < rhs;
  }
};

struct EGreater {
  bool operator()(const std::pair<GNode,int>& lhs,
		  const std::pair<GNode,int>& rhs) {
    if (lhs.second > rhs.second) return true;
    if (lhs.second < rhs.second) return false;
    return lhs > rhs;
  }
};

class SurveyPropagation {
  template<typename T>
  using Array = galois::LargeArray<T>;

  Graph graph;
  Array<GNode> literalsN;
  Array<GNode> clauses;

  // nodes' fields in SOA form
  Array<bool> isClause;
  Array<int> nameSPNode;
  Array<bool> solved;
  Array<bool> value;
//  Array<int> t;
  Array<double> Bias;
  Array<galois::GAtomic<bool>> onWL;

  galois::GAccumulator<unsigned int> nontrivial;
  galois::GReduceMax<double> maxBias;
  galois::GAccumulator<int> numBias;
  galois::GAccumulator<double> sumBias;

  void construct_random_formula(Graph& initGraph) {
    //build up clauses and literals
    clauses.allocateInterleaved(M);
    literalsN.allocateInterleaved(N);

    // construct graph topology using initGraph
    for (int m = 0; m < M; ++m) {
      GNode node = initGraph.createNode(SPNode(m));
      initGraph.addNode(node, galois::MethodFlag::UNPROTECTED);
      clauses[m] = node;
    }

    for (int n = 0; n < N; ++n) {
      GNode node = initGraph.createNode(SPNode(M+n));
      initGraph.addNode(node, galois::MethodFlag::UNPROTECTED);
      literalsN[n] = node;
    }

    for (int m = 0; m < M; ++m) {
      //Generate K unique values
      std::vector<int> touse;
      while (touse.size() != (unsigned)K) {
        //extra complex to generate uniform rand value
        int newK = (int)(((double)rand()/((double)RAND_MAX + 1)) * (double)(N));
        if (std::find(touse.begin(), touse.end(), newK) == touse.end()) {
          touse.push_back(newK);
          initGraph.getEdgeData(initGraph.addEdge(clauses[m], literalsN[newK], galois::MethodFlag::UNPROTECTED)) = SPEdge((bool)(rand() % 2));
        }
      }
    }
  }

  void save_to_FileGraph(galois::graphs::FileGraphWriter& p, Graph& initGraph) {
    p.setNumNodes(M+N);
    p.setNumEdges(M*K);
    p.setSizeofEdgeData(sizeof(SPEdge));

    p.phase1();
    for (int m = 0; m < M; ++m) {
      p.incrementDegree(m, K);
    }
    for (int n = 0; n < N; ++n) {
      p.incrementDegree(M+n, 0);
    }

    using EdgeData = galois::LargeArray<typename Graph::edge_data_type>;
    EdgeData edgeData;
    edgeData.create(M*K);

    p.phase2();
    for (int m = 0; m < M; ++m) {
      for (auto e: initGraph.edges(clauses[m], galois::MethodFlag::UNPROTECTED)) {
        auto dst = initGraph.getData(initGraph.getEdgeDst(e), galois::MethodFlag::UNPROTECTED).id;
        edgeData.set(p.addNeighbor(m, dst), initGraph.getEdgeData(e));
      }
    }

    using edge_value_type = typename EdgeData::value_type;
    edge_value_type* rawEdgeData = p.finish<edge_value_type>();
    if (EdgeData::has_value) {
      std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), rawEdgeData);
    }
  }

  void initialize_random_formula() {
    //M clauses
    //N variables
    //K vars per clause

    Graph initGraph;
    construct_random_formula(initGraph);

    // save initGraph to a FileGraph
    galois::graphs::FileGraphWriter p;
    save_to_FileGraph(p, initGraph);

    // reload initGraph from p to graph in a NUMA-aware way
    galois::graphs::readGraphDispatch(graph, Graph::read_tag(), p);

    //Resize vectors for SOA
    isClause.allocateInterleaved(M+N);
    nameSPNode.allocateInterleaved(M+N);
    solved.allocateInterleaved(M+N);
    value.allocateInterleaved(M+N);
//    t.allocateInterleaved(M+N);
    Bias.allocateInterleaved(M+N);
    onWL.allocateInterleaved(M+N);

    auto node = graph.begin();
    for (int m = 0; m < M; ++m) {
      graph.getData(*node, galois::MethodFlag::UNPROTECTED).id = m;
      nameSPNode[m] = m;
      isClause[m] = true;
      solved[m] = false;
      value[m] = false;
//      t[m] = 0;
      onWL[m] = true;

      clauses[m] = *node;
      node++;
    }

    for (int n = 0; n < N; ++n) {
      graph.getData(*node, galois::MethodFlag::UNPROTECTED).id = M+n;
      nameSPNode[M+n] = n;
      isClause[M+n] = false;
      solved[M+n] = false;
      value[M+n] = false;
//      t[M+n] = 0;
      onWL[M+n] = false;

      literalsN[n] = *node;
      node++;
    }

    //XXX: Shuffling doesn't help
    //std::random_shuffle(literalsN.begin(), literalsN.end());
    //std::random_shuffle(clauses.begin(), clauses.end());
  }

  void print_formula() {
    for (unsigned m = 0; m < clauses.size(); ++m) {
      if (m != 0)
        std::cout << " & ";
      std::cout << "c" << m << "( ";
      GNode N = clauses[m];
      for (auto ii : graph.edges(N, galois::MethodFlag::UNPROTECTED)) {
        if (ii != graph.edge_begin(N, galois::MethodFlag::UNPROTECTED))
          std::cout << " | ";
        SPEdge& E = graph.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
        if (E.isNegative)
          std::cout << "-";

        SPNode& V = graph.getData(graph.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED);
        std::cout << "v" << nameSPNode[V.id];
        if (solved[V.id])
          std::cout << "[" << (value[V.id] ? 1 : 0) << "]";
        std::cout << "{" << E.eta << "," << Bias[V.id] << "," << (value[V.id] ? 1 : 0) << "}";
        std::cout << " ";
      }
      std::cout << " )\n";
    }
    std::cout << std::endl;
  }

  void print_literal() {
    for (unsigned n = 0; n < literalsN.size(); ++n) {
      std::cout << "v" << n << "( ";
      GNode N = literalsN[n];
      for (auto ii : graph.edges(N, galois::MethodFlag::UNPROTECTED)) {
        SPEdge& E = graph.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
        if (E.isNegative)
          std::cout << "-";

        SPNode& V = graph.getData(graph.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED);
        std::cout << "c" << nameSPNode[V.id];
        if (solved[V.id])
          std::cout << "[" << (value[V.id] ? 1 : 0) << "]";
        std::cout << "{" << E.eta << "," << Bias[V.id] << "," << (value[V.id] ? 1 : 0) << "}";
        std::cout << " ";
      }
      std::cout << " )\n";
    }
    std::cout << std::endl;
  }

  void print_fixed() {
    for (unsigned n = 0; n < literalsN.size(); ++n) {
      GNode N = literalsN[n];
      SPNode& V = graph.getData(N, galois::MethodFlag::UNPROTECTED);
      if (solved[V.id])
        std::cout << nameSPNode[V.id] << "[" << (value[V.id] ? 1 : 0) << "]\n";
    }
    std::cout << "\n";
  }

  int count_fixed() {
    int retval = 0;
    for (unsigned n = 0; n < literalsN.size(); ++n) {
      GNode N = literalsN[n];
      SPNode& V = graph.getData(N, galois::MethodFlag::UNPROTECTED);
      if (solved[V.id])
        ++retval;
    }
    return retval;
  }

  double eta_for_a_i(GNode a, GNode i) {
    double etaNew = 1.0;
    //for each j
    for (auto jii : graph.edges(a, galois::MethodFlag::UNPROTECTED)) {
      GNode j = graph.getEdgeDst(jii);
      if (j != i) {
        bool ajNegative = graph.getEdgeData(jii, galois::MethodFlag::UNPROTECTED).isNegative;
        double prodP = 1.0;
        double prodN = 1.0;
        double prod0 = 1.0;
        //for each b
        for (auto bii : graph.edges(j, galois::MethodFlag::UNPROTECTED)) {
          GNode b = graph.getEdgeDst(bii);
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

  //return true if converged
  void SP_algorithm() {
    //0) at t = 0, for every edge a->i, randomly initialize the message sigma a->i(t=0) in [0,1]
    //1) for t = 1 to tmax:
    //1.1) sweep the set of edges in a random order, and update sequentially the warnings on all the edges of the graph, generating the values sigma a->i (t) using SP_update
    //1.2) if (|sigma a->i(t) - sigma a->i (t-1) < E on all the edges, the iteration has converged and generated sigma* a->i = sigma a->i(t), goto 2
    //2) if t = tmax return un-converged.  if (t < tmax) then return the set of fixed point warnings sigma* a->i = sigma a->i (t)

    //  tlimit += tmax;

    using WL = galois::worklists::dChunkedFIFO<128>;
    //using WL = galois::worklists::ParaMeter<>;
    //using WL = galois::worklists::AltChunkedFIFO<1024>;
    //using OBIM = galois::worklists::OrderedByIntegerMetric<
                   //decltype(EIndexer()),
                   //galois::worklists::dChunkedFIFO<512>
                   //>;

    galois::reportPageAlloc("MeminfoPre: SP_algorithm for_each");
    galois::for_each( galois::iterate(clauses),
        [&, self=this] (const GNode& p, auto& ctx) {
          GNode a = p;
          //std::cerr << graph.getData(a).t << " ";
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

          auto& a_data = graph.getData(a);
          //++a_data.t;
          onWL[a_data.id] = false;
//          ++t[a_data.id];

          //++graph.getData(a).t;

          //for each i
          for (auto iii : graph.edges(a, galois::MethodFlag::UNPROTECTED)) {
            GNode i = graph.getEdgeDst(iii);
            double e = self->eta_for_a_i(a, i);
            double olde = graph.getEdgeData(iii, galois::MethodFlag::UNPROTECTED).eta;
            graph.getEdgeData(iii).eta = e;
            //std::cout << olde << ',' << e << " " << "\n";
            if (fabs(olde - e) > epsilon) {
              for (auto bii : graph.edges(i, galois::MethodFlag::UNPROTECTED)) {
                GNode b = graph.getEdgeDst(bii);
                auto b_data = graph.getData(b, galois::MethodFlag::UNPROTECTED);
                if (a != b && onWL[b_data.id].cas(false, true)) { // && graph.getData(b, galois::MethodFlag::UNPROTECTED).t < tlimit)
                //if (a != b) { // && graph.getData(b, galois::MethodFlag::UNPROTECTED).t < tlimit)
                  ctx.push(b);
                }
              }
            }
          }
        },
        galois::loopname("update_eta"),
        galois::wl<WL>());
    galois::reportPageAlloc("MeminfoPost: SP_algorithm for_each");

    maxBias.reset();
    numBias.reset();
    sumBias.reset();
    nontrivial.reset();

    galois::reportPageAlloc("MeminfoPre: SP_algorithm do_all");
    // update_biases
    galois::do_all(galois::iterate(literalsN),
        [&] (GNode i) {
          SPNode& idata = graph.getData(i, galois::MethodFlag::UNPROTECTED);
          if (solved[idata.id]) return;

          double pp1 = 1.0;
          double pp2 = 1.0;
          double pn1 = 1.0;
          double pn2 = 1.0;
          double p0 = 1.0;

          //for each function a
          for (auto aii : graph.edges(i, galois::MethodFlag::UNPROTECTED)) {
            GNode i = graph.getEdgeDst(aii);
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
          Bias[idata.id] = d;
          value[idata.id] = (BiasP > BiasN);

          assert(!std::isnan(d) && !std::isnan(-d));
          maxBias.update(d);
          numBias += 1;
          sumBias += d;
        },
        galois::loopname("update_biases"));
      galois::reportPageAlloc("MeminfoPost: SP_algorithm do_all");
  }

  void decimate() {
    double m = maxBias.reduce();
    double n = nontrivial.reduce();
    int num = numBias.reduce();
    double average = num > 0 ? sumBias.reduce() / num : 0.0;
    std::cout << "NonTrivial " << n << " MaxBias " << m << " Average Bias " << average << "\n";
    double d = ((m - average) * 0.25) + average;

    const double limit = d;

    galois::reportPageAlloc("MeminfoPre: decimate");
    // fix_variables
    galois::do_all(galois::iterate(literalsN),
        [&] (GNode i) {
          SPNode& idata = graph.getData(i);
          if (solved[idata.id]){return;};
          if (Bias[idata.id] > limit) {
            solved[idata.id] = true;
            //TODO: simplify graph
            //for each b
            for (auto bii : graph.edges(i)) {
              //graph.getData(graph.getEdgeDst(bii)).solved = true;
              //graph.getData(graph.getEdgeDst(bii)).value = true;
              solved[graph.getData(graph.getEdgeDst(bii)).id] = true;
              value[graph.getData(graph.getEdgeDst(bii)).id] = true;
            }
            graph.removeNode(i);
          }
        },
        galois::loopname("fix_variables"));
    galois::reportPageAlloc("MeminfoPost: decimate");
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
    galois::preAlloc(numThreads + 120*(graph.size()) / galois::runtime::pagePoolSize());
    galois::reportPageAlloc("MeminfoPre: whole");
    do {
      SP_algorithm();
      if (nontrivial.reduce()) {
        std::cout << "DECIMATED\n";
        decimate();
      } else {
        std::cout << "SIMPLIFIED\n";
        galois::reportPageAlloc("MeminfoPost: whole");
        return false;
      }
    } while (true); // while (true);
    galois::reportPageAlloc("MeminfoPost: whole");
    return true;
  }

public:
  void run() {
    srand(seed);
    initialize_random_formula();
//  print_formula(graph, clauses);
//  print_literal(graph, literalsN);

    std::cout << "Starting..." << std::endl;

    galois::StatTimer T;
    T.start();
    galois::runtime::profileVtune(
        [&] () {
          survey_inspired_decimation();
        },
        "Main Loop"
    );
    T.stop();

    //print_formula(graph, clauses);
    //print_fixed(graph, literalsN);

    std::cout << "Fixed " << count_fixed() << " variables" << std::endl;
  }
}; // end struct SurveyPropagation

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  SurveyPropagation sp;
  sp.run();

  return 0;
}

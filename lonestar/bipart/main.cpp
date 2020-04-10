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

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <array>
#include <unordered_set>

#include "Metis.h"
#include "galois/graphs/Util.h"
#include "galois/Timer.h"
//#include "GraphReader.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/graphs/FileGraph.h"
#include "galois/LargeArray.h"

namespace cll = llvm::cl;

static const char* name = "HYPAR";
static const char* desc =
    "Partitions a hypergraph into K parts and minimizing the graph cut";
static const char* url = "HyPar";

static cll::opt<scheduleMode> schedulingMode(
    cll::desc("Choose a inital scheduling mode:"),
    cll::values(clEnumVal(PLD, "PLD"), clEnumVal(PP, "PP"), clEnumVal(WD, "WD"),
                clEnumVal(RI, "RI"), clEnumVal(MRI, "MRI"),clEnumVal(MDEG, "MDEG"),clEnumVal(DEG, "DEG"),clEnumVal(MWD, "MWD"),clEnumVal(HIS, "HIS"),clEnumVal(RAND, "random"),clEnumValEnd),
    cll::init(PLD));

static cll::opt<bool>
    mtxInput("mtxinput",
             cll::desc("Use text mtx files instead of binary galois gr files"),
             cll::init(false));
static cll::opt<bool> weighted("weighted", cll::desc("weighted"),
                               cll::init(false));
static cll::opt<bool>
    verbose("verbose",
            cll::desc("verbose output (debugging mode, takes extra time)"),
            cll::init(false));
static cll::opt<std::string> outfile("output",
                                     cll::desc("output partition file name"));
static cll::opt<std::string>
    orderedfile("ordered", cll::desc("output ordered graph file name"));
static cll::opt<std::string>
    permutationfile("permutation", cll::desc("output permutation file name"));
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned> csize(cll::Positional,
                                   cll::desc("<size of coarsest graph>"),
                                   cll::Required);

static cll::opt<unsigned> refiter(cll::Positional,
                                   cll::desc("<number of iterations in ref>"),
                                   cll::Required);
static cll::opt<double> imbalance(
    "balance",
    cll::desc("Fraction deviated from mean partition size (default 0.01)"),
    cll::init(0.01));

// const double COARSEN_FRACTION = 0.9;

int cutsize(GGraph& g) { 
  int cutsize = 0;
  std::vector<int> cells;
  for (auto n : g.getNets()) { 
    bool cut_status = false;
    for (auto e : g.edges(n)) {
      auto cell1 = g.getEdgeDst(e);
    for (auto c : g.edges(n)) {
        auto cell2 = g.getEdgeDst(c);
        if(g.getData(cell1).getPart() != g.getData(cell2).getPart() && cell1 != cell2) {
          cutsize++;
          cut_status = true;
          break;
        }
      }
      if (cut_status == true)
        break;
    }
  }
  return cutsize;
}
/**
 * Partitioning 
 */
void Partition(MetisGraph* metisGraph, unsigned coarsenTo, unsigned refineTo) {
  galois::StatTimer TM;
  TM.start();

  galois::StatTimer T("CoarsenSEP");
  T.start();
  MetisGraph* mcg = coarsen(metisGraph, coarsenTo, schedulingMode);
  T.stop();

  galois::StatTimer T2("PartitionSEP");
  T2.start();
  partition(mcg);
  T2.stop();


  galois::StatTimer T3("Refine");
  T3.start();
  refine(mcg, refineTo);
  T3.stop();
  std::cout << "clustering:," << T2.get() << '\n';
  std::cout << "coarsen:," << T.get() << "\n";
  std::cout << "Refinement:," << T3.get() << "\n";
  return;
}

int computingCut(GGraph& g) {

  GNodeBag bag;
  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
          for (auto cell : g.edges(n)) {
            auto c = g.getEdgeDst(cell);
            int part = g.getData(c).getPart();
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int partc = g.getData(cc).getPart();
              if (partc != part) {
                bag.push(n);
                return;
              }

            }
          }
        },
        galois::loopname("cutsize"));
  return std::distance(bag.begin(), bag.end());
}

int computingBalance(GGraph& g) {
  int zero = 0, one = 0;
  for (auto c : g.cellList()) {
    int part = g.getData(c).getPart();
    if (part == 0) zero++;
    else one++;
  }
  return std::abs(zero - one);
}
// printGraphBeg(*graph)

typedef galois::graphs::FileGraph FG;
typedef FG::GraphNode FN;
template <typename GNode, typename Weights>
struct order_by_degree {
  GGraph& graph;
  Weights& weights;
  order_by_degree(GGraph& g, Weights& w) : graph(g), weights(w) {}
  bool operator()(const GNode& a, const GNode& b) {
    uint64_t wa = weights[a];
    uint64_t wb = weights[b];
    int pa      = graph.getData(a, galois::MethodFlag::UNPROTECTED).getPart();
    int pb      = graph.getData(b, galois::MethodFlag::UNPROTECTED).getPart();
    if (pa != pb) {
      return pa < pb;
    }
    return wa < wb;
  }
};

typedef galois::substrate::PerThreadStorage<std::map<GNode, uint64_t>>
    PerThreadDegInfo;


std::map<uint64_t, uint64_t> cellToNet(std::map< uint64_t, std::vector< uint64_t> > netToCell) {
  std::map<uint64_t, uint64_t> celltonet;
  for (auto n : netToCell) {
    for (auto c : n.second) {
      celltonet[c]++;
    }
  }
  return celltonet;
}

int hash(unsigned val) {
  unsigned long int seed = val * 1103515245 + 12345;
  return((unsigned)(seed/65536) % 32768);
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

 // srand(-1);
  MetisGraph metisGraph;
  GGraph& graph = *metisGraph.getGraph();
  std::ifstream f(filename.c_str());
  std::string line;
  std::getline(f, line);
  std::stringstream ss(line);
  int i1, i2;
  ss >> i1 >> i2;
  const int hedges = i1, nodes = i2;
  printf("hedges: %d\n", hedges);
  printf("nodes: %d\n\n", nodes);

  // read rest of input and initialize hedges (build hgraph)
  std::unordered_set<int>* const hedge = new std::unordered_set<int> [hedges];
  int cnt = 0, entries = 0;
  while (std::getline(f, line)) {
    if (cnt >= hedges) {printf("ERROR: too many lines in input file\n"); exit(-1);}
    std::stringstream ss(line);
    int val;
    while (ss >> val) {
      if ((val < 1) || (val > nodes)) {printf("ERROR: node value %d out of bounds\n", val); exit(-1);}
      hedge[cnt].insert(val - 1);
      entries++;
    }
    cnt++;
  }
  f.close();
  std::map<int, GNode> hnets;
  // create nodes
  for(int i = 0; i < nodes; i++){
    GNode node;
    MetisNode n1;
    n1.netnum = INT_MAX;
    n1.netrand = INT_MAX;
    n1.netval = INT_MAX;
    n1.nodeid = i + 1;
    //n1.setWeight(hash(i+1));
    node = graph.createNode(n1); 
    graph.addNode(node);
    graph.addCell(node);
    hnets[i] = node;
  }
  // create hyperedges
  for (int i = 0; i < hedges; i++){
    GNode a;
    MetisNode n1;
    //int id = rand()%(hedges-1);
    n1.netnum = i+1;
    //unsigned rand = hash(i+1);
    n1.netrand = INT_MAX;
    //n1.setWeight(rand);
    a = graph.createNode(n1);
    graph.addNode(a);
    graph.addHyperedge(a);
    for (auto v : hedge[i]) {
      GNode b = hnets[v];
      graph.addEdge(a,b);
    }
  }

  graphStat(graph);
  std::cout<<"\n";
  galois::preAlloc(galois::runtime::numPagePoolAllocTotal() * 5);
  galois::reportPageAlloc("MeminfoPre");
  Partition(&metisGraph, csize, refiter);
  std::cout<<"Total Edge Cut: "<<computingCut(graph)<<"\n";
  galois::runtime::reportStat_Single("HyPar", "Edge Cut", computingCut(graph));
  galois::runtime::reportStat_Single("HyParzo", "zero-one", computingBalance(graph));
  // galois::reportPageAlloc("MeminfoPost");

  return 0;
}


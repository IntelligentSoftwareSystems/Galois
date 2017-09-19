/** GMetis -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
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

#include "Metis.h"
#include "Galois/Graphs/Util.h"
#include "Galois/Timer.h"
//#include "GraphReader.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/LargeArray.h"

namespace cll = llvm::cl;

static const char* name = "GMetis";
static const char* desc = "Partitions a graph into K parts and minimizing the graph cut";
static const char* url = "gMetis";


static cll::opt<InitialPartMode> partMode(cll::desc("Choose a inital part mode:"),
    cll::values(
      clEnumVal(GGP, "GGP."),
      clEnumVal(GGGP, "GGGP, default."),
      clEnumVal(MGGGP, "MGGGP."),
      clEnumValEnd), cll::init(GGGP));
static cll::opt<refinementMode> refineMode(cll::desc("Choose a refinement mode:"),
    cll::values(
      clEnumVal(BKL, "BKL"),
      clEnumVal(BKL2, "BKL2, default."),
      clEnumVal(ROBO, "ROBO"),
      clEnumVal(GRACLUS, "GRACLUS"),
      clEnumValEnd), cll::init(BKL2));

static cll::opt<bool> mtxInput("mtxinput", cll::desc("Use text mtx files instead binary based ones"), cll::init(false));
static cll::opt<bool> weighted("weighted", cll::desc("weighted"), cll::init(false));
static cll::opt<bool> verbose("verbose", cll::desc("verbose output (debugging mode, takes extra time)"), cll::init(false));
static cll::opt<std::string> outfile("output", cll::desc("output partition file name"));
static cll::opt<std::string> orderedfile("ordered", cll::desc("output ordered graph file name"));
static cll::opt<std::string> permutationfile("permutation", cll::desc("output permutation file name"));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> numPartitions(cll::Positional, cll::desc("<Number of partitions>"), cll::Required);
static cll::opt<double> imbalance("balance", cll::desc("Fraction deviated from mean partition size"), cll::init(0.01));
const double COARSEN_FRACTION = 0.9;

/**
 * KMetis Algorithm
 */
void Partition(MetisGraph* metisGraph, unsigned nparts) {
  Galois::StatTimer TM;
  TM.start();
  unsigned meanWeight = ( (double)metisGraph->getTotalWeight()) / (double)nparts;
  //unsigned coarsenTo = std::max(metisGraph->getNumNodes() / (40 * intlog2(nparts)), 20 * (nparts));
  unsigned coarsenTo = 20 * nparts;

  if (verbose) std::cout << "Starting coarsening: \n";
  Galois::StatTimer T("Coarsen");
  T.start();
  MetisGraph* mcg = coarsen(metisGraph, coarsenTo, verbose);
  T.stop();
  if (verbose) std::cout << "Time coarsen: " << T.get() << "\n";

  Galois::StatTimer T2("Partition");
  T2.start();
  std::vector<partInfo> parts;
  parts = partition(mcg, nparts, partMode);
  T2.stop();

  if (verbose) std::cout << "Init edge cut : " << computeCut(*mcg->getGraph()) << "\n\n";

  std::vector<partInfo> initParts = parts;
  std::cout << "Time clustering:  "<<T2.get()<<'\n';

  if (verbose)
    switch (refineMode) {
      case BKL2:    std::cout<< "Sarting refinnement with BKL2\n";    break;
      case BKL:     std::cout<< "Sarting refinnement with BKL\n";     break;
      case ROBO:    std::cout<< "Sarting refinnement with ROBO\n";    break;
      case GRACLUS: std::cout<< "Sarting refinnement with GRACLUS\n"; break;
      default: abort();
    }

  Galois::StatTimer T3("Refine");
  T3.start();
  refine(mcg, parts, 
      meanWeight - (unsigned)(meanWeight * imbalance), 
      meanWeight + (unsigned)(meanWeight * imbalance), 
      refineMode, verbose);
  T3.stop();
  if (verbose) std::cout << "Time refinement: " << T3.get() << "\n";

  TM.stop();

  std::cout << "Initial dist\n";
  printPartStats(initParts);
  std::cout << "\n";

  std::cout << "Refined dist\n";
  printPartStats(parts);
  std::cout << "\n";

  std::cout << "Time:  " << TM.get() << '\n';
  return;
}


//printGraphBeg(*graph)

struct parallelInitMorphGraph {
  GGraph &graph;
  parallelInitMorphGraph(GGraph &g):graph(g) {  }
  void operator()(GNode node) const {
    for (auto jj : graph.edges(node)) {
      graph.getEdgeData(jj)=1;
      // weight+=1;
    }
  }
};
typedef Galois::Graph::FileGraph FG;
typedef FG::GraphNode FN;
template<typename GNode, typename Weights>
struct order_by_degree {
  GGraph &graph;
  Weights& weights;
  order_by_degree(GGraph &g, Weights &w):graph(g),weights(w) {

  }
  bool operator()(const GNode& a, const GNode& b) {
    uint64_t wa = weights[a];
    uint64_t wb = weights[b];
    int pa = graph.getData(a,Galois::MethodFlag::UNPROTECTED).getPart();
    int pb = graph.getData(b,Galois::MethodFlag::UNPROTECTED).getPart();
    if (pa != pb) { 
      return pa < pb;
    }
    return wa < wb;
  }
};
typedef Galois::Substrate::PerThreadStorage<std::map<GNode,uint64_t> > PerThreadDegInfo;
struct OrderGraph {
  GGraph &graph;
  PerThreadDegInfo &threadDegInfo;
  OrderGraph(GGraph &g,PerThreadDegInfo &ti):graph(g),threadDegInfo(ti) {

  }
  template<typename Context>
  void operator()(int part, Context &lwl) {
    auto flag = Galois::MethodFlag::UNPROTECTED;
    typedef std::vector<std::pair<unsigned,GNode>, Galois::PerIterAllocTy::rebind<std::pair<unsigned,GNode> >::other> GD;
    //copy and translate all edges
    GD orderedNodes(GD::allocator_type(lwl.getPerIterAlloc()));
    for (auto n : graph) {
      auto &nd = graph.getData(n,flag); 
      if (static_cast<int>(nd.getPart()) == part) {
        int edges = std::distance(graph.edge_begin(n,flag), graph.edge_end(n,flag));
        orderedNodes.push_back(std::make_pair(edges,n));
      }
    } 
    std::sort(orderedNodes.begin(),orderedNodes.end());
    int index = 0;
    std::map<GNode, uint64_t> &threadMap(*threadDegInfo.getLocal());
    for (auto p : orderedNodes) {
      GNode n = p.second;
      threadMap[n] += index;
      for (auto eb : graph.edges(n, flag)) {
        GNode neigh = graph.getEdgeDst(eb);
        auto &nd = graph.getData(neigh,flag);
        if (static_cast<int>(nd.getPart()) == part) { 
          threadMap[neigh] += index;
        }
      }
      index++;
    }
  }
};
int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  srand(-1);
  MetisGraph metisGraph;
  GGraph* graph = metisGraph.getGraph();

  Galois::Graph::readGraph(*graph, filename);

  Galois::do_all_local(*graph, parallelInitMorphGraph(*graph));

  graphStat(graph);
  std::cout << "\n";

  //printGraphBeg(*graph);

  Galois::reportPageAlloc("MeminfoPre");
  Galois::preAlloc(Galois::Runtime::numPagePoolAllocTotal() * 5);
  Partition(&metisGraph, numPartitions);
  Galois::reportPageAlloc("MeminfoPost");

  std::cout << "Total edge cut: " << computeCut(*graph) << "\n";

  if (outfile != "") {   
    MetisGraph* coarseGraph = &metisGraph;
    while (coarseGraph->getCoarserGraph())
      coarseGraph = coarseGraph->getCoarserGraph();
    std::ofstream outFile(outfile.c_str());
    for (auto it = graph->begin(), ie = graph->end(); it!=ie; it++)
    {
      unsigned gPart = graph->getData(*it).getPart();
      outFile<< gPart<< '\n';
    }
  }
  
  if (orderedfile != "" || permutationfile != "") { 
    Galois::Graph::FileGraph g;
    g.fromFile(filename);
    typedef Galois::LargeArray<GNode> Permutation;
    Permutation perm; 
    perm.create(g.size());
    std::copy(graph->begin(),graph->end(), perm.begin());
    PerThreadDegInfo threadDegInfo; 
    OrderGraph og(*graph,threadDegInfo);
    std::vector<int> parts(numPartitions);
    for (unsigned int i=0;i<parts.size();i++){ 
      parts[i] = i;
    } 
    Galois::for_each(parts.begin(), parts.end(), og, Galois::loopname("Order Graph"));
    std::map<GNode,uint64_t> globalMap;
    for (unsigned int i = 0; i < threadDegInfo.size(); i++) { 
      std::map<GNode,uint64_t> &localMap(*threadDegInfo.getRemote(i));
      for (auto mb = localMap.begin(), me = localMap.end(); mb != me; mb++) { 
        globalMap[mb->first] = mb->second;
      }
    }
    order_by_degree<GNode,std::map<GNode,uint64_t> > fn(*graph,globalMap);
    std::map<GNode,int> nodeIdMap; 
    int id = 0;
    for (auto nb = graph->begin(), ne = graph->end(); nb != ne; nb++) { 
      nodeIdMap[*nb] = id; 
      id++;
    }
    //compute inverse
    std::stable_sort(perm.begin(), perm.end(), fn);
    Galois::LargeArray<uint64_t> perm2;
    perm2.create(g.size());
    //compute permutation
    id = 0; 
    for (auto pb = perm.begin(), pe = perm.end(); pb != pe; pb++) { 
      int prevId = nodeIdMap[*pb];
      perm2[prevId] = id; 
      //std::cout<<prevId <<" "<<id<<std::endl;
      id++;
    }
    Galois::Graph::FileGraph out;
    Galois::Graph::permute<int>(g, perm2, out);
    if (orderedfile != "")
      out.toFile(orderedfile);
    if (permutationfile != "") {
      std::ofstream file(permutationfile.c_str());
      Galois::LargeArray<uint64_t> transpose;
      transpose.create(g.size());
      uint64_t id = 0;
      for (auto ii = perm2.begin(), ei = perm2.end(); ii != ei; ++ii)
        transpose[*ii] = id++;
      for (auto ii = transpose.begin(), ei = transpose.end(); ii != ei; ++ii)
        file << *ii << "\n";
    }
  }
  return 0;
}

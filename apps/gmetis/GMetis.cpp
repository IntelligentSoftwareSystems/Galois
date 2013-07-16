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
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>

#include "Metis.h"
#include "Galois/Statistic.h"
//#include "GraphReader.h"
#include "Lonestar/BoilerPlate.h"

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
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> numPartitions(cll::Positional, cll::desc("<Number of partitions>"), cll::Required);

const double COARSEN_FRACTION = 0.9;

/**
 * KMetis Algorithm
 */
void Partition(MetisGraph* metisGraph, unsigned nparts) {
  Galois::StatTimer TM;
  TM.start();
  unsigned meanWeight = ( (double)metisGraph->getTotalWeight()) / (double)nparts;
  //unsigned coarsenTo = std::max(metisGraph->getNumNodes() / (40 * intlog2(nparts)), 20 * (nparts));
  unsigned coarsenTo = 40 * nparts;

  std::cout << "\n  Starting coarsening: \n";
  Galois::StatTimer T("Coarsen");
  T.start();
  MetisGraph* mcg = coarsen(metisGraph, coarsenTo);
  T.stop();
  std::cout << "Time coarsen:  "<<T.get()<< " Size of final graph: " <<  mcg->getNumNodes()<<'\n';
   
  Galois::StatTimer T2("Partition");
  T2.start();
  std::vector<partInfo> parts;
  switch (partMode) {
    case GGP:parts = partition(mcg, nparts, GGP); break;
    case GGGP: parts = partition(mcg, nparts, GGGP); break;
    case MGGGP: parts = BisectAll(mcg, nparts, meanWeight); break;
    default: abort();
  }
  T2.stop();

  std::vector<partInfo> initParts = parts;
  std::cout << "Time clustering:  "<<T2.get()<<'\n';
  Galois::StatTimer T3("Refine");
  T3.start();
  refinementMode refM =refineMode;
  refine(mcg, parts, meanWeight, refM);
  T3.stop();

  TM.stop();
  std::cout << "Initial dist\n";
  printPartStats(initParts);
  std::cout << "Refined dist\n";
  printPartStats(parts);
  std::cout << "Time refinement:  "<<T3.get()<<"\n\n";

  std::cout << "\nTime:  " << TM.get() << '\n';
  return;
}

struct parallelInitMorphGraph {
  GGraph &graph;
  parallelInitMorphGraph(GGraph &g):graph(g) {  }
  void operator()(GNode node) {
    for(GGraph::edge_iterator jj = graph.edge_begin(node),kk=graph.edge_end(node);jj!=kk;jj++) {
      graph.getEdgeData(jj)=1;
      // weight+=1;
    }
  }
};

int edgeCount2(GGraph& g) {
  int count=0;
  for (auto nn = g.begin(), en = g.end(); nn != en; ++nn) {
    for (auto ii = g.edge_begin(*nn), ee = g.edge_end(*nn); ii != ee; ++ii)
        count += g.getEdgeData(ii);
  }
  return count/2;
}

int computeCut(GGraph& g) {
  int cuts=0;
  for (auto nn = g.begin(), en = g.end(); nn != en; ++nn) {
    unsigned gPart = g.getData(*nn).getPart();
    for (auto ii = g.edge_begin(*nn), ee = g.edge_end(*nn); ii != ee; ++ii) {
      auto& m = g.getData(g.getEdgeDst(ii));
      if (m.getPart() != gPart) 
        cuts += g.getEdgeData(ii);
    }
  }
  return cuts/2;
}




/*void printGraphBeg(GGraph& g) {
  int i=0;
  for (auto nn = g.begin(), en = g.end() ; nn != en; ++nn)
    g.getData(*nn).num =i++;
  i=0;
  for (auto nn = g.begin(), en = g.end() ; nn != en && i<10000; ++nn) {
    for (auto ii = g.edge_begin(*nn), ee = g.edge_end(*nn); ii != ee; ++ii) {
      auto& m = g.getData(g.getEdgeDst(ii));
      std::cout<<g.getData(g.getEdgeDst(ii)).num<<' ';
      i++;
    }
   std::cout<<'\n';
  }
  std::cout<<'\n';
}*/




int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  srand(-1);
  MetisGraph metisGraph;
  GGraph* graph = metisGraph.getGraph();

  graph->structureFromFile(filename);

  if (0)  {
    std::ofstream dot("dump.dot");
    graph->dump(dot);
  }

  Galois::do_all_local(*graph, parallelInitMorphGraph(*graph));

  unsigned numEdges = 0;
  std::map<unsigned, unsigned> hist;
  for (auto ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
    unsigned val = std::distance(graph->edge_begin(*ii), graph->edge_end(*ii));
    numEdges += val;
    ++hist[val];
  }

  std::cout << "Nodes " << std::distance(graph->begin(), graph->end()) << "| Edges " << numEdges << ' ' << edgeCount2(*graph)<<std::endl;
 // for (auto pp = hist.begin(), ep = hist.end(); pp != ep; ++pp)
  //std::cout << pp->first << " : " << pp->second << "\n";

//printGraphBeg(*graph);

  Galois::reportPageAlloc("MeminfoPre");
  Galois::preAlloc(Galois::Runtime::MM::numPageAllocTotal() * 5);

  Partition(&metisGraph, numPartitions);

  std::cout << "Total edge cut : " << computeCut(*graph) << "\n\n";
  Galois::reportPageAlloc("MeminfoPost");

  MetisGraph* coarseGraph = &metisGraph;
  while (coarseGraph->getCoarserGraph())
    coarseGraph = coarseGraph->getCoarserGraph();
  std::ostringstream outfileName;
  outfileName << filename << ".part"<< numPartitions;
  std::ofstream outFile(outfileName.str().c_str());

  /*for (auto nn = graph->begin(), en = graph->end(); nn != en; ++nn) {
    unsigned gPart = graph->getData(*nn).getPart();
    outFile<< gPart<< '\n';
  }*/

  for (auto it = graph->begin(), ie = graph->end(); it!=ie; it++)
  {
    unsigned gPart = graph->getData(*it).getPart();
    outFile<< gPart<< '\n';
  }
  std::cout<< "Part saved in : "<<  outfileName<< '\n';
  
  //printCuts("Initial", coarseGraph, numPartitions);
  //printCuts("Final", &metisGraph, numPartitions);
  
  return 0;
}


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
#include "Galois/Graph/Util.h"
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
static cll::opt<bool> verbose("verbose", cll::desc("verbose output (debugging mode, takes extra time)"), cll::init(false));
static cll::opt<std::string> outfile("output", cll::desc("output file name"));

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
  std::cout << "Refined dist\n";
  printPartStats(parts);

  std::cout << "\nTime:  " << TM.get() << '\n';
  return;
}


//printGraphBeg(*graph)

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
  Galois::preAlloc(Galois::Runtime::MM::numPageAllocTotal() * 5);
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
  return 0;
}


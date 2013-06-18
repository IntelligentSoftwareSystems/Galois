/** GMetis -*- C++ -*-
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
 */

#include <vector>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <fstream>

#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "Metis.h"
#include "Metrics.h"

#include "Galois/Graph/LCGraph.h"
#include "Galois/Statistic.h"
#include "GraphReader.h"
#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* name = "GMetis";
static const char* desc = "Partitions a graph into K parts and minimizing the graph cut";
static const char* url = "gMetis";

static cll::opt<bool> mtxInput("mtxinput", cll::desc("Use text mtx files instead binary based ones"), cll::init(false));
static cll::opt<bool> weighted("weighted", cll::desc("weighted"), cll::init(false));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> numPartitions(cll::Positional, cll::desc("<Number of partitions>"), cll::Required);

const double COARSEN_FRACTION = 0.9;

bool verifyCoarsening(MetisGraph *metisGraph) {

  if(metisGraph == NULL)
    return true;
  cout<<endl<<"##### Verifying coarsening #####"<<endl;
  unsigned matchedCount=0;
  unsigned unmatchedCount=0;
  GGraph *graph = metisGraph->getGraph();

  for(GGraph::iterator ii=graph->begin(),ee=graph->end();ii!=ee;ii++) {
    GNode node = *ii;
    MetisNode &nodeData = graph->getData(node);
    GNode matchNode;
    if(!nodeData.isMatched())
      return false;
    matchNode = nodeData.getMatched();

    if(matchNode == node) {
      unmatchedCount++;
    }
    else{
      matchedCount++;
      MetisNode &matchNodeData = graph->getData(matchNode);
      GNode mmatch;
      if(!matchNodeData.isMatched())
        return false;
      mmatch = matchNodeData.getMatched();

      if(node!=mmatch){
        cout<<"Node's matched node is not matched to this node";
        return false;
      }
    }
    unsigned edges=0;
    for(GGraph::edge_iterator ii=graph->edge_begin(node),ee=graph->edge_end(node);ii!=ee;ii++) {
      edges++;
    }
    if(edges!=nodeData.getNumEdges()) {
      cout<<"Number of edges dont match";
      return false;
    }
  }
  bool ret = verifyCoarsening(metisGraph->getFinerGraph());
  cout<<matchedCount<<" "<<unmatchedCount<<endl;
  if(matchedCount+unmatchedCount != metisGraph->getNumNodes())
    return false;
  if(ret == false)
    return false;
  return true;

}

bool verifyRecursiveBisection(MetisGraph* metisGraph,int nparts) {

  GGraph *graph = metisGraph->getGraph();
  int partNodes[nparts];
  memset(partNodes,0,sizeof(partNodes));
  for(GGraph::iterator ii = graph->begin(),ee=graph->end();ii!=ee;ii++) {
    GNode node = *ii;
    MetisNode &nodeData = graph->getData(node);
    if(!(nodeData.getPart()<nparts))
      return false;
    partNodes[nodeData.getPart()]++;
    unsigned edges=0;
    for(GGraph::edge_iterator ii=graph->edge_begin(node),ee=graph->edge_end(node);ii!=ee;ii++) {
      edges++;
    }
    if(nodeData.getNumEdges()!=edges) {
      return false;
    }
  }
  unsigned sum=0;
  for(int i=0;i<nparts;i++) {
    if(partNodes[i]<=0)
      return false;
    sum+=partNodes[i];
  }


  if(sum != metisGraph->getNumNodes())
    return false;
  return true;
}


void printPartStats(std::vector<partInfo>& parts) {
  for (unsigned x = 0; x < parts.size(); ++x)
    std::cout << parts[x] << "\n";
}

/**
 * KMetis Algorithm
 */
void Partition(MetisGraph* metisGraph, unsigned nparts) {
  unsigned coarsenTo = std::max(metisGraph->getNumNodes() / (40 * intlog2(nparts)), 20 * (nparts));
  int maxVertexWeight = (int) (1.5 * ((metisGraph->getNumNodes()) / (double) coarsenTo));
  Galois::StatTimer T("Coarsening");
  Galois::Timer t;
  T.start();
  t.start();
  MetisGraph* mcg = coarsen(metisGraph, coarsenTo);
  t.stop();
  T.stop();
  cout<<"coarsening time: " << t.get() << " ms"<<endl;

  if(0) {
    if(verifyCoarsening(mcg->getFinerGraph())) {
      cout<<"#### Coarsening is correct ####"<<endl;
    } else {
      cout<<"!!!! Coarsening is wrong !!!!"<<endl;
    }
  }

  Galois::StatTimer T2("Partition");
  Galois::Timer t2;
  T2.start();
  t2.start();
  std::vector<partInfo> parts = partition(mcg, nparts);
  t2.stop();
  T2.stop();
  cout<<"initial part time: " << t2.get() << " ms"<<endl;

  printPartStats(parts);

  if(0) {
    cout<<endl<<"#### Verifying initial partition ####"<<endl;
    if(!verifyRecursiveBisection(mcg,nparts)) {
      cout<<endl<<"!!!! Initial partition is wrong !!!!"<<endl;
    }else {
      cout<<endl<<"#### Initial partition is right ####"<<endl;
    }
  }

  std::cout << "\n\n";
  Galois::StatTimer T3("Refine");
  Galois::Timer t3;
  T3.start();
  t3.start();
  coarsen(mcg, parts);
  t3.stop();
  T3.stop();
  cout<<"refinement time: " << t3.get() << " ms"<<endl;

  printPartStats(parts);

  return;
}


void verify(MetisGraph* metisGraph) {
  // if (!metisGraph->verify()) {
  //   cout<<"KMetis failed."<<endl;
  // }else{
  //   cout<<"KMetis okay"<<endl;
  // }
}

struct parallelInitMorphGraph {
  GGraph &graph;
  parallelInitMorphGraph(GGraph &g):graph(g) {

  }
  void operator()(unsigned int tid, unsigned int num) {
    int id = tid;
    for(GGraph::iterator ii = graph.local_begin(),ee=graph.local_end();ii!=ee;ii++) {
      GNode node = *ii;
      MetisNode &nodeData = graph.getData(node);
      //nodeData.setNodeId(id);
      nodeData.init();
      nodeData.setWeight(1);
      int count = std::distance(graph.edge_begin(node),graph.edge_end(node));
      nodeData.setNumEdges(count);
      int weight=0;
      for(GGraph::edge_iterator jj = graph.edge_begin(node),kk=graph.edge_end(node);jj!=kk;jj++) {
        graph.getEdgeData(jj)=1;
        weight+=1;
      }
      nodeData.setEdgeWeight(nodeData.getEdgeWeight() + weight);
      id+=num;
    }
  }
};

void printCuts(const char* str, MetisGraph* g) {
  std::vector<unsigned> ec = edgeCut(*g->getGraph(), numPartitions);
  std::cout << str << " Edge Cuts:\n";
  for (unsigned x = 0; x < ec.size(); ++x)
    std::cout << (x == 0 ? "" : " " ) << ec[x];
  std::cout << "\n";
  std::cout << str << " Average Edge Cut: " << (std::accumulate(ec.begin(), ec.end(), 0) / ec.size()) << "\n";
  std::cout << str << " Minimum Edge Cut: " << *std::min_element(ec.begin(), ec.end()) << "\n";
}


int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  srand(-1);
  MetisGraph metisGraph;
  GGraph* graph = metisGraph.getGraph();
  //bool directed = true;

  graph->structureFromFile(filename);

  if (0)  {
    std::ofstream dot("dump.dot");
    graph->dump(dot);
  }

  Galois::on_each(parallelInitMorphGraph(*graph));

  // metisGraph.setNumNodes(graph.size());
  // metisGraph.setNumEdges(graph.sizeEdges());
  unsigned numEdges = 0;
  std::map<unsigned, unsigned> hist;
  for (auto ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
    unsigned val = std::distance(graph->edge_begin(*ii), graph->edge_end(*ii));
    numEdges += val;
    ++hist[val];
  }

  cout<<"Nodes "<<std::distance(graph->begin(), graph->end())<<"| Edges " << numEdges << endl;
  for (auto pp = hist.begin(), ep = hist.end(); pp != ep; ++pp)
    std::cout << pp->first << " : " << pp->second << "\n";

  Galois::reportPageAlloc("MeminfoPre");
  Galois::preAlloc(Galois::Runtime::MM::numPageAllocTotal() * 12);

  Galois::Timer t;
  t.start();
  Partition(&metisGraph, numPartitions);
  t.stop();
  cout<<"Total Time "<<t.get()<<" ms "<<endl;
  Galois::reportPageAlloc("MeminfoPost");
  verify(&metisGraph);

  MetisGraph* coarseGraph = &metisGraph;
  while (coarseGraph->getCoarserGraph())
    coarseGraph = coarseGraph->getCoarserGraph();
  printCuts("Initial", coarseGraph);
  printCuts("Final", &metisGraph);

  return 0;
}

int getRandom(int num){
  //      int randNum = rand()%num;
  //      return (rand()>>3)%(num);
  //      return randNum;
  return ((int)(drand48()*((double)(num))));
}

// int gNodeToInt(GNode node){
//      return graph->getData(node).getNodeId();
// }

int intlog2(int a){
  int i;
  for (i=1; a > 1; i++, a = a>>1);
  return i-1;
}

/** Betweenness centrality application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Dimitrios Prountzos <dprountz@cs.utexas.edu>
 */

#define SHOULD_PRODUCE_CERTIFICATE 0

#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/UserContext.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Runtime/WorkList.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/filter_iterator.hpp>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>

static const char* name = "Betweenness Centrality";
static const char* desc =
  "Computes the betweenness centrality of all nodes in a graph\n";
static const char* url = "betweenness_centrality";

static llvm::cl::opt<std::string> filename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
static llvm::cl::opt<int> iterLimit("limit", llvm::cl::desc("Limit number of iterations to value (0 is all nodes)"), llvm::cl::init(0));

typedef Galois::Graph::LC_FileGraph<void, void> Graph;
typedef Graph::GraphNode GNode;

Graph* G;
int NumNodes;
std::vector<int> sucSize;

struct TempState  {
  std::vector<GNode> SQG;
  std::vector<double> sigmaG;
  std::vector<int> distG;

  std::vector<std::vector<GNode> > succsGlobal;

  std::vector<double> CB;

  TempState() 
    :SQG(NumNodes), sigmaG(NumNodes), distG(NumNodes), CB(NumNodes) {
    succsGlobal.resize(NumNodes);
    for (int i = 0; i < NumNodes; ++i)
      succsGlobal[i].reserve(sucSize[i]);
  }

  void reset() {
    SQG.resize(0);
    SQG.resize(NumNodes);
    sigmaG.resize(0);
    sigmaG.resize(NumNodes);
    distG.resize(0);
    distG.resize(NumNodes);
    for(int i = 0; i < succsGlobal.size(); ++i)
      succsGlobal[i].resize(0);
  }
};

GaloisRuntime::PerThreadStorage<TempState*> state;

void computeSucSize() {
  sucSize.resize(NumNodes);
  for (Graph::iterator ii = G->begin(), ee = G->end(); ii != ee; ++ii)
    sucSize[*ii] = std::distance(G->neighbor_begin(*ii, Galois::NONE),
				 G->neighbor_end(*ii, Galois::NONE));
}

struct process {
  void operator()(GNode& _req, Galois::UserContext<GNode>& lwl) {
    TempState* tmp = *state.getLocal();
    if (!tmp)
      tmp = *state.getLocal() = new TempState();
    tmp->reset();
    std::vector<GNode>& SQ = tmp->SQG;
    std::vector<double>& sigma = tmp->sigmaG;
    std::vector<int>& d = tmp->distG;
    std::vector<double>& delta = tmp->CB;
    std::vector<std::vector<GNode> >& suc = tmp->succsGlobal;

    int QPush = 0;
    int QAt = 0;
    
    int req = _req;
    
    sigma[req] = 1;
    d[req] = 1;
    SQ[QPush++] = _req;
    
    while (QAt != QPush) {
      GNode _v = SQ[QAt++];
      int v = _v;
      for (Graph::neighbor_iterator
          ii = G->neighbor_begin(_v, Galois::NONE),
          ee = G->neighbor_end(_v, Galois::NONE); ii != ee; ++ii) {
	GNode _w = *ii;
	int w = _w;
	if (!d[w]) {
	  SQ[QPush++] = _w;
	  d[w] = d[v] + 1;
	}
	if (d[w] == d[v] + 1) {
	  sigma[w] = sigma[w] + sigma[v];
          suc[v].push_back(w);
	}
      }
    }

    --QAt;
    while (QAt > 1) {
      int w = SQ[--QAt];

      double sigma_w = sigma[w];
      double delta_w = delta[w];
      for(std::vector<GNode>::iterator it = suc[w].begin(), end = suc[w].end();
	  it != end; ++it) {
	//std::cerr << "Processing node " << w << std::endl;
	GNode v = *it;
	delta_w += (sigma_w/sigma[v])*(1.0 + delta[v]);
      }
      delta[w] = delta_w;
    }
  }
};

void reduce(std::vector<double>& bcv) {
  bcv.resize(0);
  bcv.resize(NumNodes);
  for (unsigned int i = 0; i < state.size(); ++i)
    if (*state.getRemote(i))
      std::transform(bcv.begin(), bcv.end(), (*state.getRemote(i))->CB.begin(), bcv.begin(), std::plus<double>());
}

// Verification for reference torus graph inputs. 
// All nodes should have the same betweenness value.
void verify() {
    double sampleBC = 0.0;
    bool firstTime = true;
    std::vector<double> bcv;
    reduce(bcv);
    for (int i=0; i<NumNodes; ++i) {
      double bc = bcv[i];
      if (firstTime) {
        sampleBC = bc;
        std::cerr << "BC: " << sampleBC << std::endl;
        firstTime = false;
      } else {
        if (!((bc - sampleBC) <= 0.0001)) {
          std::cerr << "If torus graph, verification failed " << (bc - sampleBC) << "\n";
	  assert ((bc - sampleBC) <= 0.0001);
	  return;
	}
      }
    }
    std::cerr << "Verification ok!" << std::endl;
}

void printBCcertificate() {
  std::stringstream foutname;
  foutname << "outer_certificate_" << numThreads;
  std::ofstream outf(foutname.str().c_str());
  std::cerr << "Writting certificate..." << std::endl;
  std::vector<double> bcv;
  reduce(bcv);

  for (int i=0; i<NumNodes; ++i) {
    double bc = bcv[i];
    outf << i << ": " << setiosflags(std::ios::fixed) << std::setprecision(9) << bc << std::endl;
  }
  outf.close();
}

struct HasOut: public std::unary_function<GNode,bool> {
  Graph* graph;
  HasOut(Graph* g): graph(g) { }
  bool operator()(const GNode& n) const {
    return graph->neighbor_begin(n) != graph->neighbor_end(n);
  }
};

int main(int argc, char** argv) {
  Galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);

  Graph g;
  G = &g;
  G->structureFromFile(filename.c_str());
  NumNodes = G->size();
  computeSucSize();

  int iterations = NumNodes;
  if (iterLimit)
    iterations = iterLimit;

  boost::filter_iterator<HasOut,Graph::iterator>
    begin = boost::make_filter_iterator(HasOut(G), g.begin(), g.end()),
    end = boost::make_filter_iterator(HasOut(G), g.end(), g.end());

  iterations = std::min((int) std::distance(begin, end), iterations);

  std::cout 
    << "NumNodes: " << NumNodes 
    << " Iterations: " << iterations << "\n";
  
  end = begin;
  std::advance(end, iterations);
  std::vector<GNode> tmp;
  std::copy(begin, end, std::back_inserter(tmp));

  typedef GaloisRuntime::WorkList::dChunkedLIFO<2> WL;
  Galois::StatTimer T;
  T.start();
  Galois::for_each<WL>(tmp.begin(), tmp.end(), process());
  T.stop();

  if (!skipVerify) {
    verify();
  } else { // print bc value for first 10 nodes
    std::vector<double> bcv(NumNodes);
    for (int i = 0; i < state.size(); ++i)
      if (*state.getRemote(i))
	std::transform(bcv.begin(), bcv.end(), (*state.getRemote(i))->CB.begin(), bcv.begin(), std::plus<double>());
    for (int i=0; i<10; ++i)
      std::cout << i << ": " << setiosflags(std::ios::fixed) << std::setprecision(6) << bcv[i] << "\n";
#if SHOULD_PRODUCE_CERTIFICATE
    printBCcertificate();
#endif
  }
  std::cerr << "Application done...\n";

  Galois::StatTimer tt("cleanup");
  tt.start();
  //cleanupData();
  tt.stop();

  return 0;
}

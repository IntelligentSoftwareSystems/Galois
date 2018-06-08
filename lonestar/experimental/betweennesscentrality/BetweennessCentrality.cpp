/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#define SHOULD_PRODUCE_CERTIFICATE 0

#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/UserContext.h"
#include "galois/graphs/LCGraph.h"
#include "galois/worklists/WorkList.h"

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
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

#if defined(MAP_ANONYMOUS)
static const int _MAP_ANON = MAP_ANONYMOUS;
#elif defined(MAP_ANON)
static const int _MAP_ANON = MAP_ANON;
#else
// fail
#endif
#ifdef MAP_POPULATE
static const int _MAP_POP  = MAP_POPULATE;
#else
static const int _MAP_POP  = 0;
#endif

static const char* name = "Betweenness Centrality";
static const char* desc = "Computes the betweenness centrality of all nodes in a graph";
static const char* url = "betweenness_centrality";

static llvm::cl::opt<std::string> filename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
static llvm::cl::opt<int> iterLimit("limit", llvm::cl::desc("Limit number of iterations to value (0 is all nodes)"), llvm::cl::init(0));
static llvm::cl::opt<bool> forceVerify("forceVerify", llvm::cl::desc("Abort if not verified, only makes sense for torus graphs"));

typedef galois::graphs::LC_CSR_Graph<void, void> Graph;
typedef Graph::GraphNode GNode;

Graph* G;
int NumNodes;
std::vector<int> sucSize;

#define PAGE_SIZE (4*1024)
#define PAGE_ROUND_UP(x) ( (((uintptr_t)(x)) + PAGE_SIZE-1)  & (~(PAGE_SIZE-1)) )

struct TempState  {
  //  std::vector<GNode> SQG;
  GNode* SQG;
  //  std::vector<double> sigmaG;
  double* sigmaG;
  //  std::vector<int> distG;
  int* distG;

  std::vector<GNode>* succsGlobal;

  //  std::vector<double> CB;
  double* CB;

  TempState() {
    size_t len = PAGE_ROUND_UP(sizeof(GNode) * NumNodes);
    SQG = (GNode*)mmap(0, len, PROT_READ | PROT_WRITE, _MAP_POP | MAP_PRIVATE | _MAP_ANON, -1, 0);
    len = PAGE_ROUND_UP(sizeof(double) * NumNodes);
    sigmaG = (double*)mmap(0, len, PROT_READ | PROT_WRITE, _MAP_POP | MAP_PRIVATE | _MAP_ANON, -1, 0);
    len = PAGE_ROUND_UP(sizeof(int) * NumNodes);
    distG = (int*)mmap(0, len, PROT_READ | PROT_WRITE, _MAP_POP | MAP_PRIVATE | _MAP_ANON, -1, 0);
    len = PAGE_ROUND_UP(sizeof(std::vector<GNode>) * NumNodes);
    succsGlobal = (std::vector<GNode>*)mmap(0, len, PROT_READ | PROT_WRITE, _MAP_POP | MAP_PRIVATE | _MAP_ANON, -1, 0);
    len = PAGE_ROUND_UP(sizeof(double) * NumNodes);
    CB = (double*)mmap(0, len, PROT_READ | PROT_WRITE, _MAP_POP | MAP_PRIVATE | _MAP_ANON, -1, 0);

    //:SQG(NumNodes), sigmaG(NumNodes), distG(NumNodes), CB(NumNodes) {
    //succsGlobal.resize(NumNodes);
    for (int i = 0; i < NumNodes; ++i) {
      new (&succsGlobal[i]) std::vector<GNode>();
      succsGlobal[i].reserve(sucSize[i]);
    }
  }

  void reset() {
    // SQG.resize(0);
    // SQG.resize(NumNodes);
    // sigmaG.resize(0);
    // sigmaG.resize(NumNodes);
    // distG.resize(0);
    // distG.resize(NumNodes);
    for(int i = 0; i < NumNodes; ++i) {
      succsGlobal[i].resize(0);
      distG[i] = 0;
    }
  }
};

galois::substrate::PerThreadStorage<TempState*> state;

void computeSucSize() {
  sucSize.resize(NumNodes);
  for (Graph::iterator ii = G->begin(), ee = G->end(); ii != ee; ++ii)
    sucSize[*ii] = std::distance(G->edge_begin(*ii, galois::MethodFlag::UNPROTECTED),
				 G->edge_end(*ii, galois::MethodFlag::UNPROTECTED));
}

struct popstate {
  void operator()(int , int) {
    *state.getLocal() = *state.getLocal() = new TempState();
  }
};


struct process {
  void operator()(GNode& _req, galois::UserContext<GNode>& lwl) {
    TempState* tmp = *state.getLocal();
    tmp->reset();
    GNode* SQ = tmp->SQG;
    double* sigma = tmp->sigmaG;
    int* d = tmp->distG;
    double* delta = tmp->CB;
    std::vector<GNode>* suc = tmp->succsGlobal;

    int QPush = 0;
    int QAt = 0;
    
    int req = _req;
    
    sigma[req] = 1;
    d[req] = 1;
    SQ[QPush++] = _req;
    
    while (QAt != QPush) {
      GNode _v = SQ[QAt++];
      int v = _v;
      for (Graph::edge_iterator
          ii = G->edge_begin(_v, galois::MethodFlag::UNPROTECTED),
          ee = G->edge_end(_v, galois::MethodFlag::UNPROTECTED); ii != ee; ++ii) {
	GNode _w = G->getEdgeDst(ii);
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
      std::transform(bcv.begin(), bcv.end(), (*state.getRemote(i))->CB, bcv.begin(), std::plus<double>());
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
          if (forceVerify)
            abort();
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
    outf << i << ": " << std::setiosflags(std::ios::fixed) << std::setprecision(9) << bc << std::endl;
  }
  outf.close();
}

struct HasOut: public std::unary_function<GNode,bool> {
  Graph* graph;
  HasOut(Graph* g): graph(g) { }
  bool operator()(const GNode& n) const {
    return graph->edge_begin(n) != graph->edge_end(n);
  }
};

int main(int argc, char** argv) {
  galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);

  Graph g;
  G = &g;
  galois::graphs::readGraph(*G, filename); 
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

  galois::on_each(popstate());

  typedef galois::worklists::PerSocketChunkLIFO<8> WL;
  //typedef galois::worklists::PerThreadChunkLIFO<32> CA;
  galois::StatTimer T;
  T.start();
  galois::for_each(tmp.begin(), tmp.end(), process(), galois::wl<WL>());
  T.stop();

  if (forceVerify || !skipVerify) {
    verify();
  } else { // print bc value for first 10 nodes
    std::vector<double> bcv(NumNodes);
    for (unsigned i = 0; i < state.size(); ++i)
      if (*state.getRemote(i))
	std::transform(bcv.begin(), bcv.end(), (*state.getRemote(i))->CB, bcv.begin(), std::plus<double>());
    for (int i=0; i<10; ++i)
      std::cout << i << ": " << std::setiosflags(std::ios::fixed) << std::setprecision(6) << bcv[i] << "\n";
#if SHOULD_PRODUCE_CERTIFICATE
    printBCcertificate();
#endif
  }
  std::cerr << "Application done...\n";

  galois::StatTimer tt("cleanup");
  tt.start();
  //cleanupData();
  tt.stop();

  return 0;
}

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

#include "Galois/Accumulator.h"
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

#define DEBUG 0 
#define USE_SUCCS 1
#define SHARE_SINGLE_BC 0 
#define SHOULD_PRODUCE_CERTIFICATE 0

static const char* name = "Betweenness Centrality";
static const char* desc =
  "Computes the betweenness centrality of all nodes in a graph\n";
static const char* url = "betweenness_centrality";

static llvm::cl::opt<std::string> filename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
static llvm::cl::opt<int> iterLimit("limit", llvm::cl::desc("Limit number of iterations to value (0 is all nodes)"), llvm::cl::init(0));
static llvm::cl::opt<bool> forceVerify("forceVerify", llvm::cl::desc("Abort if not verified, only makes sense for torus graphs"));

typedef Galois::Graph::LC_FileGraph<void, void> Graph;
typedef Graph::GraphNode GNode;

Graph* G;
int NumNodes;

#if SHARE_SINGLE_BC
struct BCrecord {
  GaloisRuntime::SimpleLock<unsigned char, true> lock;
  double bc; 
  BCrecord(): lock(), bc(0.0) {}
};
std::vector<cache_line_storage<BCrecord> > *CB;
#else
struct merge {
  void operator()(std::vector<double>& lhs, std::vector<double>& rhs) {
    if (lhs.size() < rhs.size())
      lhs.resize(rhs.size());
    for (unsigned int i = 0; i < rhs.size(); i++)
      lhs[i] += rhs[i];
  }
};

Galois::GVectorElementAccumulator<std::vector<double> >* CB;
#endif

GaloisRuntime::PerThreadStorage<std::vector<GNode>*> SQG;
GaloisRuntime::PerThreadStorage<std::vector<double> *> sigmaG;
GaloisRuntime::PerThreadStorage<std::vector<double> *> deltaG;
GaloisRuntime::PerThreadStorage<std::vector<int>*> distG;

template<typename T>
struct PerIt {  
  typedef typename Galois::PerIterAllocTy::rebind<T>::other Ty;
};

GaloisRuntime::PerThreadStorage< std::vector<std::vector<GNode> > > succsGlobal;

std::vector<GNode> & getSuccs(GNode n) {
  return (*succsGlobal.getLocal())[n];
}

void initGraphData() {
  // Pre-compute successors sizes in tmp
  std::vector< std::vector<GNode> > tmp(NumNodes);
  for (Graph::iterator ii = G->begin(), ee = G->end();
      ii != ee; ++ii) {
    int nnbrs = std::distance(G->neighbor_begin(*ii, Galois::NONE),
        G->neighbor_end(*ii, Galois::NONE));
    //std::cerr << "Node : " << *ii << " has " << nnbrs << " neighbors " << std::endl;
    tmp[*ii].reserve(nnbrs); 
  }

  // Init all structures
  std::cerr << "Pre-allocating graph metadata for " << numThreads << " threads." << std::endl;
  for (int i=0; i<numThreads; ++i) {
    *succsGlobal.getRemote(i) = tmp;
    *SQG.getRemote(i) = new std::vector<GNode>(NumNodes); 
    *sigmaG.getRemote(i) = new std::vector<double>(NumNodes);
    *deltaG.getRemote(i) = new std::vector<double>(NumNodes); 
    *distG.getRemote(i) = new std::vector<int>(NumNodes); 
  }
}

void resetData() {
  std::vector<GNode> *sq = *SQG.getLocal();
  std::fill(sq->begin(), sq->end(), 0);

  std::vector<double> *sigma = *sigmaG.getLocal();
  std::fill(sigma->begin(), sigma->end(), 0);

  std::vector<double> *delta = *deltaG.getLocal();
  std::fill(delta->begin(), delta->end(), 0);

  std::vector<int> *dist = *distG.getLocal();
  std::fill(dist->begin(), dist->end(), 0);

  std::vector< std::vector<GNode> > & svec = *succsGlobal.getLocal();
  std::vector< std::vector<GNode> >::iterator it = svec.begin();
  std::vector< std::vector<GNode> >::iterator end = svec.end();
  while (it != end) {
    it->resize(0);
    ++it;
  }
}

void cleanupData() {
  for (int i=0; i<numThreads; ++i) {
    delete *SQG.getRemote(i);
    delete *sigmaG.getRemote(i);
    delete *deltaG.getRemote(i);
    delete *distG.getRemote(i);
  }
}

struct process {
  void operator()(GNode& _req, Galois::UserContext<GNode>& lwl) {
    std::vector<GNode> & SQ = *(*SQG.getLocal());
    std::vector<double> & sigma = *(*sigmaG.getLocal());
    std::vector<int> &d = *(*distG.getLocal());
    int QPush = 0;
    int QAt = 0;
    
#if DEBUG
    std::cerr << ".";
#endif

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
#if USE_SUCCS
	  std::vector<GNode> & slist = getSuccs(v);
          slist.push_back(w);
#else
	  std::vector<GNode> & plist = getSuccs(w);
          plist.push_back(v);
#endif
	}
      }
    }
    std::vector<double> & delta = *(*deltaG.getLocal());
#if USE_SUCCS
    --QAt;
#endif
    while (QAt > 1) {
      int w = SQ[--QAt];

      double sigma_w = sigma[w];
      double delta_w = delta[w];
#if USE_SUCCS
      std::vector<GNode> & slist = getSuccs(w);
      std::vector<GNode>::iterator it = slist.begin();
      std::vector<GNode>::iterator end = slist.end();
      while (it != end) {
	//std::cerr << "Processing node " << w << std::endl;
	GNode v = *it;
	delta_w += (sigma_w/sigma[v])*(1.0 + delta[v]);
	++it;
      }
      delta[w] = delta_w;
#else
      std::vector<GNode> & plist = getSuccs(w);
      std::vector<GNode>::iterator it = plist.begin();
      std::vector<GNode>::iterator end = plist.end();
      while (it != end) {
	//std::cerr << "Processing node " << w << std::endl;
	GNode v = *it;
	delta[v] += (sigma[v]/sigma_w)*(1.0 + delta_w);
	++it;
      }
#endif
#if SHARE_SINGLE_BC
      BCrecord & r = (*CB)[w].data;
      r.lock.lock();
      r.bc += delta_w;
      r.lock.unlock();
#else 
      CB->update(w, delta_w);
//      if (CB->get().size() < (unsigned int)w + 1)
//	CB->get().resize(w+1);
//      CB->get()[w] += delta_w;
#endif
    }
    resetData();
  }
};

// Verification for reference torus graph inputs. 
// All nodes should have the same betweenness value.
void verify() {
    double sampleBC = 0.0;
    bool firstTime = true;
#if SHARE_SINGLE_BC
#else
    const std::vector<double>& bcv = CB->reduce();
#endif
    for (int i=0; i<NumNodes; ++i) {
#if SHARE_SINGLE_BC
      double bc = (*CB)[i].data.bc;
#else
      double bc = bcv[i];
#endif
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
}

void printBCcertificate() {
  std::stringstream foutname;
  foutname << "outer_certificate_" << numThreads;
  std::ofstream outf(foutname.str().c_str());
  std::cerr << "Writting certificate..." << std::endl;
#if SHARE_SINGLE_BC
#else
  const std::vector<double>& bcv = CB->reduce();
#endif

  for (int i=0; i<NumNodes; ++i) {
#if SHARE_SINGLE_BC
      double bc = (*CB)[i].data.bc;
#else
      double bc = bcv[i];
#endif
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

#if SHARE_SINGLE_BC
  std::vector<cache_line_storage<BCrecord> > cb(NumNodes);
  CB = &cb;
#else
  Galois::GVectorElementAccumulator<std::vector<double> > cb;
  CB = &cb; 
#endif
  
  initGraphData();

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

  typedef GaloisRuntime::WorkList::dChunkedLIFO<1> WL;
  Galois::StatTimer T;
  T.start();
  Galois::for_each<WL>(tmp.begin(), tmp.end(), process());
  T.stop();

  if (forceVerify || !skipVerify) {
    verify();
  } else { // print bc value for first 10 nodes
#if SHARE_SINGLE_BC
#else
    const std::vector<double>& bcv = CB->reduce();
#endif
    for (int i=0; i<10; ++i)
#if SHARE_SINGLE_BC
    std::cout << i << ": " << setiosflags(std::ios::fixed) << std::setprecision(6) << (*CB)[i].data.bc << "\n";
#else
    std::cout << i << ": " << setiosflags(std::ios::fixed) << std::setprecision(6) << bcv[i] << "\n";
#endif
#if SHOULD_PRODUCE_CERTIFICATE
    printBCcertificate();
#endif
  }
  std::cerr << "Application done...\n";

  Galois::StatTimer tt("cleanup");
  tt.start();
  cleanupData();
  tt.stop();

  return 0;
}

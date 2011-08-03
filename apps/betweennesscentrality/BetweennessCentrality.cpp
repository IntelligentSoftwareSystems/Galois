/** Betweenness centrality application -*- C++ -*-
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
 * @author Dimitrios Prountzos <dprountz@cs.utexas.edu>
 */

#define DEBUG 0 
#define USE_SUCCS 1
#define SHARE_SINGLE_BC 0 
#define SHOULD_PRODUCE_CERTIFICATE 0
#define SHOULD_TOUCH_GRAPH 0

#include "Galois/Statistic.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Galois.h"

#if SHARE_SINGLE_BC
#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/CacheLineStorage.h"
#endif

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <vector>
#include <cstdlib>

static const char* name = "Betweenness Centrality";
static const char* description =
  "Computes the betweenness centrality of all nodes in a graph\n";
static const char* url = "betweenness_centrality";
static const char* help = "<input file> <number iterations>";

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

Galois::GReducible<std::vector<double>, merge >* CB;
#endif

GaloisRuntime::PerCPU<std::vector<GNode>*> SQG;
GaloisRuntime::PerCPU<std::vector<double> *> sigmaG;
GaloisRuntime::PerCPU<std::vector<double> *> deltaG;
GaloisRuntime::PerCPU<std::vector<int>*> distG;

template<typename T>
struct PerIt {  
  typedef typename Galois::PerIterAllocTy::rebind<T>::other Ty;
};

GaloisRuntime::PerCPU< std::vector<std::vector<GNode> > > succsGlobal;

std::vector<GNode> & getSuccs(GNode n) {
  return (succsGlobal.get())[n];
}

void initGraphData() {
  
  // Pre-compute successors sizes in tmp
  std::vector< std::vector<GNode> > tmp(NumNodes);
  for (Graph::active_iterator ii = G->active_begin(), ee = G->active_end();
      ii != ee; ++ii) {
    int nnbrs = std::distance(G->neighbor_begin(*ii, Galois::NONE),
        G->neighbor_end(*ii, Galois::NONE));
    //std::cerr << "Node : " << *ii << " has " << nnbrs << " neighbors " << std::endl;
    tmp[*ii].reserve(nnbrs); 
  }

  // Init all structures
  std::cerr << "Pre-allocating graph metadata for " << numThreads << " threads." << std::endl;
  for (int i=0; i<numThreads; ++i) {
    succsGlobal.get(i) = tmp;
    SQG.get(i) = new std::vector<GNode>(NumNodes); 
    sigmaG.get(i) = new std::vector<double>(NumNodes);
    deltaG.get(i) = new std::vector<double>(NumNodes); 
    distG.get(i) = new std::vector<int>(NumNodes); 
  }
}

void resetData() {
  std::vector<GNode> *sq = SQG.get();
  std::fill(sq->begin(), sq->end(), 0);

  std::vector<double> *sigma = sigmaG.get();
  std::fill(sigma->begin(), sigma->end(), 0);

  std::vector<double> *delta = deltaG.get();
  std::fill(delta->begin(), delta->end(), 0);

  std::vector<int> *dist = distG.get();
  std::fill(dist->begin(), dist->end(), 0);

  std::vector< std::vector<GNode> > & svec = succsGlobal.get();
  std::vector< std::vector<GNode> >::iterator it = svec.begin();
  std::vector< std::vector<GNode> >::iterator end = svec.end();
  while (it != end) {
    (*it).resize(0);
    ++it;
  }
}

void cleanupData() {
  for (int i=0; i<numThreads; ++i) {
    delete SQG.get(i);
    delete sigmaG.get(i);
    delete deltaG.get(i);
    delete distG.get(i);
  }
}

int dummyCnt;
void touchGraph() {
 for (Graph::active_iterator ii = G->active_begin(), ee = G->active_end();
       ii != ee; ++ii) {
   GNode v = *ii;
   for (Graph::neighbor_iterator
          jj = G->neighbor_begin(v, Galois::NONE),
          ee = G->neighbor_end(v, Galois::NONE); jj != ee; ++jj) {
	GNode w = *jj;
	dummyCnt++;
   } 
 }
 std::cerr << "Touching graph... " << dummyCnt << std::endl;
}

struct process {
  
  template<typename Context>
  void __attribute__((noinline)) operator() (GNode& _req, Context& lwl) {
    
    std::vector<GNode> & SQ = *(SQG.get());
    std::vector<double> & sigma = *(sigmaG.get());
    std::vector<int> &d = *(distG.get());
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
    std::vector<double> & delta = *(deltaG.get());
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
      if (CB->get().size() < (unsigned int)w + 1)
	CB->get().resize(w+1);
      CB->get()[w] += delta_w;
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
    for (int i=0; i<NumNodes; ++i) {
#if SHARE_SINGLE_BC
      double bc = (*CB)[i].data.bc;
#else
      double bc = CB->get()[i];
#endif
      if (firstTime) {
        sampleBC = bc;
        std::cerr << "BC: " << sampleBC << std::endl;
        firstTime = false;
      } else {
        if (!((bc - sampleBC) <= 0.0001)) {
          std::cerr << "If torus graph, verification failed " << (bc - sampleBC) << std::endl;
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
  
  for (int i=0; i<NumNodes; ++i) {
#if SHARE_SINGLE_BC
      double bc = (*CB)[i].data.bc;
#else
      double bc = CB->get()[i];
#endif
    outf << i << ": " << setiosflags(std::ios::fixed) << std::setprecision(9) << bc << std::endl;
  }
  outf.close();
}

int main(int argc, const char** argv) {

  std::vector<const char* > args = parse_command_line(argc, argv, help);

  if (args.size() < 1) {
    std::cerr
      << "incorrect number of arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  Graph g;
  G = &g;

  G->structureFromFile(args[0]);

  NumNodes = G->size();

#if SHARE_SINGLE_BC
  std::vector<cache_line_storage<BCrecord> > cb(NumNodes);
  CB = &cb;
#else
  Galois::GReducible<std::vector<double>, merge > cb;
  CB = &cb; 
#endif
  
  initGraphData();

  int iterations = NumNodes;
  if (args.size() == 2) {
    iterations = atoi(args[1]);
  }

  std::cerr << "NumNodes: " << NumNodes 
    << " Iterations: " << iterations << "\n";
  std::vector<GNode> tmp;
  
#if SHOULD_TOUCH_GRAPH
  touchGraph(); // pre-faulting graph to memory
#endif

  int cnt = 0;
  for (Graph::active_iterator ii = g.active_begin(), ee = g.active_end();
       ii != ee; ++ii) {
    if (cnt == iterations)
      break;
    // Only process nodes that actually have (out-)neighbors
    if (std::distance(g.neighbor_begin(*ii, Galois::NONE),
          g.neighbor_end(*ii, Galois::NONE)) > 0) {
      cnt++;
      tmp.push_back(*ii);
    }
  }
  std::cout << "Going parallel" << std::endl;
  Galois::setMaxThreads(numThreads);
  Galois::StatTimer T;
  T.start();
  Galois::for_each<GaloisRuntime::WorkList::LIFO<> >(tmp.begin(), tmp.end(), process());
  T.stop();

  if (!skipVerify) {
    verify();
  } else { // print bc value for first 10 nodes
    for (int i=0; i<10; ++i)
#if SHARE_SINGLE_BC
    std::cout << i << ": " << setiosflags(std::ios::fixed) << std::setprecision(6) << (*CB)[i].data.bc << "\n";
#else
    std::cout << i << ": " << setiosflags(std::ios::fixed) << std::setprecision(6) << CB->get()[i] << "\n";
#endif
#if SHOULD_PRODUCE_CERTIFICATE
    printBCcertificate();
#endif
  }
  std::cerr << "Application done...\n";
  Galois::Timer tt;
  tt.start();
  cleanupData();
  tt.stop();
  std::cerr << "Time to cleanup: " << tt.get() << std::endl;
  return 0;
}
// vim:ts=8:sts=2:sw=2

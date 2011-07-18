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
#include "Galois/Statistic.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <vector>
#include <cstdlib>

#ifdef GALOIS_VTUNE
#include "ittnotify.h"
#endif

#define DEBUG 0
#define USE_MMAP 0 
#define USE_GLOBALS 1 

static const char* name = "Betweenness Centrality";
static const char* description =
  "Computes the betweenness centrality of all nodes in a graph\n";
static const char* url = 0;
static const char* help = "<input file> <number iterations>";

typedef Galois::Graph::LC_FileGraph<int, int> Graph;
typedef Graph::GraphNode GNode;

Graph* G;
int NumNodes;

struct merge {
  void operator()(std::vector<double>& lhs, std::vector<double>& rhs) {
    if (lhs.size() < rhs.size())
      lhs.resize(rhs.size());
    for (unsigned int i = 0; i < rhs.size(); i++)
      lhs[i] += rhs[i];
  }
};

Galois::GReducible<std::vector<double>, merge >* CB;

#if USE_GLOBALS
GaloisRuntime::PerCPU<std::vector<GNode>*> SQG;
std::vector<GNode> & getSQ() {
  std::vector<GNode> *ans;
  if ((ans = SQG.get()) == 0) {
    ans = new std::vector<GNode>(NumNodes); 
    SQG.get() = ans;
  } else {
    std::fill(ans->begin(), ans->end(), 0);
  }
  return *ans;
}

GaloisRuntime::PerCPU<std::vector<double> *> sigmaG;
std::vector<double> & getSigma() {
  std::vector<double> *ans;
  if ((ans = sigmaG.get()) == 0) {
    ans = new std::vector<double>(NumNodes); 
    sigmaG.get() = ans;
  } else {
    std::fill(ans->begin(), ans->end(), 0.0);
  }
  return *ans;
}

GaloisRuntime::PerCPU<std::vector<double> *> deltaG;
std::vector<double> & getDelta() {
  std::vector<double> *ans;
  if ((ans = deltaG.get()) == 0) {
    ans = new std::vector<double>(NumNodes); 
    deltaG.get() = ans;
  } else {
    std::fill(ans->begin(), ans->end(), 0.0);
  }
  return *ans;
}

GaloisRuntime::PerCPU<std::vector<int>*> distG;
std::vector<int> & getDist() {
  std::vector<int> *ans;
  if ((ans = distG.get()) == 0) {
    ans = new std::vector<int>(NumNodes); 
    distG.get() = ans;
  } else {
    std::fill(ans->begin(), ans->end(), 0);
  }
  return *ans;
}
#endif

template<typename T>
struct PerIt {  
  typedef typename Galois::PerIterAllocTy::rebind<T>::other Ty;
};

GaloisRuntime::PerCPU< std::vector<std::vector<GNode> > > succsGlobal;

std::vector<GNode> & getSuccs(GNode n) {
  return (succsGlobal.get())[n];
}

void resetNodeSuccs() {
  std::vector< std::vector<GNode> > & svec = succsGlobal.get();
  std::vector< std::vector<GNode> >::iterator it = svec.begin();
  std::vector< std::vector<GNode> >::iterator end = svec.end();
  while (it != end) {
    std::vector<GNode> & v = *it;
//    std::fill(v.begin(), v.end(), 0);
    (*it).resize(0);
    ++it;
  }
}

struct process {
  

  typedef int tt_needs_per_iter_alloc;

  template<typename Context>
  void operator()(GNode& _req, Context& lwl) {
    
    resetNodeSuccs();

    int initSize = NumNodes;
    
    Galois::PerIterAllocTy& lalloc = lwl.getPerIterAlloc();

#if USE_GLOBALS
    std::vector<GNode> & SQ = getSQ();
    std::vector<double> & sigma = getSigma();
    std::vector<int> &d = getDist();
#else 
    std::vector<GNode,typename PerIt<GNode>::Ty> SQ(initSize, GNode(), lalloc);
    std::vector<double,typename PerIt<double>::Ty> sigma(initSize, 0.0, lalloc);
    std::vector<int,typename PerIt<int>::Ty> d(initSize, 0, lalloc);
#endif

#if USE_MMAP
    typedef std::multimap<int,int, std::less<int>,
            typename PerIt<std::pair<const int,int> >::Ty> MMapTy;
    MMapTy Succs(std::less<int>(), lalloc);
#else
//    std::cerr << "Not using mmap" << std::endl;
    //std::vector<VecTy*, typename PerIt<VecTy*>::Ty > Succs(NumNodes,  NULL, lalloc);
#endif
    
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
#if USE_MMAP
	  Succs.insert(std::pair<int, int>(v,w));
#else
          std::vector<GNode> & slist = getSuccs(v);
          slist.push_back(w);
#endif
	}
      }
    }
#if USE_GLOBALS
    std::vector<double> & delta = getDelta();
#else
    std::vector<double, typename PerIt<double>::Ty> delta(NumNodes, 0.0, lalloc);
#endif
    --QAt;
    while (QAt) {
      int w = SQ[--QAt];
      
      double sigma_w = sigma[w];
      double delta_w = delta[w];

#if USE_MMAP
      std::pair<MMapTy::iterator, MMapTy::iterator> ppp = Succs.equal_range(w);
      for (MMapTy::iterator ii = ppp.first, ee = ppp.second;
	  ii != ee; ++ii) {
	int v = ii->second;
	delta_w += (sigma_w/sigma[v])*(1.0 + delta[v]);
      }
      delta[w] = delta_w;
#else
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

#endif
      if (w != req) {
	if (CB->get().size() < (unsigned int)w + 1)
	  CB->get().resize(w+1);
	CB->get()[w] += delta_w;
      }
    }
  }
};

// Verification for reference torus graph inputs. 
// All nodes should have the same betweenness value.
void verify() {
    double sampleBC = 0.0;
    bool firstTime = true;
    for (int i=0; i<NumNodes; ++i) {
      if (firstTime) {
        sampleBC = CB->get()[i];
        std::cerr << "BC: " << sampleBC << std::endl;
        firstTime = false;
      } else {
        double bc = CB->get()[i];
        if (!((bc - sampleBC) <= 0.0001)) {
          std::cerr << "If torus graph, verification failed " << (bc - sampleBC) << std::endl;
	  assert ((bc - sampleBC) <= 0.0001);
	  return;
	}
      }
    }
    std::cerr << "Verification ok!" << std::endl;
}

void initNodeSuccs() {
  std::vector< std::vector<GNode> > tmp(NumNodes);
  for (Graph::active_iterator ii = G->active_begin(), ee = G->active_end();
      ii != ee; ++ii) {
    int nnbrs = std::distance(G->neighbor_begin(*ii, Galois::NONE),
        G->neighbor_end(*ii, Galois::NONE));
    //std::cerr << "Node : " << *ii << " has " << nnbrs << " neighbors " << std::endl;
    tmp[*ii].reserve(nnbrs); 
  }
  for (int i=0; i<numThreads; ++i) {
    succsGlobal.get(i) = tmp;
  }
  //succsGlobal.reset(tmp);
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
  G->emptyNodeData();

  NumNodes = G->size();

  Galois::GReducible<std::vector<double>, merge > cb;
  CB = &cb; 

  initNodeSuccs();

  int iterations = NumNodes;
  if (args.size() == 2) {
    iterations = atoi(args[1]);
  }

  std::cerr << "NumNodes: " << NumNodes 
    << " Iterations: " << iterations << "\n";
  std::vector<GNode> tmp;
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
  std::cout << "Going Parallel\n";
  Galois::setMaxThreads(numThreads);
  Galois::StatTimer T;
  T.start();
  Galois::for_each<GaloisRuntime::WorkList::LIFO<> >(tmp.begin(), tmp.end(), process());
  T.stop();

  if (!skipVerify) {
    verify();
  } else { // print bc value for first 10 nodes
    for (int i=0; i<10; ++i)
      std::cout << i << ": " << CB->get()[i] << "\n";
  }
  return 0;
}
// vim:ts=8:sts=2:sw=2

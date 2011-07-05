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
#include <stack>
#include <queue>
#include <vector>
#include <cstdlib>

#define DEBUG 0

static const char* name = "Betweenness Centrality";
static const char* description =
  "Computes the betweenness centrality of all nodes in a graph\n";
static const char* url = 0;
static const char* help = "<input file> <number iterations>";

typedef Galois::Graph::LC_FileGraph<int, int> Graph;
typedef Graph::GraphNode GNode;

Graph* G;
int NumNodes;

GaloisRuntime::PerCPU<std::vector<double> >* CB;

struct process {
  template<typename T>
  struct PerIt {
    typedef typename Galois::PerIterAllocTy::rebind<T>::other Ty;
  };

  typedef int tt_needs_per_iter_alloc;

  template<typename Context>
  void operator()(GNode& _req, Context& ctx) {
    int initSize = NumNodes;
    Galois::PerIterAllocTy& lalloc = ctx.getPerIterAlloc();
    
    std::vector<GNode,typename PerIt<GNode>::Ty> SQ(initSize, GNode(), lalloc);
    std::vector<double,typename PerIt<double>::Ty> sigma(initSize, 0.0, lalloc);
    std::vector<int,typename PerIt<int>::Ty> d(initSize, 0, lalloc);
 
    typedef std::multimap<int,int, std::less<int>,
            typename PerIt<std::pair<const int,int> >::Ty> MMapTy;
    MMapTy P(std::less<int>(), lalloc);
    
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
          ii = G->neighbor_begin(_v, Galois::Graph::NONE),
          ee = G->neighbor_end(_v, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode _w = *ii;
	int w = _w;
	if (!d[w]) {
	  SQ[QPush++] = _w;
	  d[w] = d[v] + 1;
	}
	if (d[w] == d[v] + 1) {
	  sigma[w] = sigma[w] + sigma[v];
	  P.insert(std::pair<int, int>(w,v));
	}
      }
    }
    
    std::vector<double> delta(NumNodes);
    while (QAt) {
      int w = SQ[--QAt];
      std::pair<MMapTy::iterator, MMapTy::iterator> ppp = P.equal_range(w);
      
      double sigma_w = sigma[w];
      double delta_w = delta[w];

      for (MMapTy::iterator ii = ppp.first, ee = ppp.second;
	   ii != ee; ++ii) {
	int v = ii->second;
	delta[v] = delta[v] + (sigma[v]/sigma_w)*(1.0 + delta_w);
      }
      if (w != req) {
	if (CB->get().size() < (unsigned int)w + 1)
	  CB->get().resize(w+1);
	CB->get()[w] += delta_w;
      }
    }
  }
};


  void merge(std::vector<double>& lhs, std::vector<double>& rhs) {
    if (lhs.size() < rhs.size())
    lhs.resize(rhs.size());
  for (unsigned int i = 0; i < rhs.size(); i++)
    lhs[i] += rhs[i];
}

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

int main(int argc, const char** argv) {

  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 1) {
    std::cerr
      << "incorrect number of arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  Graph g;
  G = &g;
  GaloisRuntime::PerCPU_merge<std::vector<double> > cb(merge);
  CB = &cb;

  G->structureFromFile(args[0]);
  G->emptyNodeData();

  NumNodes = G->size();
  
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
    if (std::distance(g.neighbor_begin(*ii, Galois::Graph::NONE),
          g.neighbor_end(*ii, Galois::Graph::NONE)) > 0) {
      cnt++;
      tmp.push_back(*ii);
    }
  }
  Galois::setMaxThreads(numThreads);
  Galois::StatTimer T;
  T.start();
  Galois::for_each<GaloisRuntime::WorkList::ChunkedLIFO<16> >(tmp.begin(), tmp.end(), process());
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

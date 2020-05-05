/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/UserContext.h"
#include "galois/Graph/LCGraph.h"
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
#include <deque>
#include <cstdlib>

#define SHOULD_PRODUCE_CERTIFICATE 0

static const char* name = "Betweenness Centrality";
static const char* desc =
    "Computes the betweenness centrality of all nodes in a graph";
static const char* url = "betweenness_centrality";

static llvm::cl::opt<std::string> filename(llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::Required);
static llvm::cl::opt<int> iterLimit(
    "limit",
    llvm::cl::desc("Limit number of iterations to value (0 is all nodes)"),
    llvm::cl::init(0));
static llvm::cl::opt<unsigned int>
    startNode("startNode", llvm::cl::desc("Node to start search from"),
              llvm::cl::init(0));
static llvm::cl::opt<bool> forceVerify(
    "forceVerify",
    llvm::cl::desc("Abort if not verified, only makes sense for torus graphs"));

typedef galois::graphs::LC_CSR_Graph<void, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode;

Graph* G;
int NumNodes;

galois::runtime::PerThreadStorage<double*> CB;

template <typename T>
struct PerIt {
  typedef typename galois::PerIterAllocTy::rebind<T>::other Ty;
};

struct process {
  typedef int tt_does_not_need_aborts;
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_push;

  void operator()(GNode& _req, galois::UserContext<GNode>& lwl) {
    typedef std::deque<GNode, PerIt<GNode>::Ty> GNdeque;
    GNdeque SQ(lwl.getPerIterAlloc());
    std::deque<double, PerIt<double>::Ty> sigma(NumNodes, 0.0,
                                                lwl.getPerIterAlloc());
    std::deque<int, PerIt<int>::Ty> d(NumNodes, 0, lwl.getPerIterAlloc());
    std::deque<double, PerIt<double>::Ty> delta(NumNodes, 0.0,
                                                lwl.getPerIterAlloc());
    std::deque<GNdeque, PerIt<GNdeque>::Ty> succ(
        NumNodes, GNdeque(lwl.getPerIterAlloc()), lwl.getPerIterAlloc());
    unsigned int QAt = 0;

    int req = _req;

    sigma[req] = 1;
    d[req]     = 1;

    SQ.push_back(_req);

    while (QAt != SQ.size()) {
      GNode _v = SQ[QAt++];
      int v    = _v;
      for (Graph::edge_iterator
               ii = G->edge_begin(_v, galois::MethodFlag::NONE),
               ee = G->edge_end(_v, galois::MethodFlag::NONE);
           ii != ee; ++ii) {
        GNode _w = G->getEdgeDst(ii);
        int w    = _w;
        if (!d[w]) {
          SQ.push_back(_w);
          d[w] = d[v] + 1;
        }
        if (d[w] == d[v] + 1) {
          sigma[w] = sigma[w] + sigma[v];
          succ[v].push_back(w);
        }
      }
    }

    while (SQ.size()) {
      int w = SQ.back();
      SQ.pop_back();

      double sigma_w = sigma[w];
      double delta_w = delta[w];
      auto slist     = succ[w];
      for (auto ii = slist.begin(), ee = slist.end(); ii != ee; ++ii) {
        // std::cerr << "Processing node " << w << std::endl;
        GNode v = *ii;
        delta_w += (sigma_w / sigma[v]) * (1.0 + delta[v]);
      }
      delta[w] = delta_w;
    }
    double* Vec = *CB.getLocal();
    for (unsigned int i = 0; i < delta.size(); ++i) {
      Vec[i] += delta[i];
    }
  }
};

// Verification for reference torus graph inputs.
// All nodes should have the same betweenness value.
void verify() {
  double sampleBC = 0.0;
  bool firstTime  = true;
  for (int i = 0; i < NumNodes; ++i) {
    double bc = (*CB.getRemote(0))[i];
    for (unsigned int j = 1; j < galois::getActiveThreads(); ++j)
      bc += (*CB.getRemote(j))[i];
    if (firstTime) {
      sampleBC = bc;
      std::cerr << "BC: " << sampleBC << std::endl;
      firstTime = false;
    } else {
      if (!((bc - sampleBC) <= 0.0001)) {
        std::cerr << "If torus graph, verification failed " << (bc - sampleBC)
                  << "\n";
        if (forceVerify)
          abort();
        assert((bc - sampleBC) <= 0.0001);
        return;
      }
    }
  }
}

void printBCcertificate() {
  std::stringstream foutname;
  foutname << "outer_certificate_" << numThreads;
  std::ofstream outf(foutname.str().c_str());
  std::cerr << "Writing certificate..." << std::endl;

  for (int i = 0; i < NumNodes; ++i) {
    double bc = (*CB.getRemote(0))[i];
    for (unsigned int j = 1; j < galois::getActiveThreads(); ++j)
      bc += (*CB.getRemote(j))[i];
    outf << i << ": " << std::setiosflags(std::ios::fixed)
         << std::setprecision(9) << bc << std::endl;
  }
  outf.close();
}

struct HasOut : public std::unary_function<GNode, bool> {
  Graph* graph;
  HasOut(Graph* g) : graph(g) {}
  bool operator()(const GNode& n) const {
    return graph->edge_begin(n) != graph->edge_end(n);
  }
};

struct InitializeLocal {
  void operator()(unsigned, unsigned) {
    *CB.getLocal() = (double*)galois::runtime::MM::pageAlloc();
    std::fill(&(*CB.getLocal())[0], &(*CB.getLocal())[NumNodes], 0.0);
  }
};

int main(int argc, char** argv) {
  galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);

  Graph g;
  G = &g;
  galois::graphs::readGraph(*G, filename);

  NumNodes = G->size();

  // CB.resize(NumNodes);
  // FIXME
  assert(galois::runtime::MM::pageSize >= NumNodes * sizeof(double));
  galois::on_each(InitializeLocal());

  galois::reportPageAlloc("MeminfoPre");
  galois::preAlloc(numThreads * galois::runtime::MM::numPageAllocTotal() / 3);
  galois::reportPageAlloc("MeminfoMid");

  boost::filter_iterator<HasOut, Graph::iterator>
      begin = boost::make_filter_iterator(HasOut(G), g.begin(), g.end()),
      end   = boost::make_filter_iterator(HasOut(G), g.end(), g.end());

  boost::filter_iterator<HasOut, Graph::iterator> begin2 =
      iterLimit ? galois::safe_advance(begin, end, (int)iterLimit) : end;

  size_t iterations = std::distance(begin, begin2);

  std::vector<GNode> v(begin, begin2);

  std::cout << "NumNodes: " << NumNodes << " Start Node: " << startNode
            << " Iterations: " << iterations << "\n";

  typedef galois::worklists::StableIterator<true> WLL;
  galois::StatTimer T;
  T.start();
  galois::for_each<WLL>(v.begin(), v.end(), process());
  T.stop();

  if (!skipVerify) {
    for (int i = 0; i < 10; ++i) {
      double bc = (*CB.getRemote(0))[i];
      for (unsigned int j = 1; j < galois::getActiveThreads(); ++j)
        bc += (*CB.getRemote(j))[i];
      std::cout << i << ": " << std::setiosflags(std::ios::fixed)
                << std::setprecision(6) << bc << "\n";
    }
#if SHOULD_PRODUCE_CERTIFICATE
    printBCcertificate();
#endif
  }

  galois::reportPageAlloc("MeminfoPost");

  if (forceVerify || !skipVerify) {
    verify();
  }
  std::cerr << "Application done...\n";

  return 0;
}

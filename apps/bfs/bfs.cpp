/** Single source shortest paths -*- C++ -*-
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
 * @section Description
 *
 * Example breadth-first search application for demoing Galois system. For optimized
 * version, use SSSP application with BFS option instead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>

namespace cll = llvm::cl;

static const char* name = "Breadth-first Search Example";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = 0;

static cll::opt<unsigned int> startNode("startnode", cll::desc("Node to start search from"), cll::init(1));
static cll::opt<unsigned int> reportNode("reportnode", cll::desc("Node to report distance to"), cll::init(2));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> algo("algo", cll::desc("Algorithm"), cll::init(0));

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

template<typename GrNode>
struct UpdateRequestCommon {
  GrNode n;
  unsigned int w;

  UpdateRequestCommon(const GrNode& N, unsigned int W): n(N), w(W) { }
  UpdateRequestCommon(): n(), w(0) { }
};

struct SNode {
  unsigned int id;
  unsigned int dist;
  
  SNode(int _id = -1) : id(_id), dist(DIST_INFINITY) {}
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out << "(id: " << n.id << ", dist: " << n.dist << ")";
  return out;
}

typedef Galois::Graph::LC_Linear_Graph<SNode, unsigned int> Graph;
typedef Graph::GraphNode GNode;
typedef UpdateRequestCommon<GNode> UpdateRequest;

struct UpdateRequestIndexer {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w;
    return t;
  }
};

Graph graph;

struct SerialAlgo {
  std::string name() const { return "Serial"; }

  void operator()(const GNode source) const {
    Galois::Statistic<unsigned int> counter("Iterations");

    std::deque<UpdateRequest> wl;
    wl.push_back(UpdateRequest(source, 0));

    while (!wl.empty()) {
      UpdateRequest req = wl.front();
      wl.pop_front();

      SNode& data = graph.getData(req.n);

      if (req.w < data.dist) {
        for (Graph::edge_iterator
               ii = graph.edge_begin(req.n), 
               ee = graph.edge_end(req.n);
             ii != ee; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          unsigned int newDist = req.w + 1;
          if (newDist < graph.getData(dst).dist)
            wl.push_back(UpdateRequest(dst, newDist));
        }
        data.dist = req.w;
      }

      counter += 1;
    }
  }
};

struct SerialFlagOptAlgo {
  std::string name() const { return "Serial (Flag Optimized)"; }

  void operator()(const GNode source) const {
    Galois::Statistic<unsigned int> counter("Iterations");

    std::deque<UpdateRequest> wl;
    wl.push_back(UpdateRequest(source, 0));

    while (!wl.empty()) {
      UpdateRequest req = wl.front();
      wl.pop_front();

      SNode& data = graph.getData(req.n, Galois::NONE);

      if (req.w < data.dist) {
        for (Graph::edge_iterator
               ii = graph.edge_begin(req.n, Galois::NONE), 
               ee = graph.edge_end(req.n, Galois::NONE);
             ii != ee; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          unsigned int newDist = req.w + 1;
          if (newDist < graph.getData(dst, Galois::NONE).dist)
            wl.push_back(UpdateRequest(dst, newDist));
        }
        data.dist = req.w;
      }

      counter += 1;
    }
  }
};

struct GaloisAlgo {
  std::string name() const { return "Galois"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<16> dChunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
  }

  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& ctx) const {
    SNode& data = graph.getData(req.n);
    if (req.w < data.dist) {
      for (Graph::edge_iterator 
             ii = graph.edge_begin(req.n),
             ee = graph.edge_end(req.n);
           ii != ee; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        unsigned int newDist = req.w + 1;
        SNode& rdata = graph.getData(dst);
        if (newDist < rdata.dist)
          ctx.push(UpdateRequest(dst, newDist));
      }
    }
    data.dist = req.w;
  }
};

struct GaloisNoLockAlgo {
  std::string name() const { return "Galois No Lock"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<16> dChunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
  }

  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& ctx) const {
    SNode& data = graph.getData(req.n, Galois::NONE);
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	for (Graph::edge_iterator 
               ii = graph.edge_begin(req.n, Galois::NONE),
	       ee = graph.edge_end(req.n, Galois::NONE);
             ii != ee; ++ii) {
	  GNode dst = graph.getEdgeDst(ii);
	  unsigned int newDist = req.w + 1;
	  SNode& rdata = graph.getData(dst, Galois::NONE);
	  if (newDist < rdata.dist)
	    ctx.push(UpdateRequest(dst, newDist));
	}
	break;
      }
    }
  }
};

bool verify(GNode source) {
  if (graph.getData(source,Galois::NONE).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  
  for (Graph::active_iterator src = graph.active_begin(), ee =
	 graph.active_end(); src != ee; ++src) {
    unsigned int dist = graph.getData(*src,Galois::NONE).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr << "found node = " << graph.getData(*src,Galois::NONE).id
	   << " with label >= INFINITY = " << dist << "\n";
      return false;
    }
    
    for (Graph::edge_iterator 
	   ii = graph.edge_begin(*src, Galois::NONE),
	   ee = graph.edge_end(*src, Galois::NONE); ii != ee; ++ii) {
      GNode neighbor = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(*src,Galois::NONE).dist;
      int d = 1;
      if (ddist > dist + d) {
        std::cerr << "bad level value at "
          << graph.getData(*src,Galois::NONE).id
	  << " which is a neighbor of " 
          << graph.getData(neighbor,Galois::NONE).id << "\n";
	return false;
      }
    }
  }
  return true;
}

template<typename AlgoTy>
void run(const AlgoTy& algo, const GNode& source) {
  Galois::StatTimer T;
  std::cerr << "Running " << algo.name() << " version\n";
  T.start();
  algo(source);
  T.stop();
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  graph.structureFromFile(filename);

  std::cout << "Read " << graph.size() << " nodes\n";
  
  unsigned int id = 0;
  bool foundReport = false;
  bool foundSource = false;
  GNode source = *graph.active_begin();
  GNode report = *graph.active_begin();
  for (Graph::active_iterator src = graph.active_begin(), ee =
      graph.active_end(); src != ee; ++src) {
    SNode& node = graph.getData(*src, Galois::NONE);
    node.id = id++;
    node.dist = DIST_INFINITY;
    if (node.id == startNode) {
      source = *src;
      foundSource = true;
    } 
    if (node.id == reportNode) {
      foundReport = true;
      report = *src;
    }
  }

  if (!foundReport || !foundSource) {
    std::cerr << "failed to set report: " << reportNode 
      << "or failed to set source: " << startNode << ".\n";
    assert(0);
    abort();
  }

  switch (algo) {
    case 0: run(SerialAlgo(), source); break;
    case 1: run(SerialFlagOptAlgo(), source); break;
    case 2: run(GaloisAlgo(), source); break;
    case 3: run(GaloisNoLockAlgo(), source); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }

  std::cout << "Report node: " << graph.getData(report, Galois::NONE) << "\n";

  if (!skipVerify && !verify(source)) {
    std::cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}

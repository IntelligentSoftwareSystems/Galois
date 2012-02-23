/** Breadth-first search -*- C++ -*-
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
#ifdef GALOIS_EXP
#include "PriorityScheduling/WorkList.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>


static const char* name = "Breadth-first Search Example";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = 0;

//****** Command Line Options ******
namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(1));
static cll::opt<unsigned int> reportNode("reportnode",
    cll::desc("Node to report distance to"),
    cll::init(2));
static cll::opt<int> algo("algo",
    cll::desc("Algorithm"),
    cll::init(0));
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int id;
  unsigned int dist;
  
  SNode(int _id = -1) : id(_id), dist(DIST_INFINITY) {}
};

typedef Galois::Graph::LC_Linear_Graph<SNode, unsigned int> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

struct UpdateRequest {
  GNode n;
  unsigned int w;

  UpdateRequest(): w(0) { }
  UpdateRequest(const GNode& N, unsigned int W): n(N), w(W) { }
  bool operator<(const UpdateRequest& o) const { return w < o.w; }
  bool operator>(const UpdateRequest& o) const { return w > o.w; }
  unsigned getID() const { return graph.getData(n).id; }
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out << "(id: " << n.id << ", dist: " << n.dist << ")";
  return out;
}

struct UpdateRequestIndexer {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w;
    return t;
  }
};

//! Simple verifier
static bool verify(GNode source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  
  for (Graph::active_iterator src = graph.active_begin(), ee =
	 graph.active_end(); src != ee; ++src) {
    unsigned int dist = graph.getData(*src).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr
        << "found node = " << graph.getData(*src).id
	<< " with label >= INFINITY = " << dist << "\n";
      return false;
    }
    
    for (Graph::edge_iterator 
	   ii = graph.edge_begin(*src),
	   ee = graph.edge_end(*src); ii != ee; ++ii) {
      GNode neighbor = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(*src).dist;
      if (ddist > dist + 1) {
        std::cerr
          << "bad level value at " << graph.getData(*src).id
	  << " which is a neighbor of " << graph.getData(neighbor).id << "\n";
	return false;
      }
    }
  }
  return true;
}

static void readGraph(GNode& source, GNode& report) {
  graph.structureFromFile(filename);
  source = *graph.active_begin();
  report = *graph.active_begin();

  std::cout << "Read " << graph.size() << " nodes\n";
  
  unsigned int id = 0;
  bool foundReport = false;
  bool foundSource = false;
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
    std::cerr 
      << "failed to set report: " << reportNode 
      << "or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
}


//! Serial BFS using Galois graph
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

//! Serial BFS using optimized flags but using Galois graph
struct SerialFlagOptAlgo {
  std::string name() const { return "Serial (Flag Optimized)"; }

  void operator()(const GNode source) const {
    Galois::Statistic<unsigned int> counter("Iterations");

    std::deque<UpdateRequest> wl;
    wl.push_back(UpdateRequest(source, 0));

    while (!wl.empty()) {
      UpdateRequest req = wl.front();
      wl.pop_front();

      // Operator begins here
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

//! Galois BFS
struct GaloisAlgo {
  std::string name() const { return "Galois"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
#ifdef GALOIS_EXP
    Exp::StartWorklistExperiment<OBIM,dChunk,Chunk,UpdateRequestIndexer,std::less<UpdateRequest>,std::greater<UpdateRequest> >()(std::cout, &one[0], &one[1], *this);
#else
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
#endif
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

//! Galois BFS using optimized flags
struct GaloisNoLockAlgo {
  std::string name() const { return "Galois No Lock"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
#ifdef GALOIS_EXP
    Exp::StartWorklistExperiment<OBIM,dChunk,Chunk,UpdateRequestIndexer,std::less<UpdateRequest>,std::greater<UpdateRequest> >()(std::cout, &one[0], &one[1], *this);
#else
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
#endif
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

  GNode source, report;
  readGraph(source, report);

  switch (algo) {
    case 0: run(SerialAlgo(), source); break;
    case 1: run(SerialFlagOptAlgo(), source); break;
    case 2: run(GaloisAlgo(), source); break;
    case 3: run(GaloisNoLockAlgo(), source); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }

  std::cout << "Report node: " << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify(source)) {
      std::cout << "Verification successful.\n";
    } else {
      std::cerr << "Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }

  return 0;
}

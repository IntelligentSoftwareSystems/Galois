/** Single source shortest paths -*- C++ -*-
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
 * @section Description
 *
 * Single source shortest paths.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "SSSP.h"

#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/UserContext.h"
#include "Galois/Graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_USE_EXP
#include "Galois/PriorityScheduling.h"
#endif

#include <iostream>
#include <set>

namespace cll = llvm::cl;

static const bool trackBadWork = false;

static const char* name = "Single Source Shortest Path";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = "single_source_shortest_path";

enum SSSPAlgo {
  serialStl,
  serialPairing,
  parallel
};

static cll::opt<unsigned int> startNode("startnode", cll::desc("Node to start search from"), cll::init(1));
static cll::opt<unsigned int> reportNode("reportnode", cll::desc("Node to report distance to"), cll::init(2));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> stepShift("delta", cll::desc("Shift value for the deltastep"), cll::init(10));
static cll::opt<SSSPAlgo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serialStl, "Serial using STL heap"),
      clEnumVal(serialPairing, "Serial using pairing heap"),
      clEnumVal(parallel, "Parallel"),
      clEnumValEnd), cll::init(parallel));

typedef Galois::Graph::LC_Linear_Graph<SNode, unsigned int> Graph;
typedef Graph::GraphNode GNode;

typedef UpdateRequestCommon<GNode> UpdateRequest;

struct UpdateRequestIndexer: public std::unary_function<UpdateRequest, unsigned int> {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w >> stepShift;
    return t;
  }
};

Graph graph;

struct SerialStlAlgo {
  std::string name() const { return "serial stl heap"; }

  void operator()(const GNode src) const {
    graph.getData(src, Galois::NONE).dist = 0;

    std::set<UpdateRequest, std::less<UpdateRequest> > initial;
    for (Graph::edge_iterator
           ii = graph.edge_begin(src, Galois::NONE), 
           ee = graph.edge_end(src, Galois::NONE); 
         ii != ee; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      int w = graph.getEdgeData(ii);
      UpdateRequest up(dst, w);
      initial.insert(up);
    }

    Galois::Statistic counter("Iterations");
    
    while (!initial.empty()) {
      counter += 1;
      UpdateRequest req = *initial.begin();
      initial.erase(initial.begin());
      SNode& data = graph.getData(req.n, Galois::NONE);

      if (req.w < data.dist) {
        data.dist = req.w;
        for (Graph::edge_iterator
               ii = graph.edge_begin(req.n, Galois::NONE), 
               ee = graph.edge_end(req.n, Galois::NONE);
             ii != ee; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          int d = graph.getEdgeData(ii);
          unsigned int newDist = req.w + d;
          if (newDist < graph.getData(dst,Galois::NONE).dist)
            initial.insert(UpdateRequest(dst, newDist));
        }
      }
    }
  }
};

#ifdef GALOIS_USE_EXP
struct SerialPairingHeapAlgo {
  std::string name() const { return "serial pairing heap"; }

  void operator()(const GNode src) const {
    graph.getData(src, Galois::NONE).dist = 0;

    Galois::PairingHeap<UpdateRequest, std::less<UpdateRequest> > initial;
    for (Graph::edge_iterator
           ii = graph.edge_begin(src, Galois::NONE), 
           ee = graph.edge_end(src, Galois::NONE); 
         ii != ee; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      int w = graph.getEdgeData(ii);
      UpdateRequest up(dst, w);
      initial.add(up);
    }

    Galois::Statistic counter("Iterations");
    
    boost::optional<UpdateRequest> req;

    while ((req = initial.pollMin())) { 
      counter += 1;
      SNode& data = graph.getData(req->n, Galois::NONE);

      if (req->w < data.dist) {
        data.dist = req->w;
        for (Graph::edge_iterator
               ii = graph.edge_begin(req->n, Galois::NONE), 
               ee = graph.edge_end(req->n, Galois::NONE);
             ii != ee; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          int d = graph.getEdgeData(ii);
          unsigned int newDist = req->w + d;
          if (newDist < graph.getData(dst,Galois::NONE).dist)
            initial.add(UpdateRequest(dst, newDist));
        }
      }
    }
  }
};
#endif

static Galois::Statistic* BadWork;
static Galois::Statistic* WLEmptyWork;

struct ParallelAlgo {
  std::string name() const { return "parallel"; }

  void operator()(const GNode src) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<16> dChunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk,true> OBIM;

    std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
    std::cout << "WARNING: Performance varies considerably due to -delta.  Do not expect the default to be good for your graph\n";

    UpdateRequest one[1] = { UpdateRequest(src, 0) };
#ifdef GALOIS_USE_EXP
    Exp::PriAuto<16, UpdateRequestIndexer, OBIM, std::less<UpdateRequest>, std::greater<UpdateRequest> >::for_each(&one[0], &one[1], *this);
#else
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
#endif
  }

  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& lwl) const {
    SNode& data = graph.getData(req.n,Galois::NONE);
    if (trackBadWork && req.w >= data.dist)
      *WLEmptyWork += 1;
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	if (trackBadWork && v != DIST_INFINITY)
	   *BadWork += 1;
	for (Graph::edge_iterator ii = graph.edge_begin(req.n, Galois::NONE),
	       ee = graph.edge_end(req.n, Galois::NONE); ii != ee; ++ii) {
	  GNode dst = graph.getEdgeDst(ii);
	  int d = graph.getEdgeData(ii);
	  unsigned int newDist = req.w + d;
	  SNode& rdata = graph.getData(dst,Galois::NONE);
	  if (newDist < rdata.dist)
	    lwl.push(UpdateRequest(dst, newDist));
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
  
  for (Graph::iterator src = graph.begin(), ee =
	 graph.end(); src != ee; ++src) {
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
      int d = graph.getEdgeData(ii);
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

void initGraph(GNode& source, GNode& report) {
  graph.structureFromFile(filename);

  unsigned int id = 0;
  bool foundReport = false;
  bool foundSource = false;
  source = *graph.begin();
  report = *graph.begin();
  for (Graph::iterator src = graph.begin(), ee =
      graph.end(); src != ee; ++src) {
    SNode& node = graph.getData(*src,Galois::NONE);
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

  if (!foundReport) {
    std::cerr << "Failed to set report (" << reportNode << ").\n";
    abort();
  }

  if (!foundSource) {
    std::cerr << "Failed to set source (" << startNode << ").\n";
    abort();
  }
}

template<typename AlgoTy>
void run(const AlgoTy& algo, GNode source) {
  Galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  algo(source);
  T.stop();
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (trackBadWork) {
    BadWork = new Galois::Statistic("BadWork");
    WLEmptyWork = new Galois::Statistic("EmptyWork");
  }

  GNode source, report;
  initGraph(source, report);
  std::cout << "Read " << graph.size() << " nodes\n";

  switch (algo) {
    case serialStl: run(SerialStlAlgo(), source); break;
#ifdef GALOIS_USE_EXP
    case serialPairing: run(SerialPairingHeapAlgo(), source); break;
#endif
    case parallel: run(ParallelAlgo(), source); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }

  if (trackBadWork) {
    delete BadWork;
    delete WLEmptyWork;
  }

  std::cout << graph.getData(report,Galois::NONE).toString() << "\n";
  if (!skipVerify && !verify(source)) {
    std::cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}

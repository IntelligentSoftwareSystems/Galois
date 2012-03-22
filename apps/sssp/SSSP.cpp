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
#ifdef GALOIS_EXP
#include "Galois/PriorityScheduling.h"
#endif

#include <iostream>
#include <set>

namespace cll = llvm::cl;

static const char* name = "Single Source Shortest Path";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = "single_source_shortest_path";

static cll::opt<unsigned int> startNode("startnode", cll::desc("Node to start search from"), cll::init(1));
static cll::opt<unsigned int> reportNode("reportnode", cll::desc("Node to report distance to"), cll::init(2));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> stepShift("delta", cll::desc("Shift value for the deltastep"), cll::init(10));
static cll::opt<bool> useBfs("bfs", cll::desc("Use BFS"), cll::init(false));

typedef Galois::Graph::LC_Linear_Graph<SNode, unsigned int> Graph;
typedef Graph::GraphNode GNode;

typedef UpdateRequestCommon<GNode> UpdateRequest;

struct UpdateRequestIndexer
  : std::unary_function<UpdateRequest, unsigned int> {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w >> stepShift;
    return t;
  }
};

Graph graph;

void runBody(const GNode src) {
  std::set<UpdateRequest, std::less<UpdateRequest> > initial;
  for (Graph::edge_iterator
	 ii = graph.edge_begin(src, Galois::NONE), 
	 ee = graph.edge_end(src, Galois::NONE); 
       ii != ee; ++ii) {
    GNode dst = graph.getEdgeDst(ii);
    int w = useBfs ? 1 : graph.getEdgeData(ii);
    UpdateRequest up(dst, w);
    initial.insert(up);
  }

  Galois::Statistic<unsigned int> counter("Iterations");
  Galois::StatTimer T;
  T.start();
  
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
	int d = useBfs ? 1 : graph.getEdgeData(ii);
	unsigned int newDist = req.w + d;
	if (newDist < graph.getData(dst,Galois::NONE).dist)
	  initial.insert(UpdateRequest(dst, newDist));
      }
    }
  }
  T.stop();
}

//static Galois::Statistic<unsigned int>* BadWork;
//static Galois::Statistic<unsigned int>* WLEmptyWork;

struct process {
  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& lwl) {
    SNode& data = graph.getData(req.n,Galois::NONE);
    // if (req.w >= data.dist)
    //   *WLEmptyWork += 1;
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	// if (v != DIST_INFINITY)
	//   *BadWork += 1;
	for (Graph::edge_iterator ii = graph.edge_begin(req.n, Galois::NONE),
	       ee = graph.edge_end(req.n, Galois::NONE); ii != ee; ++ii) {
	  GNode dst = graph.getEdgeDst(ii);
	  int d = useBfs ? 1 : graph.getEdgeData(ii);
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

void runBodyParallel(const GNode src) {
  using namespace GaloisRuntime::WorkList;
  typedef dChunkedLIFO<16> dChunk;
  typedef ChunkedLIFO<16> Chunk;
  typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

  Galois::StatTimer T;

  UpdateRequest one[1] = { UpdateRequest(src, 0) };
  T.start();
#ifdef GALOIS_EXP
    Exp::WorklistExperiment<OBIM,dChunk,Chunk,UpdateRequestIndexer,
      std::less<UpdateRequest>,std::greater<UpdateRequest> >().for_each(
        std::cout, &one[0], &one[1], process());
#else
  Galois::for_each<OBIM>(&one[0], &one[1], process());
#endif
  T.stop();
}


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
      int d = useBfs ? 1 : graph.getEdgeData(ii);
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

int main(int argc, char **argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  // Galois::Statistic<unsigned int> sBadWork("BadWork");
  // Galois::Statistic<unsigned int> sWLEmptyWork("WLEmptyWork");
  // BadWork = &sBadWork;
  // WLEmptyWork = &sWLEmptyWork;

  graph.structureFromFile(filename);

  std::cout << "Read " << graph.size() << " nodes\n";
  std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
  
  if (useBfs && stepShift > 1) {
    std::cout << "WARNING: Using a large delta-step for bfs. Expect long execution times.\n";
  }

  std::cout << "WARNING: Performance varies considerably due to -delta.  Do not expect the default to be good for your graph\n";

  unsigned int id = 0;
  bool foundReport = false;
  bool foundSource = false;
  GNode source = *graph.begin();
  GNode report = *graph.begin();
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
    assert(0);
    abort();
  }
  if (!foundSource) {
    std::cerr << "Failed to set source (" << startNode << ".\n";
    assert(0);
    abort();
  }

  if (numThreads) {
    runBodyParallel(source);
  } else {
    std::cout << "Running Sequentially\n";
    runBody(source);
  }

  std::cout << graph.getData(report,Galois::NONE).toString() << "\n";
  if (!skipVerify && !verify(source)) {
    std::cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}

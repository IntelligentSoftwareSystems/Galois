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
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/DebugWorkList.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include "WorkListTL.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>

static const char* name = "Single Source Shortest Path";
static const char* description =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = "single_source_shortest_path";
static const char* help =
  "<input file> <startnode> <reportnode> [-delta <deltaShift>] [-bfs]";

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

static unsigned int stepShift = 10;

static std::string wlName;

struct SNode {
  unsigned int id;
  unsigned int dist;
  
  SNode(int _id = -1) : id(_id), dist(DIST_INFINITY) {}
  std::string toString() {
    std::ostringstream s;
    s << '[' << id << "] dist: " << dist;
    return s.str();
  }
};

typedef Galois::Graph::LC_FileGraph<SNode, unsigned int> Graph;
typedef Galois::Graph::LC_FileGraph<SNode, unsigned int>::GraphNode GNode;

struct UpdateRequest {
  GNode n;
  unsigned int w;

  UpdateRequest(GNode& N, unsigned int W)
    :n(N), w(W)
  {}

  UpdateRequest()
    :n(), w(0)
  {}

  bool operator>(const UpdateRequest& rhs) const {
    return w > rhs.w;
  }

  bool operator<(const UpdateRequest& rhs) const {
    return w < rhs.w;
  }
};

struct seq_less {
  bool operator()(const UpdateRequest& lhs, const UpdateRequest& rhs) {
    //return (lhs.w  >> stepShift) < (rhs.w >> stepShift);
    if (lhs.w < rhs.w) return true;
    else if (lhs.w > rhs.w) return false;
    else return lhs.n < rhs.n;
  }
};
struct seq_gt {
  bool operator()(const UpdateRequest& lhs, const UpdateRequest& rhs) {
    //return (lhs.w  >> stepShift) < (rhs.w >> stepShift);
    if (lhs.w > rhs.w) return true;
    else if (lhs.w < rhs.w) return false;
    else return lhs.n > rhs.n;
  }
};

struct UpdateRequestIndexer
  : std::binary_function<UpdateRequest, unsigned int, unsigned int> {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w >> stepShift;
    return t;
  }
};

Graph graph;
bool do_bfs = false;

template<typename WLTy>
void getInitialRequests(const GNode src, WLTy& wl) {
  for (Graph::neighbor_iterator
      ii = graph.neighbor_begin(src, Galois::NONE), 
      ee = graph.neighbor_end(src, Galois::NONE); 
      ii != ee; ++ii) {
    GNode dst = *ii;
    int w = do_bfs ? 1 : graph.getEdgeData(src, dst, Galois::NONE);
    UpdateRequest up(dst, w);
    wl.push_back(up);
  }
}

void runBody(const GNode src) {
  std::set<UpdateRequest, seq_less> initial;
  for (Graph::neighbor_iterator
      ii = graph.neighbor_begin(src, Galois::NONE), 
      ee = graph.neighbor_end(src, Galois::NONE); 
      ii != ee; ++ii) {
    GNode dst = *ii;
    int w = do_bfs ? 1 : graph.getEdgeData(src, dst, Galois::NONE);
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
      for (Graph::neighbor_iterator
          ii = graph.neighbor_begin(req.n, Galois::NONE), 
	  ee = graph.neighbor_end(req.n, Galois::NONE);
	  ii != ee; ++ii) {
	GNode dst = *ii;
	int d = do_bfs ? 1 : graph.getEdgeData(req.n, dst, Galois::NONE);
	unsigned int newDist = req.w + d;
	if (newDist < graph.getData(dst,Galois::NONE).dist)
	  initial.insert(UpdateRequest(dst, newDist));
      }
    }
  }
  T.stop();
}

static Galois::Statistic<unsigned int> BadWork("BadWork");

struct process {
  template<typename ContextTy>
  void __attribute__((noinline)) operator()(UpdateRequest& req, ContextTy& lwl) {
    SNode& data = graph.getData(req.n,Galois::NONE);
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	if (v != DIST_INFINITY)
	  BadWork += 1;
	for (Graph::neighbor_iterator
            ii = graph.neighbor_begin(req.n, Galois::NONE),
            ee = graph.neighbor_end(req.n, Galois::NONE); ii != ee; ++ii) {
	  GNode dst = *ii;
	  int d = do_bfs ? 1 : graph.getEdgeData(req.n, dst, Galois::NONE);
	  unsigned int newDist = req.w + d;
	  if (newDist < graph.getData(dst,Galois::NONE).dist)
	    lwl.push(UpdateRequest(dst, newDist));
	}
	break;
      }
    }
  }
};

void runBodyParallel(const GNode src) {
  using namespace GaloisRuntime::WorkList;
  typedef dChunkedLIFO<16> IChunk;
  typedef ChunkedLIFO<16> NAChunk;

  typedef OrderedByIntegerMetric<UpdateRequestIndexer, IChunk> OBIM;
  typedef TbbPriQueue<seq_gt> TBB;
  typedef LocalStealing<TBB > LTBB;
  typedef SkipListQueue<seq_less> SLQ;
  typedef SimpleOrderedByIntegerMetric<UpdateRequestIndexer> SOBIM;
  typedef LocalStealing<SOBIM > LSOBIM;
  typedef OrderedByIntegerMetric<UpdateRequestIndexer, NAChunk> NAOBIM;
  typedef CTOrderedByIntegerMetric<UpdateRequestIndexer, IChunk> CTOBIM;
  typedef LocalStealing<SOBIM > LSOBIM;
  typedef FCPairingHeapQueue<seq_less> FCPH;

  typedef WorkListTracker<UpdateRequestIndexer, OBIM> TR_OBIM;
  typedef WorkListTracker<UpdateRequestIndexer, TBB>  TR_TBB;
  typedef WorkListTracker<UpdateRequestIndexer, LTBB> TR_LTBB;
  typedef WorkListTracker<UpdateRequestIndexer, SLQ>  TR_SLQ;
  typedef WorkListTracker<UpdateRequestIndexer, SOBIM> TR_SOBIM;
  typedef WorkListTracker<UpdateRequestIndexer, LSOBIM> TR_LSOBIM;
  typedef WorkListTracker<UpdateRequestIndexer, NAOBIM> TR_NAOBIM;
  typedef WorkListTracker<UpdateRequestIndexer, CTOBIM> TR_CTOBIM;

  typedef NoInlineFilter<OBIM> NI_OBIM;
  typedef NoInlineFilter<TBB>  NI_TBB;
  typedef NoInlineFilter<LTBB> NI_LTBB;
  typedef NoInlineFilter<SLQ>  NI_SLQ;
  typedef NoInlineFilter<SOBIM> NI_SOBIM;
  typedef NoInlineFilter<LSOBIM> NI_LSOBIM;
  typedef NoInlineFilter<NAOBIM> NI_NAOBIM;
  typedef NoInlineFilter<CTOBIM> NI_CTOBIM;


  std::vector<UpdateRequest> wl;
  getInitialRequests(src, wl);
  Galois::StatTimer T;

#define WLFOO(__x, __y) else if (wlName == #__x) { T.start(); Galois::for_each<__y>(wl.begin(), wl.end(), process()); T.stop(); } else if (wlName == "tr_" #__x) { T.start(); Galois::for_each<TR_ ## __y>(wl.begin(), wl.end(), process()); T.stop(); }else if (wlName == "ni_" #__x) { T.start(); Galois::for_each<NI_ ## __y>(wl.begin(), wl.end(), process()); T.stop(); }

  if (wlName == "fcph") {
    T.start();
    Galois::for_each<FCPH>(wl.begin(), wl.end(), process());
    T.stop();
  }
  WLFOO(obim, OBIM)
  WLFOO(sobim, SOBIM)
  WLFOO(lsobim, LSOBIM)
  WLFOO(naobim, NAOBIM)
  WLFOO(ctobim, CTOBIM)
  WLFOO(slq, SLQ)
  WLFOO(tbb, TBB)
  WLFOO(ltbb, LTBB)
  else {
    std::cout << "Unrecognized worklist " << wlName << "\n";
  }
}


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
    
    for (Graph::neighbor_iterator 
        ii = graph.neighbor_begin(*src, Galois::NONE),
        ee = graph.neighbor_end(*src, Galois::NONE); ii != ee; ++ii) {
      GNode neighbor = *ii;
      unsigned int ddist = graph.getData(*src,Galois::NONE).dist;
      int d = do_bfs ? 1 : graph.getEdgeData(*src, neighbor, Galois::NONE);
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

int main(int argc, const char **argv) {
  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 3) {
    std::cerr << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);
  
  const char* inputfile = args[0];
  unsigned int startNode = atoi(args[1]);
  unsigned int reportNode = atoi(args[2]);
  for (unsigned i = 3; i < args.size(); ++i) {
    if (strcmp(args[i], "-delta") == 0 && i + 1 < args.size()) {
      stepShift = atoi(args[i+1]);
      ++i;
    } else if (strcmp(args[i], "-bfs") == 0) {
      do_bfs = true;
    } else if (strcmp(args[i], "-wl") == 0) {
      wlName = args[i+1];
      ++i;
    } else {
      std::cerr << "unknown argument, use -help for usage information\n";
      return 1;
    }
  }

  GNode source = -1;
  GNode report = -1;
  
  graph.structureFromFile(inputfile);
  graph.emptyNodeData();
  std::cout << "Read " << graph.size() << " nodes\n";
  std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
  std::cout << "Using worklist of " << wlName << "\n";
  
  unsigned int id = 0;
  for (Graph::active_iterator src = graph.active_begin(), ee =
      graph.active_end(); src != ee; ++src) {
    SNode& node = graph.getData(*src,Galois::NONE);
    node.id = id++;
    node.dist = DIST_INFINITY;
    //std::cout << node.toString() << "\n";
    
    if (*src == startNode) {
      //if (node.id == startNode) {
      source = *src;
      node.dist = 0;
    } else if (*src == reportNode) {
      //} else if (node.id == reportNode) {
      report = *src;
    }
  }
  if (report == GNode(-1)) {
    std::cerr << "Failed to set report (" << reportNode << ").\n";
    assert(0);
    abort();
  }
  if (source == GNode(-1)) {
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

  std::cout << report << " " 
       << graph.getData(report,Galois::NONE).toString() << "\n";
  if (!skipVerify && !verify(source)) {
    std::cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}
// vim sw=2:ts=8:sts=2

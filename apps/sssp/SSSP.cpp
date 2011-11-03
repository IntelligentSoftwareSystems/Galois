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
#include "SSSP.h"

#include "Exp/PriorityScheduling/WorkListTL.h"

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
static const char* help = "[-delta <deltaShift>] [-bfs] <input file> <startnode> <reportnode>";

static unsigned int stepShift = 10;


typedef Galois::Graph::LC_FileGraph<SNode, unsigned int> Graph;
typedef Galois::Graph::LC_FileGraph<SNode, unsigned int>::GraphNode GNode;

typedef UpdateRequestCommon<GNode> UpdateRequest;



struct seq_less: public std::binary_function<const UpdateRequest&,const UpdateRequest&,bool> {
  bool operator()(const UpdateRequest& lhs, const UpdateRequest& rhs) const {
    //return (lhs.w  >> stepShift) < (rhs.w >> stepShift);
    if (lhs.w < rhs.w) return true;
    else if (lhs.w > rhs.w) return false;
    else return lhs.n < rhs.n;
  }
};

struct seq_greater: public std::binary_function<const UpdateRequest&,const UpdateRequest&,bool> {
  bool operator()(const UpdateRequest& lhs, const UpdateRequest& rhs) const {
    //return (lhs.w  >> stepShift) < (rhs.w >> stepShift);
    if (lhs.w > rhs.w) return true;
    else if (lhs.w < rhs.w) return false;
    else return lhs.n > rhs.n;
  }
};

struct UpdateRequestIndexer
  : std::unary_function<UpdateRequest, unsigned int> {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w >> stepShift;
    return t;
  }
};

Graph graph;
bool do_bfs = false;

void runBody(const GNode src) {
  std::set<UpdateRequest, seq_less> initial;
  for (Graph::neighbor_iterator
      ii = graph.neighbor_begin(src, Galois::NONE), 
      ee = graph.neighbor_end(src, Galois::NONE); 
      ii != ee; ++ii) {
    GNode dst = *ii;
    int w = do_bfs ? 1 : graph.getEdgeData(src, dst, Galois::NONE);
    UpdateRequest up(dst, w, 0);
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
	  initial.insert(UpdateRequest(dst, newDist, 0));
      }
    }
  }
  T.stop();
}

static Galois::Statistic<unsigned int>* BadWork;
static Galois::Statistic<unsigned int>* WLEmptyWork;

struct process {
  template<typename ContextTy>
  void __attribute__((noinline)) operator()(UpdateRequest& req, ContextTy& lwl) {
    SNode& data = graph.getData(req.n,Galois::NONE);
    if (req.w >= data.dist)
      *WLEmptyWork += 1;
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	if (v != DIST_INFINITY)
	  *BadWork += 1;
	for (Graph::neighbor_iterator
            ii = graph.neighbor_begin(req.n, Galois::NONE),
            ee = graph.neighbor_end(req.n, Galois::NONE); ii != ee; ++ii) {
	  GNode dst = *ii;
	  int d = do_bfs ? 1 : graph.getEdgeData(req.n, dst, Galois::NONE);
	  unsigned int newDist = req.w + d;
	  SNode& rdata = graph.getData(dst,Galois::NONE);
	  if (newDist < rdata.dist)
	    lwl.push(UpdateRequest(dst, newDist, rdata.id));
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

  UpdateRequest one[1] = { UpdateRequest(src, 0, graph.getData(src,Galois::NONE).id ) };
  T.start();
  Exp::StartWorklistExperiment<OBIM,dChunk,Chunk,UpdateRequestIndexer,seq_less,seq_greater>()(std::cout, &one[0], &one[1], process());
  T.stop();
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
  Exp::parse_worklist_command_line(args);

  Galois::Statistic<unsigned int> sBadWork("BadWork");
  Galois::Statistic<unsigned int> sWLEmptyWork("WLEmptyWork");
  BadWork = &sBadWork;
  WLEmptyWork = &sWLEmptyWork;

  for (std::vector<const char*>::iterator ii = args.begin(), ei = args.end(); ii != ei; ++ii) {
    if (strcmp(*ii, "-delta") == 0 && ii + 1 != ei) {
      stepShift = atoi(ii[1]);
      ii = args.erase(ii);
      ii = args.erase(ii);
      --ii;
      ei = args.end();
    } else if (strcmp(*ii, "-bfs") == 0) {
      do_bfs = true;
      ii = args.erase(ii);
      --ii;
      ei = args.end();
    }
  }
  
  if (args.size() < 3) {
    std::cerr << "not enough arguments, use -help for usage information\n";
    return 1;
  }

  printBanner(std::cout, name, description, url);
  
  const char* inputfile = args[0];
  unsigned int startNode = atoi(args[1]);
  unsigned int reportNode = atoi(args[2]);

  GNode source = -1;
  GNode report = -1;
  
  graph.structureFromFile(inputfile);
  graph.emptyNodeData();
  std::cout << "Read " << graph.size() << " nodes\n";
  std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
  
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

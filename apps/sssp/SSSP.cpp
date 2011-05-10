/*
 * SSSP.cpp
 *
 *  Created on: Oct 18, 2010
 *      Author: amin, reza
 */

#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/IO/gr.h"

#include "Galois/Graphs/Serialize.h"
#include "Galois/Graphs/FileGraph.h"

#include "Galois/Runtime/DebugWorkList.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
//#include <fstream>
#include <set>
using namespace std;

static const char* name = "Single Source Shortest Path";
static const char* description = "Computes the shortest path from a source node to all nodes in a directed graph using a modified Bellman-Ford algorithm\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/sssp.html";
static const char* help = "<input file> <startnode> <reportnode> [-delta <deltaShift>]";

static const unsigned int DIST_INFINITY = std::numeric_limits<unsigned int>::max() - 1;

static unsigned int stepShift = 10;

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
    if (lhs.w < rhs.w) 
      return true;
    if (lhs.w > rhs.w)
      return false;
    return lhs.n < rhs.n;
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

template<typename WLTy>
void getInitialRequests(const GNode src, WLTy& wl) {
  for (Graph::neighbor_iterator ii = graph.neighbor_begin(src, Galois::Graph::NONE), 
	 ee = graph.neighbor_end(src, Galois::Graph::NONE); 
       ii != ee; ++ii) {
    GNode dst = *ii;
    int w = graph.getEdgeData(src, dst, Galois::Graph::NONE);
    UpdateRequest up(dst, w);
    wl.push(up);
  }
}

void runBody(const GNode src) {
  std::set<UpdateRequest, seq_less> initial;
  for (Graph::neighbor_iterator ii = graph.neighbor_begin(src, Galois::Graph::NONE), 
        ee = graph.neighbor_end(src, Galois::Graph::NONE); 
       ii != ee; ++ii) {
    GNode dst = *ii;
    int w = graph.getEdgeData(src, dst, Galois::Graph::NONE);
    UpdateRequest up(dst, w);
    initial.insert(up);
  }

  unsigned int counter = 0;

  Galois::Launcher::startTiming();
  
  while (!initial.empty()) {
    ++counter;
    UpdateRequest req = *initial.begin();
    initial.erase(initial.begin());
    SNode& data = graph.getData(req.n, Galois::Graph::NONE);

    if (req.w < data.dist) {
      data.dist = req.w;
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req.n, Galois::Graph::NONE), 
	     ee = graph.neighbor_end(req.n, Galois::Graph::NONE);
	   ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph.getEdgeData(req.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req.w + d;
	if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	  initial.insert(UpdateRequest(dst, newDist));
      }
    }
  }

  Galois::Launcher::stopTiming();
  GaloisRuntime::reportStat("Iterations: ", counter);
}

struct process {
  template<typename ContextTy>
  void __attribute__((noinline)) operator()(UpdateRequest& req, ContextTy& lwl) {
    SNode& data = graph.getData(req.n,Galois::Graph::NONE);
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	for (Graph::neighbor_iterator ii = graph.neighbor_begin(req.n, Galois::Graph::NONE), ee = graph.neighbor_end(req.n, Galois::Graph::NONE); ii != ee; ++ii) {
	  GNode dst = *ii;
	  int d = graph.getEdgeData(req.n, dst, Galois::Graph::NONE);
	  unsigned int newDist = req.w + d;
	  if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	    lwl.push(UpdateRequest(dst, newDist));
	}
	break;
      }
    }
  }
};

void runBodyParallel(const GNode src) {
  using namespace GaloisRuntime::WorkList;

  typedef dChunkedLIFO<UpdateRequest, 16> IChunk;
  typedef OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer, IChunk> OBIM;

  OBIM wl;

  getInitialRequests(src, wl);
  Galois::Launcher::startTiming();
  Galois::for_each(wl, process());
  Galois::Launcher::stopTiming();
}


bool verify(GNode source) {
  if (graph.getData(source,Galois::Graph::NONE).dist != 0) {
    cerr << "source has non-zero dist value" << endl;
    return false;
  }
  
  for (Graph::active_iterator src = graph.active_begin(), ee =
	 graph.active_end(); src != ee; ++src) {
    unsigned int dist = graph.getData(*src,Galois::Graph::NONE).dist;
    if (dist >= DIST_INFINITY) {
      cerr << "found node = " << graph.getData(*src,Galois::Graph::NONE).id
	   << " with label >= INFINITY = " << dist << endl;
      return false;
    }
    
    for (Graph::neighbor_iterator ii = graph.neighbor_begin(*src, Galois::Graph::NONE), ee =
	   graph.neighbor_end(*src, Galois::Graph::NONE); ii != ee; ++ii) {
      GNode neighbor = *ii;
      unsigned int ddist = graph.getData(*src,Galois::Graph::NONE).dist;
      
      if (ddist > dist + graph.getEdgeData(*src, neighbor, Galois::Graph::NONE)) {
	cerr << "bad level value at " << graph.getData(*src,Galois::Graph::NONE).id
	     << " which is a neighbor of " << graph.getData(neighbor,Galois::Graph::NONE).id << endl;
	return false;
      }
      
    }
  }
  return true;
}

int main(int argc, const char **argv) {

  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 3) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);
  
  const char* inputfile = args[0];
  unsigned int startNode = atoi(args[1]);
  unsigned int reportNode = atoi(args[2]);
  if (args.size() >= 5 && strcmp(args[3], "-delta") == 0)
    stepShift = atoi(args[4]);
 
  GNode source = -1;
  GNode report = -1;
  
  graph.structureFromFile(inputfile);
  graph.emptyNodeData();
  std::cout << "Read " << graph.size() << " nodes\n";
  std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
  
  unsigned int id = 0;
  for (Graph::active_iterator src = graph.active_begin(), ee =
      graph.active_end(); src != ee; ++src) {
    SNode& node = graph.getData(*src,Galois::Graph::NONE);
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
  if (report == -1) {
    std::cerr << "Failed to set report (" << reportNode << ").\n";
    assert(0);
    abort();
  }
  if (source == -1) {
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

  GaloisRuntime::reportStat("Time", Galois::Launcher::elapsedTime());
  cout << report << " " 
       << graph.getData(report,Galois::Graph::NONE).toString() << endl;
  if (!skipVerify && !verify(source)) {
    cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}
// vim sw=2:ts=8:sts=2

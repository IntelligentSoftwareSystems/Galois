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

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
using namespace std;

static const char* name = "Single Source Shortest Path";
static const char* description = "Computes the shortest path from a source node to all nodes in a directed graph using a modified Bellman-Ford algorithm\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/sssp.html";
static const char* help = "<input file> <startnode> <reportnode> [-bfs]";

static const unsigned int DIST_INFINITY = std::numeric_limits<unsigned int>::max() - 1;

struct SNode {
  unsigned int id;
  unsigned int dist;
  
  SNode(int _id) : id(_id), dist(DIST_INFINITY) {}
  std::string toString() {
    std::ostringstream s;
    s << '[' << id << "] dist: " << dist;
    return s.str();
  }
};

typedef Galois::Graph::FirstGraph<SNode, unsigned int, true> Graph;
typedef Galois::Graph::FirstGraph<SNode, unsigned int, true>::GraphNode GNode;

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

struct UpdateRequestIndexer
  : std::binary_function<UpdateRequest, unsigned int, unsigned int> {
  unsigned int operator() (const UpdateRequest& val, unsigned int size) const {
    unsigned int ret = val.w / 700;
    return std::min(ret, size - 1);
  }
};

template<typename WLTy>
void getInitialRequests(const GNode src, Graph& graph, WLTy& wl) {
  for (Graph::neighbor_iterator ii = graph.neighbor_begin(src, Galois::Graph::NONE), 
	 ee = graph.neighbor_end(src, Galois::Graph::NONE); 
       ii != ee; ++ii) {
    GNode dst = *ii;
    int w = graph.getEdgeData(src, dst, Galois::Graph::NONE);
    UpdateRequest up(dst, w);
    wl.push(up);
  }
}

void runBody(const GNode src, Graph& graph) {
  priority_queue<UpdateRequest> initial;
  getInitialRequests(src, graph, initial);
  
  while (!initial.empty()) {
    UpdateRequest req = initial.top();
    initial.pop();
    SNode& data = req.n.getData(Galois::Graph::NONE);
    //    std::cerr << data.id << " " << data.dist << " " << req.w << "\n";
    if (req.w < data.dist) {
      data.dist = req.w;
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req.n, Galois::Graph::NONE), 
	     ee = graph.neighbor_end(req.n, Galois::Graph::NONE);
	   ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph.getEdgeData(req.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req.w + d;
	if (newDist < dst.getData(Galois::Graph::NONE).dist)
	  initial.push(UpdateRequest(dst, newDist));
      }
    }
  }
}
  
void process(UpdateRequest& req, Galois::Context<UpdateRequest>& lwl) {
  SNode& data = req.n.getData(Galois::Graph::NONE);
  Graph* graph = req.n.getGraph();
  unsigned int v;
  //  std::cerr << data.id << " " << data.dist << " " << req.w << "\n";
  while (req.w < (v = data.dist)) {
    if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
      for (Graph::neighbor_iterator ii = graph->neighbor_begin(req.n, Galois::Graph::NONE), ee = graph->neighbor_end(req.n, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph->getEdgeData(req.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req.w + d;
	if (newDist < dst.getData(Galois::Graph::NONE).dist)
	  lwl.push(UpdateRequest(dst, newDist));
      }
      break;
    }
  }
}
 
void runBodyParallel(const GNode src, unsigned int numNodes) {
  //GaloisRuntime::WorkList::PriQueue<UpdateRequest> wl;
  typedef GaloisRuntime::WorkList::OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer> OBIM;
  OBIM wl(30*1024);
  //  GaloisRuntime::WorkList::CacheByIntegerMetric<OBIM, 1, UpdateRequestIndexer> wl2(wl);
  
  getInitialRequests(src, *src.getGraph(), wl);
  Galois::for_each(wl, process);
}


bool verify(GNode source, Graph& graph) {
  if (source.getData(Galois::Graph::NONE).dist != 0) {
    cerr << "source has non-zero dist value" << endl;
    return false;
  }
  
  for (Graph::active_iterator src = graph.active_begin(), ee =
	 graph.active_end(); src != ee; ++src) {
    unsigned int dist = src->getData(Galois::Graph::NONE).dist;
    if (dist >= DIST_INFINITY) {
      cerr << "found node = " << src->getData(Galois::Graph::NONE).id
	   << " with label >= INFINITY = " << dist << endl;
      return false;
    }
    
    for (Graph::neighbor_iterator ii = graph.neighbor_begin(*src, Galois::Graph::NONE), ee =
	   graph.neighbor_end(*src, Galois::Graph::NONE); ii != ee; ++ii) {
      GNode neighbor = *ii;
      unsigned int ddist = src->getData(Galois::Graph::NONE).dist;
      
      if (ddist > dist + graph.getEdgeData(*src, neighbor, Galois::Graph::NONE)) {
	cerr << "bad level value at " << src->getData(Galois::Graph::NONE).id
	     << " which is a neighbor of " << neighbor.getData(Galois::Graph::NONE).id << endl;
	return false;
      }
      
    }
  }
  return true;
}

int main(int argc, const char **argv) {

  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 4) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);
  
  const char* inputfile = args[1];
  unsigned int startNode = atoi(args[2]);
  unsigned int reportNode = atoi(args[3]);
  bool bfs = args.size() == 5 && strcmp(args[4], "-bfs") == 0;

  Graph graph;
  GNode source;
  GNode sink;

  std::pair<unsigned int, unsigned int> r;
  if (bfs)
    r = Galois::IO::readFile_gr_unit(inputfile, &graph);
  else
    r = Galois::IO::readFile_gr(inputfile, &graph);
  unsigned int numNodes = r.first;
  unsigned int numEdges = r.second;
  std::cout << "Read " << numNodes << " nodes and " << numEdges << " edges.\n"
	    << "Starting at " << startNode << " and reporting at " << reportNode << "\n";

  if (startNode > numNodes) {
    std::cerr << "Invalid start node\n";
    assert(0);
    abort();
  }
  if (reportNode >numNodes) {
    std::cerr << "Invalid report node\n";
    assert(0);
    abort();
  }
  
  for (Graph::active_iterator src = graph.active_begin(), ee =
	 graph.active_end(); src != ee; ++src) {
    SNode& node = src->getData(Galois::Graph::NONE);
    node.dist = DIST_INFINITY;
    //std::cout << node.toString() << "\n";

    if (node.id == startNode) {
      source = *src;
      node.dist = 0;
    } else if (node.id == reportNode) {
      sink = *src;
    }
  }
  if (sink.isNull()) {
    std::cerr << "Failed to set sink (" << reportNode << ").\n";
    assert(0);
    abort();
  }
  if (source.isNull()) {
    std::cerr << "Failed to set source (" << startNode << ".\n";
    assert(0);
    abort();
  }

  if (numThreads == 0) {
    std::cout << "Running Sequentially\n";
    Galois::Launcher::startTiming();
    runBody(source, graph);
    Galois::Launcher::stopTiming();
  } else {
    Galois::setMaxThreads(numThreads);
    Galois::Launcher::startTiming();
    runBodyParallel(source, numNodes);
    Galois::Launcher::stopTiming();
  }

  cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";
  cout << sink.getData(Galois::Graph::NONE).toString() << endl;
  if (!skipVerify && !verify(source, graph)) {
    cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}

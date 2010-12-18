/*
 * SSSP.cpp
 *
 *  Created on: Oct 18, 2010
 *      Author: amin, reza
 */

#include "SSSP.h"
#include "Galois/IO/gr.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <iostream>
#include <fstream>
using namespace std;

static const char* name = "Single Source Shortest Path";
static const char* description = "Computes the shortest path from a source node to all nodes in a directed graph using a modified Bellman-Ford algorithmRefines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/sssp.html";
static const char* help = "<input file> <bfs> <startnode>";

int main(int argc, const char **argv) {

  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() != 4) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);
  
  const char* inputfile = args[0];
  bool bfs = strcmp(args[1], "f") == 0 ? false : true;
  int maxNodes = atoi(args[2]);
  
  SSSP sssp;
  sssp.run(bfs, inputfile, numThreads, maxNodes);
  return 0;
}

void SSSP::updateSourceAndSink(const int sourceId, const int sinkId) {
	if (sourceId > numNodes || sourceId <= 0 || sinkId > numNodes || sinkId <= 0) {
		cerr<<"Invalid maxNode!" <<endl;
		exit(-1);
	}
	for (Graph::active_iterator src = graph.active_begin(), ee =
			graph.active_end(); src != ee; ++src) {
		SNode& node = src->getData(Galois::Graph::NONE);
		node.dist = DIST_INFINITY;
		//std::cout << node.toString() << "\n";

		if (node.id == sourceId) {
			source = *src;
			node.dist = 0;
		} else if (node.id == sinkId) {
			sink = *src;
		}
	}
	if (sink.isNull()) {
	  std::cerr << "Failed to set sink (" << sinkId << ").\n";
	  assert(0);
	  abort();
	}
	if (source.isNull()) {
	  std::cerr << "Failed to set source (" << sourceId << ".\n";
	  assert(0);
	  abort();
	}
}

int SSSP::getEdgeData(GNode src, GNode dst) {
  int retval;
  if (executorType.bfs)
    retval = 1;
  else
    retval = graph.getEdgeData(src, dst, Galois::Graph::NONE);
  assert(retval >= 0);
  return retval;
}

void SSSP::initializeGraph(const char *filename) {
  Galois::IO::readFile_gr<Graph, true>(filename, &graph);
}

void SSSP::run(bool bfs, const char *filename, int threadnum, int maxNodes) {
	executorType = ExecutorType(bfs);
	initializeGraph(filename);
	updateSourceAndSink(maxNodes, numNodes - 1 - maxNodes); //FIXME:!!?

	if (threadnum == 0) {
		Galois::Launcher::startTiming();
		runBody(source);
		Galois::Launcher::stopTiming();
	} else {
		Galois::setMaxThreads(threadnum);
		Galois::Launcher::startTiming();
		runBodyParallel(source);
		Galois::Launcher::stopTiming();
	}
	cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";
	cout << this->sink.getData(Galois::Graph::NONE).toString() << endl;
	if (!verify()) {
		cerr << "Verification failed.\n";
		assert(0 && "Verification failed");
		abort();
	}
}

SSSP *sssp;
void process(UpdateRequest& req, Galois::Context<UpdateRequest>& lwl) {
  SNode& data = req.n.getData(Galois::Graph::NONE);
  int v;
  while (req.w < (v = data.dist)) {
    if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
      for (Graph::neighbor_iterator ii = sssp->graph.neighbor_begin(req.n, Galois::Graph::NONE), ee =
	     sssp->graph.neighbor_end(req.n, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode dst = *ii;
	int d = sssp->getEdgeData(req.n, dst);
	int newDist = req.w + d;
	if (newDist < dst.getData(Galois::Graph::NONE).dist)
	  lwl.push(UpdateRequest(dst, newDist));
      }
      break;
    }
  }
}


void SSSP::runBodyParallel(const GNode src) {
  GaloisRuntime::WorkList::PriQueue<UpdateRequest, std::less<UpdateRequest> > wl;
  
  for (Graph::neighbor_iterator ii = graph.neighbor_begin(src, Galois::Graph::NONE), ee =
	 graph.neighbor_end(src, Galois::Graph::NONE); ii != ee; ++ii) {
    GNode dst = *ii;
    int w = getEdgeData(src, dst);
    UpdateRequest up = UpdateRequest(dst, w);
    wl.push(up);
  }
  sssp = this;
  Galois::for_each(wl, process);
}

bool SSSP::verify() {
  if (source.getData(Galois::Graph::NONE).dist != 0) {
    cerr << "source has non-zero dist value" << endl;
    return false;
  }

  for (Graph::active_iterator src = graph.active_begin(), ee =
	 graph.active_end(); src != ee; ++src) {
    const int dist = src->getData(Galois::Graph::NONE).dist;
    if (dist >= DIST_INFINITY) {
      cerr << "found node = " << src->getData(Galois::Graph::NONE).id
	   << " with label >= INFINITY = " << dist << endl;
      return false;
    }

    for (Graph::neighbor_iterator ii = graph.neighbor_begin(*src, Galois::Graph::NONE), ee =
	   graph.neighbor_end(*src, Galois::Graph::NONE); ii != ee; ++ii) {
      GNode neighbor = *ii;
      int ddist = src->getData(Galois::Graph::NONE).dist;

      if (ddist > dist + getEdgeData(*src, neighbor)) {
	cerr << "bad level value at " << src->getData(Galois::Graph::NONE).id
	     << " which is a neighbor of " << neighbor.getData(Galois::Graph::NONE).id << endl;
	return false;
      }

    }
  }
  return true;
}

void SSSP::runBody(const GNode src) {
  priority_queue<UpdateRequest> initial;
  for (Graph::neighbor_iterator ii = graph.neighbor_begin(src, Galois::Graph::NONE), 
	 ee = graph.neighbor_end(src, Galois::Graph::NONE); 
       ii != ee; ++ii) {
    GNode dst = *ii;
    int w = getEdgeData(src, dst);
    UpdateRequest up(dst, w);
    initial.push(up);
  }

  while (!initial.empty()) {
    UpdateRequest req = initial.top();
    initial.pop();
    SNode& data = req.n.getData(Galois::Graph::NONE);
    //    std::cout << data.toString() << " -> " << req.w << " (" << initial.size() << ")\n";
    if (req.w < data.dist) {
      data.dist = req.w;
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req.n, Galois::Graph::NONE), 
	     ee = graph.neighbor_end(req.n, Galois::Graph::NONE);
	   ii != ee; ++ii) {
	GNode dst = *ii;
	int d = getEdgeData(req.n, dst);
	int newDist = req.w + d;
	if (dst.getData(Galois::Graph::NONE).dist > newDist)
	  initial.push(UpdateRequest(dst, newDist));
      }
    }
  }
}

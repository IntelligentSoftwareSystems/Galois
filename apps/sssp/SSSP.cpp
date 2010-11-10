/*
 * SSSP.cpp
 *
 *  Created on: Oct 18, 2010
 *      Author: amin, reza
 */

#include <list>
#include "SSSP.h"

void SSSP::updateSourceAndSink(const int sourceId, const int sinkId) {
	if (sourceId > numNodes || sourceId <= 0 || sinkId > numNodes || sinkId <= 0) {
		cerr<<"Invalid maxNode!" <<endl;
		exit(-1);
	}
	for (Graph::active_iterator src = graph->active_begin(), ee =
			graph->active_end(); src != ee; ++src) {
		SNode& node = src->getData(Galois::Graph::NONE);
		node.dist = DIST_INFINITY;
		if (node.id == sourceId) {
			source = *src;
			node.dist = 0;
		} else if (node.id == sinkId) {
			sink = *src;
		}
	}
}

int SSSP::getEdgeData(GNode src, GNode dst) {
	if (executorType.bfs)
		return 1;
	else
	  return graph->getEdgeData(src, dst, Galois::Graph::NONE);
}

void SSSP::initializeGraph(char *filename) {
  ifstream infile;
  infile.open(filename, ifstream::in); // opens the vector file
  if (!infile) { // file couldn't be opened
    cerr << "Error: vector file could not be opened" << endl;
    exit(-1);
  }
  
  string name;
  int LEdgeCount = 0;
  GNode *gnodes = NULL;
  while (!infile.eof()) {
    string firstchar;
    infile >> firstchar;
    if (firstchar == "c") {
     string line;
     getline(infile, line);
    } else if (firstchar == "p") {
      infile >> name;
      infile >> numNodes;
      infile >> numEdges;
      graph = new Graph();
      gnodes = new GNode[numNodes];
      for (int i = 0; i < numNodes; i++) {
	gnodes[i] = graph->createNode(SNode(i+1));
	graph->addNode(gnodes[i], Galois::Graph::NONE);
      }
      //			cout << "graph name is " << name << " and it has " << numNodes
      //					<< " nodes and some edges" << endl;
    } else if (firstchar == "a") {
      ++LEdgeCount;
      int src, dest, weight;
      infile >> src >> dest >> weight;
      graph->addEdge(gnodes[src - 1], gnodes[dest - 1], weight, Galois::Graph::NONE);
      //			cout << "node: " << src << " " << dest << " " << weight << endl;
    }
  }
  assert(LEdgeCount == numEdges);
  delete[] gnodes;
  this->delta = 700;
  
  infile.close();
}

void SSSP::run(bool bfs, char *filename, int threadnum, int maxNodes) {
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
void process(UpdateRequest& req, Galois::WorkList<UpdateRequest>& lwl) {
  SNode& data = req.n.getData(Galois::Graph::NONE);
  int v;
  while (req.w < (v = data.dist)) {
    if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
      for (Graph::neighbor_iterator ii = sssp->graph->neighbor_begin(req.n, Galois::Graph::NONE), ee =
	     sssp->graph->neighbor_end(req.n, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode dst = *ii;
	int d = sssp->getEdgeData(req.n, dst);
	int newDist = req.w + d;
	lwl.push(UpdateRequest(dst, newDist, d <= sssp->delta));
      }
      break;
    }
  }
}


template<typename T>
void SSSP::runBodyParallel(const GNode src, T& wl) {
  for (Graph::neighbor_iterator ii = graph->neighbor_begin(src, Galois::Graph::NONE), ee =
	 graph->neighbor_end(src, Galois::Graph::NONE); ii != ee; ++ii) {
    GNode dst = *ii;
    int w = getEdgeData(src, dst);
    UpdateRequest up = UpdateRequest(dst, w, w <= delta);
    wl.push(up);
  }
  sssp = this;
  Galois::for_each(wl, process);
}

void SSSP::runBodyParallel(const GNode src) {
	if (executorType.bfs) {
	  threadsafe::ts_queue<UpdateRequest> wl;
		runBodyParallel(src,wl);
	} else {
	  threadsafe::ts_pqueue<UpdateRequest, std::greater<UpdateRequest> > wl;
		runBodyParallel(src,wl);
	}
}

bool SSSP::verify() {
	if (source.getData(Galois::Graph::NONE).dist != 0) {
		cerr << "source has non-zero dist value" << endl;
		return false;
	}

	for (Graph::active_iterator src = graph->active_begin(), ee =
			graph->active_end(); src != ee; ++src) {
		const int dist = src->getData(Galois::Graph::NONE).dist;
		if (dist >= DIST_INFINITY) {
			cerr << "found node = " << src->getData(Galois::Graph::NONE).id
					<< " with label >= INFINITY = " << dist << endl;
			return false;
		}

		for (Graph::neighbor_iterator ii = graph->neighbor_begin(*src, Galois::Graph::NONE), ee =
				graph->neighbor_end(*src, Galois::Graph::NONE); ii != ee; ++ii) {
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
  priority_queue<UpdateRequest, vector<UpdateRequest> , std::greater<UpdateRequest> > initial;
	//	queue<UpdateRequest *> initial;
	for (Graph::neighbor_iterator ii = graph->neighbor_begin(src, Galois::Graph::NONE), ee =
			graph->neighbor_end(src, Galois::Graph::NONE); ii != ee; ++ii) {
		GNode dst = *ii;
		int w = getEdgeData(src, dst);
		UpdateRequest up(dst, w, w <= delta);
		initial.push(up);
	}

	while (!initial.empty()) {
		UpdateRequest req = initial.top();
		initial.pop();
		SNode& data = req.n.getData(Galois::Graph::NONE);
		int v;
		while (req.w < (v = data.dist)) {
			if (__sync_bool_compare_and_swap(&data.dist, v, req.w) == true) {
				for (Graph::neighbor_iterator ii = graph->neighbor_begin(req.n, Galois::Graph::NONE), ee =
						graph->neighbor_end(req.n, Galois::Graph::NONE); ii != ee; ++ii) {
					GNode dst = *ii;
					int d = getEdgeData(req.n, dst);
					int newDist = req.w + d;
					initial.push(UpdateRequest(dst, newDist, d <= delta));
				}
				break;
			}
		}
	}

}

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

#include "Galois/Runtime/DistributedWorkList.h"
#include "Galois/Runtime/DebugWorkList.h"

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
  
  SNode(int _id = -1) : id(_id), dist(DIST_INFINITY) {}
  std::string toString() {
    std::ostringstream s;
    s << '[' << id << "] dist: " << dist;
    return s.str();
  }
};

//typedef Galois::Graph::FirstGraph<SNode, unsigned int, true> Graph;
//typedef Galois::Graph::FirstGraph<SNode, unsigned int, true>::GraphNode GNode;
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

struct UpdateRequestIndexer
  : std::binary_function<UpdateRequest, unsigned int, unsigned int> {
  unsigned int operator() (const UpdateRequest& val) const {
    return val.w / 700;
  }
  unsigned int operator() (const unsigned int val) const {
    return val / 700;
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
  priority_queue<UpdateRequest> initial;
  getInitialRequests(src, initial);
  
  while (!initial.empty()) {
    UpdateRequest req = initial.top();
    initial.pop();
    SNode& data = graph.getData(req.n, Galois::Graph::NONE);
    //    std::cerr << data.id << " " << data.dist << " " << req.w << "\n";
    if (req.w < data.dist) {
      data.dist = req.w;
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req.n, Galois::Graph::NONE), 
	     ee = graph.neighbor_end(req.n, Galois::Graph::NONE);
	   ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph.getEdgeData(req.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req.w + d;
	if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	  initial.push(UpdateRequest(dst, newDist));
      }
    }
  }
}

struct process {
  template<typename ContextTy>
  void __attribute__((noinline)) operator()(UpdateRequest& req, ContextTy& lwl) {
    SNode& data = graph.getData(req.n,Galois::Graph::NONE);
    unsigned int v;
    //  std::cerr << data.id << " " << data.dist << " " << req.w << "\n";
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

struct processBatch {
  template<typename ContextTy, typename QT>
  void __attribute__((noinline)) supermagic4(UpdateRequest& req1, UpdateRequest& req2, UpdateRequest& req3, UpdateRequest& req4, ContextTy& lwl, QT& Q, unsigned int Current ) {
    UpdateRequestIndexer I;

    SNode& data1 = graph.getData(req1.n,Galois::Graph::NONE);
    SNode& data2 = graph.getData(req2.n,Galois::Graph::NONE);
    SNode& data3 = graph.getData(req3.n,Galois::Graph::NONE);
    SNode& data4 = graph.getData(req4.n,Galois::Graph::NONE);
    unsigned int v1,v2,v3,v4;
    bool b1,b2,b3,b4;
    bool d1,d2,d3,d4;
    d1 = d2 = d3 = d4 = false;
    do {
      b1 = req1.w < (v1 = data1.dist);
      b2 = req2.w < (v2 = data2.dist);
      b3 = req3.w < (v3 = data3.dist);
      b4 = req4.w < (v4 = data4.dist);
      if (b1)
	d1 |= __sync_bool_compare_and_swap(&data1.dist, v1, req1.w);
      if (b2)
	d2 |= __sync_bool_compare_and_swap(&data2.dist, v2, req2.w);
      if (b3)
	d3 |= __sync_bool_compare_and_swap(&data3.dist, v3, req3.w);
      if (b4)
	d4 |= __sync_bool_compare_and_swap(&data4.dist, v4, req4.w);
      if (!b1 && !b2 && !b3 && !b4)
	break;
    } while (true);

    if (d1) {
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req1.n, Galois::Graph::NONE), ee = graph.neighbor_end(req1.n, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph.getEdgeData(req1.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req1.w + d;
	if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	  if (I(newDist) <= Current)
	    Q.push(UpdateRequest(dst, newDist)); //Local
	  else
	    lwl.push(UpdateRequest(dst, newDist)); //Global
      }
    }
   if (d2) {
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req2.n, Galois::Graph::NONE), ee = graph.neighbor_end(req2.n, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph.getEdgeData(req2.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req2.w + d;
	if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	  if (I(newDist) <= Current)
	    Q.push(UpdateRequest(dst, newDist)); //Local
	  else
	    lwl.push(UpdateRequest(dst, newDist)); //Global
      }
    }
   if (d3) {
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req3.n, Galois::Graph::NONE), ee = graph.neighbor_end(req3.n, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph.getEdgeData(req3.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req3.w + d;
	if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	  if (I(newDist) <= Current)
	    Q.push(UpdateRequest(dst, newDist)); //Local
	  else
	    lwl.push(UpdateRequest(dst, newDist)); //Global
      }
    }
   if (d4) {
      for (Graph::neighbor_iterator ii = graph.neighbor_begin(req4.n, Galois::Graph::NONE), ee = graph.neighbor_end(req4.n, Galois::Graph::NONE); ii != ee; ++ii) {
	GNode dst = *ii;
	int d = graph.getEdgeData(req4.n, dst, Galois::Graph::NONE);
	unsigned int newDist = req4.w + d;
	if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	  if (I(newDist) <= Current)
	    Q.push(UpdateRequest(dst, newDist)); //Local
	  else
	    lwl.push(UpdateRequest(dst, newDist)); //Global
      }
    }
  }

  template<typename ContextTy>
  void __attribute__((noinline)) operator()(UpdateRequest& reqIn, ContextTy& lwl) {

    UpdateRequestIndexer I;
    typedef std::deque<UpdateRequest, Galois::PerIterMem::ItAllocTy::rebind<UpdateRequest>::other> DQ;
    std::queue<UpdateRequest, DQ> Q(DQ(lwl.PerIterationAllocator));
    unsigned int Current = I(reqIn);
    Q.push(reqIn);

    while (!Q.empty()) {
      if (Q.size() >= 4) {
	//	std::cerr << '*';
	UpdateRequest r1 = Q.front();
	Q.pop();
	UpdateRequest r2 = Q.front();
	Q.pop();
	UpdateRequest r3 = Q.front();
	Q.pop();
	UpdateRequest r4 = Q.front();
	Q.pop();
	supermagic4(r1,r2,r3,r4, lwl, Q, graph, Current);
      } else {
	UpdateRequest req = Q.front();
	Q.pop();
	
	SNode& data = graph.getData(req.n,Galois::Graph::NONE);
	unsigned int v;
	while (req.w < (v = data.dist)) {
	  if (__sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	    for (Graph::neighbor_iterator ii = graph.neighbor_begin(req.n, Galois::Graph::NONE), ee = graph.neighbor_end(req.n, Galois::Graph::NONE); ii != ee; ++ii) {
	      GNode dst = *ii;
	      int d = graph.getEdgeData(req.n, dst, Galois::Graph::NONE);
	      unsigned int newDist = req.w + d;
	      if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
		if (I(newDist) <= Current)
		  Q.push(UpdateRequest(dst, newDist)); //Local
		else
		  lwl.push(UpdateRequest(dst, newDist)); //Global
	    }
	    break;
	  }
	}
      }
    }
  }
};

struct processSMT {
  UpdateRequestIndexer I;
  typedef std::deque<UpdateRequest, Galois::PerIterMem::ItAllocTy::rebind<UpdateRequest>::other> DQ;

  template<typename ContextTy>
  struct Instance {
    UpdateRequest req;
    SNode* data;
    ContextTy* lwl;
    unsigned int v;
    Graph::neighbor_iterator ii;
    Graph::neighbor_iterator ee;
    GNode dst;
    unsigned int newDist;
    DQ* Q;
    unsigned int Current;
    bool done;
    UpdateRequestIndexer I;

    int state;

    Instance() :Q(0), done(false), state(0) {}

    void initialize(DQ* q, unsigned int C, ContextTy* L) {
      Q = q;
      Current = C;
      lwl = L;
    }

    bool doM() {
      switch(state) {
      case 0:
	do0();
	break;
      case 1:
	do1();
	break;
      case 2:
	do2();
	break;
      case 3:
	do3();
	break;
      case 4:
	do4();
	break;
      case 5:
	do5();
	break;
      case 6:
	do6();
	break;
      case 7:
	do7();
	break;
      case 8:
	do8();
	break;
      case 9:
	do9();
	break;
      case 10:
	do10();
	break;
      case 11:
	do11();
	break;
      case 12:
	do12();
	break;
      default:
	assert(0);
      }
      return done;
    }

    void do0() {
      if (!Q->empty()) {
	req = Q->front();
	Q->pop_front();
	++state;
	done = false;
      } else {
	done = true;
      }
    }

    void do1() {
      data = &graph.getData(req.n,Galois::Graph::NONE);
      ++state;
    }
    void do2() {
      ++state;
    }
    void do3() {
      if (req.w < (v = data->dist))
	++state;
      else
	state = 0; //done
    }
    void do4() {
      if (__sync_bool_compare_and_swap(&data->dist, v, req.w))
	++state;
      else
	state = 3; // while entrance
    }
    void do5() {
      ii = graph.neighbor_begin(req.n, Galois::Graph::NONE);
      ++state;
    }
    void do6() {
      ee = graph.neighbor_end(req.n, Galois::Graph::NONE);
      ++state;
    }
    void do7() {
      if (ii != ee)
	++state;
      else
	state = 0; //done
    }
    void do8() {
      dst = *ii;
      ++state;
    }
    void do9() {
      newDist = req.w + graph.getEdgeData(req.n, dst, Galois::Graph::NONE);
      ++state;
    }
    void do10() {
      if (newDist < graph.getData(dst,Galois::Graph::NONE).dist)
	++state;
      else
	state += 2; //skip push
    }
    void do11() {
      if (I(newDist) <= Current)
	Q->push_back(UpdateRequest(dst, newDist));
      else
	lwl->push(UpdateRequest(dst, newDist));
      ++state;
    }
    void do12() {
      ++ii;
      state = 7; //loop header ii!=ee
    }
  };

  template<typename ContextTy>
  void __attribute__((noinline)) operator()(UpdateRequest& req, ContextTy& lwl) {

    DQ Q(lwl.PerIterationAllocator);
    unsigned int C = I(req);

    Instance<ContextTy> Inst[4];
    for (int i = 0; i < 4; ++i)
      Inst[i].initialize(&Q, C, &lwl);

    Q.push_back(req);
    
    bool c = true;
    while (c) {
      c = false;
      for (int i = 0; i < 4; ++i)
	c |= Inst[i].doM();
    }
  }
};

void runBodyParallel(const GNode src) {
  using namespace GaloisRuntime::WorkList;

  //GaloisRuntime::WorkList::PriQueue<UpdateRequest> wl;
  //  typedef GaloisRuntime::WorkList::OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer, GaloisRuntime::WorkList::ChunkedFIFO<UpdateRequest, 8, true, GaloisRuntime::WorkList::FIFO<UpdateRequest> > > OBIM;

  //typedef GaloisRuntime::WorkList::LocalQueues<UpdateRequest, GaloisRuntime::WorkList::PriQueue<UpdateRequest>, GaloisRuntime::WorkList::PriQueue<UpdateRequest> > OBIM;
  //  OBIM wl;

  ApproxOrderByIntegerMetric<UpdateRequest, UpdateRequestIndexer, ChunkedFIFO<UpdateRequest, 32> > wl;
  //OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer, ChunkedFIFO<UpdateRequest, 32> > wl;
  //WorkListTracker<UpdateRequest, UpdateRequestIndexer, ApproxOrderByIntegerMetric<UpdateRequest, UpdateRequestIndexer, ChunkedFIFO<UpdateRequest, 32> > > wl;
  //WorkListTracker<UpdateRequest, UpdateRequestIndexer, OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer, ChunkedFIFO<UpdateRequest, 32> > > wl;
  
  //WorkListTracker<UpdateRequest, UpdateRequestIndexer, OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer, ChunkedFIFO<UpdateRequest, 256> > > wl;


  //typedef GaloisRuntime::WorkList::FIFO<UpdateRequest> OBIM;
  //OBIM wl;

  //typedef GaloisRuntime::WorkList::OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer> OBIM;
  //  typedef GaloisRuntime::WorkList::OrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer, GaloisRuntime::WorkList::StealingLocalWL<UpdateRequest> > OBIM;
  //  OBIM wl(30*1024);
 
  //  GaloisRuntime::WorkList::CacheByIntegerMetric<OBIM, 1, UpdateRequestIndexer> wl2(wl);

  //  typedef GaloisRuntime::WorkList::AdaptiveOrderedByIntegerMetric<UpdateRequest, UpdateRequestIndexer> AOBIM;
  //  AOBIM wl;

  //ReductionWL<UpdateRequest, ApproxOrderByIntegerMetric<UpdateRequest, UpdateRequestIndexer, ChunkedFIFO<UpdateRequest, 32> >, FaradayPolicy> wl;

  //RequestHirarchy<UpdateRequest, ApproxOrderByIntegerMetric<UpdateRequest, UpdateRequestIndexer>, FIFO<UpdateRequest>, FaradayPolicy> wl;
  
  //StealingLocalWL<UpdateRequest, ApproxOrderByIntegerMetric<UpdateRequest, UpdateRequestIndexer> > wl;

  //WorkListTracker<UpdateRequest, UpdateRequestIndexer, DistApproxOrderByIntegerMetric<UpdateRequest, UpdateRequestIndexer> > wl;

  getInitialRequests(src, wl);
  Galois::for_each(wl, process());
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
  bool bfs = args.size() == 4 && strcmp(args[3], "-bfs") == 0;

  GNode source = -1;
  GNode sink = -1;

  graph.structureFromFile(inputfile);
  graph.emptyNodeData();
  std::cout << "Read " << graph.size() << " nodes\n";

  if (bfs) {
    //do something
  }
  // std::pair<unsigned int, unsigned int> r;
  // r = Galois::IO::readFile_gr(inputfile, &graph);
  // unsigned int numNodes = r.first;
  // unsigned int numEdges = r.second;
  // std::cout << "Read " << numNodes << " nodes and " << numEdges << " edges.\n"
  //  	    << "Starting at " << startNode << " and reporting at " << reportNode << "\n";

  //  if (startNode > numNodes) {
    //    std::cerr << "Invalid start node\n";
    //    assert(0);
    //    abort();
    //  }
  //if (reportNode >numNodes) {
    //  std::cerr << "Invalid report node\n";
  //   assert(0);
  //   abort();
  // }

  // if (outputGraph("test.structure", graph))
  //   std::cerr << "graph written\n";
  // else
  //   std::cerr << "graph write failed\n";

  
  for (Graph::active_iterator src = graph.active_begin(), ee =
	 graph.active_end(); src != ee; ++src) {
    SNode& node = graph.getData(*src,Galois::Graph::NONE);
    node.dist = DIST_INFINITY;
    //std::cout << node.toString() << "\n";

    if (*src == startNode) {
      //if (node.id == startNode) {
      source = *src;
      node.dist = 0;
    } else if (*src == reportNode) {
      //} else if (node.id == reportNode) {
      sink = *src;
    }
  }
  if (sink == -1) {
    std::cerr << "Failed to set sink (" << reportNode << ").\n";
    assert(0);
    abort();
  }
  if (source == -1) {
    std::cerr << "Failed to set source (" << startNode << ".\n";
    assert(0);
    abort();
  }

  if (numThreads == 0) {
    std::cout << "Running Sequentially\n";
    Galois::Launcher::startTiming();
    runBody(source);
    Galois::Launcher::stopTiming();
  } else {
    Galois::setMaxThreads(numThreads);
    Galois::Launcher::startTiming();
    runBodyParallel(source);
    Galois::Launcher::stopTiming();
  }

  GaloisRuntime::reportStat("Time", Galois::Launcher::elapsedTime());
  cout << sink << " " 
       << graph.getData(sink,Galois::Graph::NONE).toString() << endl;
  if (!skipVerify && !verify(source)) {
    cerr << "Verification failed.\n";
    assert(0 && "Verification failed");
    abort();
  }

  return 0;
}

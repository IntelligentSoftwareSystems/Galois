/** Single source shortest paths -*- C++ -*-
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
#include <deque>
#include <set>

namespace cll = llvm::cl;

static const bool trackBadWork = false;

static const char* name = "Single Source Shortest Path";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified chaotic iteration algorithm\n";
static const char* url = "single_source_shortest_path";

enum SSSPAlgo {
  serialStl,
  parallel,
  parallelCas,
  parallelLessDups,
  parallelCasLessDups
};

static cll::opt<unsigned int> startNode("startnode", cll::desc("Node to start search from"), cll::init(1));
static cll::opt<unsigned int> reportNode("reportnode", cll::desc("Node to report distance to"), cll::init(2));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> stepShift("delta", cll::desc("Shift value for the deltastep"), cll::init(10));
static cll::opt<SSSPAlgo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serialStl, "Serial using STL heap"),
      clEnumVal(parallel, "Parallel"),
      clEnumVal(parallelCas, "Parallel with CAS"),
      clEnumVal(parallelLessDups, "Parallel with duplicate prevention"),
      clEnumVal(parallelCasLessDups, "Parallel with duplicate prevention and CAS"),
      clEnumValEnd), cll::init(parallelCasLessDups));

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
    std::set<UpdateRequest, std::less<UpdateRequest> > initial;
    UpdateRequest init(src, 0);
    initial.insert(init);

    Galois::Statistic counter("Iterations");
    
    while (!initial.empty()) {
      counter += 1;
      UpdateRequest req = *initial.begin();
      initial.erase(initial.begin());
      SNode& data = graph.getData(req.n, Galois::NONE);
      if (req.w < data.dist) {
        data.dist = req.w;
	for(Graph::edge_iterator
	      ii = graph.edge_begin(req.n, Galois::NONE), 
	      ee = graph.edge_end(req.n, Galois::NONE);
	    ii != ee; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          unsigned int d = graph.getEdgeData(ii);
          unsigned int newDist = req.w + d;
          if (newDist < graph.getData(dst,Galois::NONE).dist) {
            initial.insert(UpdateRequest(dst, newDist));
	  }
        }
      }
    }
  }
};

static Galois::Statistic* BadWork;
static Galois::Statistic* WLEmptyWork;

template<bool useCas>
struct ParallelAlgo {
  std::string name() const { return useCas ? "parallel with CAS" : "parallel"; }

  void operator()(const GNode src) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<16> dChunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
    std::cout << "Warning: Performance varies considerably due to delta parameter. Do not expect the default to be good for your graph\n";

#ifdef GALOIS_USE_EXP
    Exp::PriAuto<16, UpdateRequestIndexer, OBIM, std::less<UpdateRequest>, std::greater<UpdateRequest> >::for_each(UpdateRequest(src, 0), *this);
#else
    Galois::for_each<OBIM>(UpdateRequest(src, 0), *this);
#endif
  }

  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& ctx) const {
    Galois::MethodFlag flag = useCas ? Galois::NONE : Galois::ALL;
    SNode& data = graph.getData(req.n, flag);

    if (trackBadWork && req.w >= data.dist)
      *WLEmptyWork += 1;
    
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (!useCas || __sync_bool_compare_and_swap(&data.dist, v, req.w)) {
	for (Graph::edge_iterator ii = graph.edge_begin(req.n, flag),
	       ee = graph.edge_end(req.n, flag); ii != ee; ++ii) {
	  GNode dst = graph.getEdgeDst(ii);
	  unsigned int d = graph.getEdgeData(ii);
	  unsigned int newDist = req.w + d;
	  SNode& rdata = graph.getData(dst, Galois::NONE);
	  if (newDist < rdata.dist)
	    ctx.push(UpdateRequest(dst, newDist));
	}
        if (!useCas)
          data.dist = req.w;
	if (trackBadWork && v != DIST_INFINITY)
	   *BadWork += 1;
	break;
      }
    }
  }
};

namespace Galois {
template<>
struct does_not_need_aborts<ParallelAlgo<true> > : public boost::true_type {};
}

template<bool useCas>
struct ParallelLessDupsAlgo {
  std::string name() const {
    return useCas ? "parallel with duplicate detection and CAS" : "parallel with duplicate detection"; 
  }

  void operator()(const GNode src) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<16> dChunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
    std::cout << "WARNING: Performance varies considerably due to delta parameter. Do not expect the default to be good for your graph\n";

    std::deque<UpdateRequest> initial;
    graph.getData(src).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(src), ei = graph.edge_end(src); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& data = graph.getData(dst);
      unsigned int d = graph.getEdgeData(ii);
      if (d < data.dist) {
        initial.push_back(UpdateRequest(dst, d));
        data.dist = d;
      }
    }

#ifdef GALOIS_USE_EXP
    Exp::PriAuto<16, UpdateRequestIndexer, OBIM, std::less<UpdateRequest>, std::greater<UpdateRequest> >::for_each(initial.begin(), initial.end(), *this);
#else
    Galois::for_each<OBIM>(initial.begin(), initial.end(), *this);
#endif
  }

  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& ctx) const {
    Galois::MethodFlag flag = useCas ? Galois::NONE : Galois::ALL;
    SNode& data = graph.getData(req.n, flag);

    if (trackBadWork && req.w > data.dist) {
      *WLEmptyWork += 1;
      return;
    }
    
    for (Graph::edge_iterator ii = graph.edge_begin(req.n, flag),
           ee = graph.edge_end(req.n, flag); ii != ee; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      unsigned int d = graph.getEdgeData(ii);
      SNode& rdata = graph.getData(dst, Galois::NONE);
      unsigned int newDist = data.dist + d;
      unsigned int v;
      while (newDist < (v = rdata.dist)) {
        if (!useCas || __sync_bool_compare_and_swap(&rdata.dist, v, newDist)) {
          if (!useCas)
            rdata.dist = newDist;
          if (trackBadWork && v != DIST_INFINITY)
             *BadWork += 1;
          ctx.push(UpdateRequest(dst, newDist));
          break;
        }
      }
    }
  }
};

namespace Galois {
template<>
struct does_not_need_aborts<ParallelLessDupsAlgo<true> > : public boost::true_type {};
}

bool verifyConnected() {
  for (Graph::iterator src = graph.begin(), ee = graph.end(); src != ee; ++src) {
    unsigned int dist = graph.getData(*src,Galois::NONE).dist;
    if (dist == DIST_INFINITY) {
      std::cerr << "WARNING: found node = " << graph.getData(*src, Galois::NONE).id
		<< " with label INFINITY\n";
      return false;
    }
  }
  return true;
}

bool verify(GNode source) {
  bool retval = true;
  if (graph.getData(source,Galois::NONE).dist != 0) {
    std::cerr << "ERROR: source has non-zero dist value\n";
    retval = false;
  }
  
  for (Graph::iterator src = graph.begin(), ee = graph.end(); src != ee; ++src) {
    unsigned int dist = graph.getData(*src, Galois::NONE).dist;
    if (dist > DIST_INFINITY) {
      std::cerr << "ERROR: found node = " << graph.getData(*src,Galois::NONE).id
		<< " with label greater than INFINITY ( ? ? )\n";
      retval = false;
    }
    
    if (dist != DIST_INFINITY) { //avoid overflow on dist + d
      for (Graph::edge_iterator 
	     ii = graph.edge_begin(*src, Galois::NONE),
	     ee = graph.edge_end(*src, Galois::NONE); ii != ee; ++ii) {
	GNode neighbor = graph.getEdgeDst(ii);
	unsigned int ddist = graph.getData(*src, Galois::NONE).dist;
	unsigned int d = graph.getEdgeData(ii);
	if (ddist > dist + d) {
	  std::cerr << "ERROR: bad level value at "
		    << graph.getData(*src, Galois::NONE).id
		    << " which is a neighbor of " 
		    << graph.getData(neighbor, Galois::NONE).id << "\n";
	  retval = false;
	}
      }
    }
  }
  return retval;
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
      if (graph.edge_begin(source) == graph.edge_end(source)) {
	std::cerr << "ERROR: source has no neighbors\n";
        assert(0);
	abort();
      }
    } 
    if (node.id == reportNode) {
      foundReport = true;
      report = *src;
    }
  }

  if (!foundReport) {
    std::cerr << "ERROR: failed to set report (" << reportNode << ").\n";
    assert(0);
    abort();
  }

  if (!foundSource) {
    std::cerr << "ERROR: failed to set source (" << startNode << ").\n";
    assert(0);
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
    case parallel: run(ParallelAlgo<false>(), source); break;
    case parallelCas: run(ParallelAlgo<true>(), source); break;
    case parallelLessDups: run(ParallelLessDupsAlgo<false>(), source); break;
    case parallelCasLessDups: run(ParallelLessDupsAlgo<true>(), source); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }

  if (trackBadWork) {
    delete BadWork;
    delete WLEmptyWork;
  }

  std::cout << graph.getData(report,Galois::NONE).toString() << "\n";
  if (!skipVerify) {
    if (!verifyConnected()) {
      std::cerr << "WARNING: graph not fully connected.\n";
    }
    if (!verify(source)) {
      std::cerr << "ERROR: Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }

  return 0;
}

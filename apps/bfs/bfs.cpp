/** Breadth-first search -*- C++ -*-
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
 * Example breadth-first search application for demoing Galois system. For optimized
 * version, use SSSP application with BFS option instead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#ifdef GALOIS_EXP
#include "Galois/PriorityScheduling.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>


static const char* name = "Breadth-first Search Example";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = 0;

//****** Command Line Options ******
enum BFSAlgo {
  serial,
  serialOpt,
  serialSet,
  serialBarrier,
  serialBare,
  galois,
  galoisOpt,
  galoisSet,
  galoisManualBarrier,
  galoisBarrierCas,
  galoisBarrier
};

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(1));
static cll::opt<unsigned int> reportNode("reportnode",
    cll::desc("Node to report distance to"),
    cll::init(2));
static cll::opt<BFSAlgo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(serialOpt, "Serial optimized "),
      clEnumVal(serialSet, "Serial optimized with workset semantics"),
      clEnumVal(serialBarrier, "Serial optimized with workset and barrier"),
      clEnumVal(serialBare, "Serial optimized with bare runtime"),
      clEnumVal(galois, "Galois"),
      clEnumVal(galoisOpt, "Galois optimized"),
      clEnumVal(galoisSet, "Galois optimized with workset semantics"),
      clEnumVal(galoisManualBarrier, "Galois optimized with workset and manual barrier"),
      clEnumVal(galoisBarrierCas, "Galois optimized with workset and barrier but using CAS"),
      clEnumVal(galoisBarrier, "Galois optimized with workset and barrier"),
      clEnumValEnd), cll::init(serial));
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int dist;
};

typedef Galois::Graph::LC_Linear_Graph<SNode, void> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

struct UpdateRequest {
  GNode n;
  unsigned int w;

  UpdateRequest(): w(0) { }
  UpdateRequest(const GNode& N, unsigned int W): n(N), w(W) { }
  bool operator<(const UpdateRequest& o) const { return w < o.w; }
  bool operator>(const UpdateRequest& o) const { return w > o.w; }
  unsigned getID() const { return /* graph.getData(n).id; */ 0; }
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out <<  "(dist: " << n.dist << ")";
  return out;
}

struct UpdateRequestIndexer {
  unsigned int operator()(const UpdateRequest& val) const {
    unsigned int t = val.w;
    return t;
  }
};

struct GNodeIndexer {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, Galois::NONE).dist;
  }
};

struct GNodeLess {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, Galois::NONE).dist < graph.getData(b, Galois::NONE).dist;
  }
};

struct GNodeGreater {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, Galois::NONE).dist > graph.getData(b, Galois::NONE).dist;
  }
};

//! Simple verifier
static bool verify(GNode source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  
  size_t id = 0;

  for (Graph::iterator src = graph.begin(), ee =
	 graph.end(); src != ee; ++src, ++id) {
    unsigned int dist = graph.getData(*src).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr
        << "found node = " << id
	<< " with label >= INFINITY = " << dist << "\n";
      return false;
    }
    
    for (Graph::edge_iterator 
	   ii = graph.edge_begin(*src),
	   ee = graph.edge_end(*src); ii != ee; ++ii) {
      GNode neighbor = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(*src).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value at "  << id << "\n";
	return false;
      }
    }
  }
  return true;
}

static void readGraph(GNode& source, GNode& report) {
  graph.structureFromFile(filename);

  source = *graph.begin();
  report = *graph.begin();

  std::cout << "Read " << graph.size() << " nodes\n";
  
  size_t id = 0;
  bool foundReport = false;
  bool foundSource = false;
  for (Graph::iterator src = graph.begin(), ee =
      graph.end(); src != ee; ++src) {
    SNode& node = graph.getData(*src, Galois::NONE);
    node.dist = DIST_INFINITY;
    if (id == startNode) {
      source = *src;
      foundSource = true;
    } 
    if (id == reportNode) {
      foundReport = true;
      report = *src;
    }
    ++id;
  }

  if (!foundReport || !foundSource) {
    std::cerr 
      << "failed to set report: " << reportNode 
      << "or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
}


//! Serial BFS using Galois graph
struct SerialAlgo {
  std::string name() const { return "Serial"; }

  void operator()(const GNode source) const {
    Galois::Statistic<unsigned int> counter("Iterations");

    std::deque<UpdateRequest> wl;
    wl.push_back(UpdateRequest(source, 0));

    while (!wl.empty()) {
      UpdateRequest req = wl.front();
      wl.pop_front();

      counter += 1;

      SNode& data = graph.getData(req.n);

      if (data.dist <= req.w)
        continue;

      for (Graph::edge_iterator
             ii = graph.edge_begin(req.n), 
             ee = graph.edge_end(req.n);
           ii != ee; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        unsigned int newDist = req.w + 1;
        if (newDist < graph.getData(dst).dist)
          wl.push_back(UpdateRequest(dst, newDist));
      }
      data.dist = req.w;
    }
  }
};

//! Serial BFS using optimized flags but using Galois graph
struct SerialFlagOptAlgo {
  std::string name() const { return "Serial (Flag Optimized)"; }

  void operator()(const GNode source) const {
    Galois::Statistic<unsigned int> counter("Iterations");

    std::deque<UpdateRequest> wl;
    wl.push_back(UpdateRequest(source, 0));

    while (!wl.empty()) {
      UpdateRequest req = wl.front();
      wl.pop_front();

      counter += 1;

      // Operator begins here
      SNode& data = graph.getData(req.n, Galois::NONE);

      if (data.dist <= req.w)
        continue;

      for (Graph::edge_iterator
             ii = graph.edge_begin(req.n, Galois::NONE), 
             ee = graph.edge_end(req.n, Galois::NONE);
           ii != ee; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        unsigned int newDist = req.w + 1;
        if (newDist < graph.getData(dst, Galois::NONE).dist)
          wl.push_back(UpdateRequest(dst, newDist));
      }
      data.dist = req.w;
    }
  }
};

//! Serial BFS using optimized flags and workset semantics
struct SerialWorkSet {
  std::string name() const { return "Serial (Workset)"; }

  void operator()(const GNode source) const {
    Galois::Statistic<unsigned int> counter("Iterations");

    std::deque<GNode> wl;
    graph.getData(source).dist = 0;

    for (Graph::edge_iterator ii = graph.edge_begin(source), 
           ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      wl.push_back(dst);
    }

    while (!wl.empty()) {
      GNode n = wl.front();
      wl.pop_front();

      counter += 1;

      SNode& data = graph.getData(n);

      unsigned int newDist = data.dist + 1;

      for (Graph::edge_iterator ii = graph.edge_begin(n),
            ei = graph.edge_end(n); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst);

        if (newDist < ddata.dist) {
          ddata.dist = newDist;
          wl.push_back(dst);
        }
      }
    }
  }
};

//! Serial BFS using optimized flags and workset semantics
struct SerialBarrier {
  typedef std::vector<GNode> WL;

  std::string name() const { return "Serial (Barrier)"; }

  void operator()(const GNode source) const {
    Galois::Statistic<unsigned int> counter("Iterations");

    WL wls[2];

    graph.getData(source).dist = 0;

    for (Graph::edge_iterator ii = graph.edge_begin(source), 
           ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      wls[0].push_back(dst);
    }

    unsigned newDist = 1;
    unsigned next = 0;
    unsigned cur;

    do {
      cur = next;
      next = (next + 1) & 1;
      ++newDist;
      WL& wl = wls[cur];
      WL& wlNext = wls[next];

      while (!wl.empty()) {
        GNode n = wl.back();
        wl.pop_back();

        counter += 1;

        for (Graph::edge_iterator ii = graph.edge_begin(n),
              ei = graph.edge_end(n); ii != ei; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          SNode& ddata = graph.getData(dst);

          if (newDist < ddata.dist) {
            ddata.dist = newDist;
            wlNext.push_back(dst);
          }
        }
      }
    } while (!wls[next].empty());
  }
};

//! Serial BFS using optimized flags and workset semantics
struct SerialBare {
  std::string name() const { return "Serial (Bare)"; }

  void operator()(const GNode source) const {
    typedef GaloisRuntime::WorkList::FIFO<GNode,false> WL;

    Galois::Statistic<unsigned int> counter("Iterations");

    WL wls[2];
    graph.getData(source).dist = 0;

    for (Graph::edge_iterator ii = graph.edge_begin(source), 
           ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      wls[0].push(dst);
    }

    unsigned newDist = 1;
    unsigned next = 0;
    unsigned cur;
    boost::optional<GNode> r = wls[0].pop();

    do {
      cur = next;
      next = (next + 1) & 1;
      ++newDist;
      WL& wl = wls[cur];
      WL& wlNext = wls[next];

      while (r) {
        GNode n = *r;

        counter += 1;

        for (Graph::edge_iterator ii = graph.edge_begin(n),
              ei = graph.edge_end(n); ii != ei; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          SNode& ddata = graph.getData(dst);

          if (newDist < ddata.dist) {
            ddata.dist = newDist;
            wlNext.push(dst);
          }
        }
        r = wl.pop();
      }
      r = wlNext.pop();
    } while (r);
  }
};


//! Galois BFS
struct GaloisAlgo {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  std::string name() const { return "Galois"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
#ifdef GALOIS_EXP
    Exp::WorklistExperiment<OBIM,dChunk,Chunk,UpdateRequestIndexer,std::less<UpdateRequest>,std::greater<UpdateRequest> >().for_each(std::cout, &one[0], &one[1], *this);
#else
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
#endif
  }

  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& ctx) const {
    SNode& data = graph.getData(req.n);
    if (data.dist <= req.w)
      return;

    for (Graph::edge_iterator 
           ii = graph.edge_begin(req.n),
           ee = graph.edge_end(req.n);
         ii != ee; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      unsigned int newDist = req.w + 1;
      SNode& rdata = graph.getData(dst);
      if (newDist < rdata.dist)
        ctx.push(UpdateRequest(dst, newDist));
    }
    data.dist = req.w;
  }
};

//! Galois BFS using optimized flags
struct GaloisNoLockAlgo {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  std::string name() const { return "Galois (No Lock)"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
#ifdef GALOIS_EXP
    Exp::WorklistExperiment<OBIM,dChunk,Chunk,UpdateRequestIndexer,std::less<UpdateRequest>,std::greater<UpdateRequest> >().for_each(std::cout, &one[0], &one[1], *this);
#else
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
#endif
  }

  void operator()(UpdateRequest& req, Galois::UserContext<UpdateRequest>& ctx) const {
    SNode& data = graph.getData(req.n, Galois::NONE);
    unsigned int v;
    while (req.w < (v = data.dist)) {
      if (!__sync_bool_compare_and_swap(&data.dist, v, req.w))
        continue;

      for (Graph::edge_iterator 
             ii = graph.edge_begin(req.n, Galois::NONE),
             ee = graph.edge_end(req.n, Galois::NONE);
           ii != ee; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        unsigned int newDist = req.w + 1;
        SNode& rdata = graph.getData(dst, Galois::NONE);
        if (newDist < rdata.dist)
          ctx.push(UpdateRequest(dst, newDist));
      }
      break;
    }
  }
};

//! Galois BFS using optimized flags and workset semantics
struct GaloisWorkSet {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  std::string name() const { return "Galois (Workset)"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;
    
    std::vector<GNode> initial;
    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      initial.push_back(dst);
    }

#ifdef GALOIS_EXP
    Exp::WorklistExperiment<OBIM,dChunk,Chunk,GNodeIndexer,GNodeLess,GNodeGreater>().for_each(std::cout, initial.begin(), initial.end(), *this);
#else
    Galois::for_each<OBIM>(initial.begin(), initial.end(), *this);
#endif
  }

  void operator()(GNode& n, Galois::UserContext<GNode>& ctx) const {
    SNode& data = graph.getData(n, Galois::NONE);

    unsigned int newDist = data.dist + 1;

    for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
          ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          break;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          ctx.push(dst);
          break;
        }
      }
    }
  }
};

//! Galois BFS using optimized flags and barrier scheduling 
struct GaloisManualBarrier {
  std::string name() const { return "Galois (Manual Barrier)"; }

  struct Fn {
    Galois::InsertBag<GNode>& wl;
    unsigned int newDist;
    Fn(Galois::InsertBag<GNode>& w, unsigned int d): wl(w), newDist(d) { }

    void operator()(GNode& n) {
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
            ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::NONE);

        if (ddata.dist <= newDist)
          continue;
        
        ddata.dist = newDist;
        wl.push(dst);
      }
    }
  };

  void operator()(const GNode& source) const {
    Galois::InsertBag<GNode> wls[2];
    unsigned round = 0;

    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      wls[round].push(dst);
    }

    unsigned int newDist = 2;

    while (true) {
      unsigned next = (round + 1) & 1;
      Galois::do_all(wls[round].begin(), wls[round].end(), Fn(wls[next], newDist));
      wls[round].clear();
      Galois::InsertBag<GNode>& next_wl = wls[next];
      round = next;
      ++newDist;
      if (next_wl.begin() == next_wl.end())
        break;
    }
  }
};

//! Galois BFS using optimized flags and barrier scheduling 
template<bool useCas>
struct GaloisBarrier {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  std::string name() const { return "Galois (Barrier)"; }
  typedef std::pair<GNode,int> ItemTy;

  void operator()(const GNode& source) const {
    std::vector<ItemTy> initial;

    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      initial.push_back(ItemTy(dst, 2));
    }

#ifdef GALOIS_EXP
    using namespace GaloisRuntime::WorkList;
    //GaloisRuntime::for_each_x(initial.begin(), initial.end(), *this);
    //else
    Galois::for_each<FIFO<int,false> >(initial.begin(), initial.end(), *this);
    //Galois::for_each<BulkSynchronous<dChunkedLIFO<256> > >(initial.begin(), initial.end(), *this);
#else
    std::cerr << "Need Galois EXP support\n";
    abort();
#endif
  }

  void operator()(const ItemTy& item, Galois::UserContext<ItemTy>& ctx) const {
    GNode n = item.first;

    unsigned int newDist = item.second;

    for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
          ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          break;
        if (!useCas || __sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          if (!useCas)
            ddata.dist = newDist;
          ctx.push(ItemTy(dst, newDist + 1));
          break;
        }
      }
    }
  }
};

template<typename AlgoTy>
void run(const AlgoTy& algo, const GNode& source) {
  Galois::StatTimer T;
  std::cerr << "Running " << algo.name() << " version\n";
  T.start();
  algo(source);
  T.stop();
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  GNode source, report;
  readGraph(source, report);
  std::cout
    << "MEMINFO P1: " << GaloisRuntime::MM::pageAllocInfo() << " pages "
//    << "and "
//    << "(large: " << GaloisRuntime::MM::largeAllocInfo() << " bytes)"
    << "\n";

  switch (algo) {
    case serial: run(SerialAlgo(), source); break;
    case serialOpt: run(SerialFlagOptAlgo(), source); break;
    case serialSet: run(SerialWorkSet(), source); break;
    case serialBarrier: run(SerialBarrier(), source); break;
    case serialBare: run(SerialBare(), source); break;
    case galois: run(GaloisAlgo(), source); break;
    case galoisOpt: run(GaloisNoLockAlgo(), source); break;
    case galoisSet: run(GaloisWorkSet(), source);  break;
    case galoisManualBarrier: run(GaloisManualBarrier(), source); break;
    case galoisBarrierCas: run(GaloisBarrier<true>(), source); break;
    case galoisBarrier: run(GaloisBarrier<false>(), source); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }

  std::cout
    << "MEMINFO P2: " << GaloisRuntime::MM::pageAllocInfo() << " pages "
//    << "and "
//    << "(large: " << GaloisRuntime::MM::largeAllocInfo() << " bytes)"
    << "\n";

  std::cout << "Report node: " << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify(source)) {
      std::cout << "Verification successful.\n";
    } else {
      std::cerr << "Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }

  return 0;
}

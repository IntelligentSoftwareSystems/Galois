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
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_USE_TBB
#include "tbb/parallel_for_each.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/concurrent_vector.h"
#include "tbb/task_scheduler_init.h"
#endif

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
  serialSchardl,
  serialSchardlLCCSRSize,
  serialSchardlLCLinearSize,
  serialMin,
  parallel,
  parallelOpt,
  parallelSet,
  parallelManualBarrier,
  parallelBarrierCas,
  parallelBarrier,
  parallelBarrierInline,
  parallelTBB
};

enum DetAlgo {
  nondet,
  detBase,
  detDisjoint
};

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(0));
static cll::opt<unsigned int> reportNode("reportnode",
    cll::desc("Node to report distance to"),
    cll::init(1));
static cll::opt<BFSAlgo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(serialOpt, "Serial optimized "),
      clEnumVal(serialSet, "Serial optimized with workset semantics"),
      clEnumVal(serialBarrier, "Serial optimized with workset and barrier"),
      clEnumVal(serialBare, "Serial optimized with bare runtime"),
      clEnumVal(serialMin, "Serial optimized with minimal runtime"),
      clEnumVal(serialSchardl, "Serial version of Schardl's implementation"),
      clEnumVal(serialSchardlLCCSRSize, "Serial version of Schardl's implementation using graph representation that is the same size as LC_CSR_Graph"),
      clEnumVal(serialSchardlLCLinearSize, "Serial version of Schardl's implementation using graph representation that is the same size as LC_Linear_Graph"),
      clEnumVal(parallel, "Parallel using Galois"),
      clEnumVal(parallelOpt, "Galois optimized"),
      clEnumVal(parallelSet, "Galois optimized with workset semantics"),
      clEnumVal(parallelManualBarrier, "Galois optimized with workset and manual barrier"),
      clEnumVal(parallelBarrierCas, "Galois optimized with workset and barrier but using CAS"),
      clEnumVal(parallelBarrier, "Galois optimized with workset and barrier"),
#ifdef GALOIS_USE_EXP
      clEnumVal(parallelBarrierInline, "Galois optimized with inlined workset and barrier"),
#endif
#ifdef GALOIS_USE_TBB
      clEnumVal(parallelTBB, "Use TBB instead of Galois"),
#endif
      clEnumValEnd), cll::init(parallelBarrier));
#ifdef GALOIS_USE_DET
static cll::opt<DetAlgo> detAlgo(cll::desc("Deterministic algorithm:"),
    cll::values(
      clEnumVal(nondet, "Non-deterministic"),
      clEnumVal(detBase, "Base execution"),
      clEnumVal(detDisjoint, "Disjoint execution"),
      clEnumValEnd), cll::init(detBase));
#endif
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int dist;
  unsigned int id;
};

typedef Galois::Graph::LC_CSR_Graph<SNode, void> Graph;
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

struct not_consistent {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    for (Graph::edge_iterator 
	   ii = graph.edge_begin(n),
	   ee = graph.edge_end(n); ii != ee; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(dst).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value: " << ddist << " > " << (dist + 1) << "\n";
	return true;
      }
    }
    return false;
  }
};

struct not_visited {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr << "unvisted node: " << dist << " >= INFINITY\n";
      return true;
    }
    return false;
  }
};

struct max_dist {
  Galois::GReduceMax<unsigned int>& m;
  max_dist(Galois::GReduceMax<unsigned int>& _m): m(_m) { }

  void operator()(GNode n) const {
    m.update(graph.getData(n).dist);
  }
};

//! Simple verifier
static bool verify(GNode source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  
  size_t id = 0;
  
  bool okay = Galois::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && Galois::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();

  if (okay) {
    Galois::GReduceMax<unsigned int> m;
    Galois::do_all(graph.begin(), graph.end(), max_dist(m));
    std::cout << "max dist: " << m.get() << "\n";
  }
  
  return okay;
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
      graph.end(); src != ee; ++src, ++id) {
    SNode& node = graph.getData(*src, Galois::NONE);
    node.dist = DIST_INFINITY;
    node.id = id;
    if (id == startNode) {
      source = *src;
      foundSource = true;
    } 
    if (id == reportNode) {
      foundReport = true;
      report = *src;
    }
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
    std::deque<UpdateRequest> wl;
    wl.push_back(UpdateRequest(source, 0));

    while (!wl.empty()) {
      UpdateRequest req = wl.front();
      wl.pop_front();

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
    std::deque<UpdateRequest> wl;
    wl.push_back(UpdateRequest(source, 0));

    while (!wl.empty()) {
      UpdateRequest req = wl.front();
      wl.pop_front();

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
    std::deque<GNode> wl;
    graph.getData(source, Galois::NONE).dist = 0;

    for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::NONE), 
           ei = graph.edge_end(source, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);
      ddata.dist = 1;
      wl.push_back(dst);
    }

    while (!wl.empty()) {
      GNode n = wl.front();
      wl.pop_front();

      SNode& data = graph.getData(n, Galois::NONE);

      unsigned int newDist = data.dist + 1;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
            ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::NONE);

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
    WL wls[2];

    graph.getData(source, Galois::NONE).dist = 0;

    for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::NONE), 
           ei = graph.edge_end(source, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);
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

        for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
              ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          SNode& ddata = graph.getData(dst, Galois::NONE);

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

    WL wls[2];
    graph.getData(source, Galois::NONE).dist = 0;

    for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::NONE), 
           ei = graph.edge_end(source, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);
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

        for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
              ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          SNode& ddata = graph.getData(dst, Galois::NONE);

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

  std::string name() const { return "Galois"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
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

  std::string name() const { return "Galois (No Lock)"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

    UpdateRequest one[1] = { UpdateRequest(source, 0) };
    Galois::for_each<OBIM>(&one[0], &one[1], *this);
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

    Galois::for_each<OBIM>(initial.begin(), initial.end(), *this);
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

//! TBB version based off of GaloisManualBarrier
#ifdef GALOIS_USE_TBB
struct TBB {
  std::string name() const { return "TBB (parallel_for_each)"; }
  typedef tbb::concurrent_vector<GNode,tbb::cache_aligned_allocator<GNode> > ContainerTy;

  struct Fn {
    ContainerTy& wl;
    unsigned int newDist;
    Fn(ContainerTy& w, unsigned int d): wl(w), newDist(d) { }

    void operator()(const GNode& n) const {
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
            ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::NONE);

        if (ddata.dist <= newDist)
          continue;
        
        ddata.dist = newDist;
        wl.push_back(dst);
      }
    }
  };

  void operator()(const GNode& source) const {
    tbb::task_scheduler_init init(numThreads);
    
    ContainerTy wls[2];
    unsigned round = 0;

    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      wls[round].push_back(dst);
    }

    unsigned int newDist = 2;

    while (true) {
      unsigned next = (round + 1) & 1;
      tbb::parallel_for_each(wls[round].begin(), wls[round].end(), Fn(wls[next], newDist));
      wls[round].clear();
      ContainerTy& next_wl = wls[next];
      round = next;
      ++newDist;
      if (next_wl.begin() == next_wl.end())
        break;
    }
  }

};
#else
struct TBB {
  std::string name() const { return "TBB (parallel_for_each)"; }
  void operator()(const GNode& source) const { }
};
#endif

//! Galois BFS using optimized flags and barrier scheduling 
template<typename WL,bool useCas>
struct GaloisBarrier {
  typedef int tt_does_not_need_aborts;

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
    Galois::for_each<WL>(initial.begin(), initial.end(), *this);
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

#ifdef GALOIS_USE_DET
//! Galois BFS using optimized flags and barrier scheduling 
template<DetAlgo Version>
struct GaloisDetBarrier {
  typedef int tt_needs_per_iter_alloc; // For LocalState

  std::string name() const { return "Galois (Deterministic Barrier)"; }
  typedef std::pair<GNode,int> ItemTy;

  struct LocalState {
    typedef std::deque<GNode,Galois::PerIterAllocTy> Pending;
    Pending pending;
    LocalState(GaloisDetBarrier<Version>& self, Galois::PerIterAllocTy& alloc): pending(alloc) { }
  };

  struct IdFn {
    unsigned long operator()(const ItemTy& item) const {
      return graph.getData(item.first, Galois::NONE).id;
    }
  };

  void operator()(const GNode& source) const {
    typedef GaloisRuntime::WorkList::BulkSynchronousInline<> WL;
    std::vector<ItemTy> initial;

    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      initial.push_back(ItemTy(dst, 2));
    }
    switch (Version) {
      case nondet: 
        Galois::for_each<WL>(initial.begin(), initial.end(), *this); break;
      case detBase:
        Galois::for_each_det<false>(initial.begin(), initial.end(), *this); break;
      case detDisjoint:
        Galois::for_each_det<true>(initial.begin(), initial.end(), *this); break;
      default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
    }
  }

  void build(const ItemTy& item, typename LocalState::Pending* pending) const {
    GNode n = item.first;

    unsigned int newDist = item.second;
    
    for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
          ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::ALL);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          break;
        pending->push_back(dst);
        break;
      }
    }
  }

  void modify(const ItemTy& item, Galois::UserContext<ItemTy>& ctx, typename LocalState::Pending* ppending) const {
    GNode n = item.first;

    unsigned int newDist = item.second;
    bool useCas = false;

    for (typename LocalState::Pending::iterator ii = ppending->begin(), ei = ppending->end(); ii != ei; ++ii) {
      GNode dst = *ii;
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

  void operator()(const ItemTy& item, Galois::UserContext<ItemTy>& ctx) const {
    typename LocalState::Pending* ppending;
    if (Version == detDisjoint) {
      bool used;
      LocalState* localState = (LocalState*) ctx.getLocalState(used);
      ppending = &localState->pending;
      if (used) {
        modify(item, ctx, ppending);
        return;
      }
    }
    if (Version == detDisjoint) {
      build(item, ppending);
    } else {
      typename LocalState::Pending pending(ctx.getPerIterAlloc());
      build(item, &pending);
      graph.getData(item.first, Galois::WRITE); // Failsafe point
      modify(item, ctx, &pending);
    }
  }
};
#endif

template<typename NodeIdTy,typename EdgeIdTy>
struct SerialSchardl {
  std::string name() const { return "Serial (Schardl)"; }
  typedef NodeIdTy NodeId;
  typedef EdgeIdTy EdgeId;
  typedef unsigned int Dist;

  unsigned int nNodes;
  unsigned int nEdges;
  EdgeId* nodes;
  NodeId* edges;

  // Convert FileGraph to raw CSR
  void loadGraph(std::string& f) {
    typedef Galois::Graph::LC_FileGraph<NodeId,void> G;
    G g;
    g.structureFromFile(f.c_str());
    g.emptyNodeData();

    nNodes = g.size();
    nEdges = g.sizeEdges();

    NodeId nid = 0;
    for (typename G::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
      g.getData(*ii) = nid++;
    }

    nodes = new EdgeId[nNodes + 1];
    edges = new NodeId[nEdges];

    nid = 0;
    EdgeId eid = 0;
    for (typename G::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
      nodes[nid] = eid;
      for (typename G::edge_iterator jj = g.edge_begin(*ii), ej = g.edge_end(*ii); jj != ej; ++jj) {
        edges[eid++] = g.getEdgeDst(jj);
      }
      ++nid;
    }
    nodes[nid] = eid;
  }

  void operator()(std::string& f) {
    loadGraph(f);

    Dist* distances = new Dist[nNodes];

    Galois::StatTimer T;
    std::cout << "Running " << name() << " version\n";
    T.start();
    bfs(startNode, distances);
    T.stop();

    std::cout << "Report node: " << reportNode << " (dist: " << distances[reportNode] << ")\n";
  }

  void bfs(const NodeId s, Dist distances[]) {
    NodeId *queue = new NodeId[nNodes];
    long head, tail;
    NodeId current;
    Dist newdist;

    for (unsigned i = 0; i < nNodes; ++i) {
      distances[i] = std::numeric_limits<Dist>::max();
    }

    if (s < 0 || s > (NodeId) nNodes)
      return;

    current = s;
    distances[s] = 0;
    head = 0;
    tail = 0;
     do {
      newdist = distances[current]+1;

      EdgeId edgeZero = nodes[current];
      EdgeId edgeLast = nodes[current+1];
      NodeId edge;

      for (EdgeId i = edgeZero; i < edgeLast; i++) {
        edge = edges[i];
        if (newdist < distances[edge]) {
          queue[tail++] = edge;
          distances[edge] = newdist;
        }
      }
      current = queue[head++];
    } while (head <= tail);

    delete[] queue;
  }
};

template<typename AlgoTy>
void run(const AlgoTy& algo) {
  GNode source, report;
  readGraph(source, report);
  Galois::preAlloc((numThreads + (graph.size() * sizeof(SNode) * 2) / GaloisRuntime::MM::pageSize)*8);
  Galois::Statistic("MeminfoPre", GaloisRuntime::MM::pageAllocInfo());

  Galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  algo(source);
  T.stop();
  
  Galois::Statistic("MeminfoPost", GaloisRuntime::MM::pageAllocInfo());

  std::cout << "Report node: " << reportNode << " " << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify(source)) {
      std::cout << "Verification successful.\n";
    } else {
      std::cerr << "Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  using namespace GaloisRuntime::WorkList;
  typedef BulkSynchronous<dChunkedLIFO<256> > BSWL;

#ifdef GALOIS_USE_EXP
  typedef BulkSynchronousInline<> BSInline;
#else
  typedef BSWL BSInline;
#endif

#ifdef GALOIS_USE_DET
  switch (detAlgo) {
    case nondet: run(GaloisDetBarrier<nondet>()); break;
    case detBase: run(GaloisDetBarrier<detBase>()); break;
    case detDisjoint: run(GaloisDetBarrier<detDisjoint>()); break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
#else
  switch (algo) {
    case serial: run(SerialAlgo()); break;
    case serialOpt: run(SerialFlagOptAlgo()); break;
    case serialSet: run(SerialWorkSet()); break;
    case serialBarrier: run(SerialBarrier()); break;
    case serialBare: run(SerialBare()); break;
    case serialSchardl: SerialSchardl<int,int>()(filename); break;
    case serialSchardlLCCSRSize: SerialSchardl<int,long>()(filename); break;
    case serialSchardlLCLinearSize: SerialSchardl<long,long>()(filename); break;
    case serialMin: run(GaloisBarrier<FIFO<int,false>,false>()); break;
    case parallel: run(GaloisAlgo()); break;
    case parallelOpt: run(GaloisNoLockAlgo()); break;
    case parallelSet: run(GaloisWorkSet());  break;
    case parallelManualBarrier: run(GaloisManualBarrier()); break;
    case parallelBarrierCas: run(GaloisBarrier<BSWL,true>()); break;
    case parallelBarrier: run(GaloisBarrier<BSWL,false>()); break;
    case parallelBarrierInline: run(GaloisBarrier<BSInline,false>()); break;
    case parallelTBB: run(TBB()); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }
#endif

  return 0;
}

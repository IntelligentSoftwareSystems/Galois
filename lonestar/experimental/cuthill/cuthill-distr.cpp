/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

// kik
#include "galois/Atomic.h"
#include "galois/runtime/Context.h"
#include "galois/substrate/PtrLock.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/Barrier.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <cmath>
#include <functional>
#include <numeric>

#include <sys/time.h>

//#define FINE_GRAIN_TIMING
//#define GALOIS_JUNE
//#define NO_SORT
//#define SERIAL_SWAP
//#define TOTAL_PREFIX

static const char* name = "Cuthill-McKee Reordering";
static const char* desc = "Computes a reordering of matrix rows and columns "
                          "(or a relabeling of graph nodes)"
                          "according to the Cuthill-McKee heuristic";
static const char* url = 0;

//****** Command Line Options ******
enum BFSAlgo { barrierCM };

enum ExecPhase {
  INIT,
  RUN,
  CLEANUP,
  TOTAL,
};

static const unsigned int DIST_INFINITY =
    std::numeric_limits<unsigned int>::max() - 1;

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
                                        cll::desc("Node to start search from"),
                                        cll::init(DIST_INFINITY));
static cll::opt<unsigned int>
    reportNode("reportnode", cll::desc("Node to report distance to"),
               cll::init(1));
static cll::opt<bool> scaling(
    "scaling",
    llvm::cl::desc(
        "Scale to the number of threads with a given step starting from"),
    llvm::cl::init(false));
static cll::opt<unsigned int> scalingStep("step", cll::desc("Scaling step"),
                                          cll::init(2));
static cll::opt<unsigned int>
    niter("iter", cll::desc("Number of benchmarking iterations"), cll::init(5));
static cll::opt<unsigned int>
    qlen("qlen", cll::desc("Minimum queue length for parallel prefix sum"),
         cll::init(50));
static cll::opt<BFSAlgo> algo(
    cll::desc("Choose an algorithm:"),
    cll::values(clEnumVal(barrierCM, "Barrier-based Parallel Cuthill-McKee"),
                clEnumValEnd),
    cll::init(barrierCM));
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);

struct SNode;
// Hack: Resolve circular definition of Graph and SNode.parent with fact that
// all LC_CSR_Graph::GraphNodes have the same type.
typedef galois::graphs::LC_CSR_Graph<void, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type DummyGraph;
typedef DummyGraph::GraphNode GNode;

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int dist;
  unsigned int id;
  // unsigned int numChildren;
  unsigned int order;
  unsigned int sum;
#ifndef NO_SORT
  unsigned int startindex;
#endif
  galois::GAtomic<unsigned int> numChildren;
  // bool rflag;
  // bool pflag;
  // bool have;
  GNode parent;
  // std::vector<galois::graphs::LC_CSR_Graph<SNode, void>::GraphNode> bucket;
  // galois::gdeque<galois::graphs::LC_CSR_Graph<SNode, void>::GraphNode>*
  // bucket; galois::runtime::LL::SimpleLock mutex;
};

typedef DummyGraph::with_node_data<SNode>::type Graph;

// Check hack above
struct CheckAssertion {
  CheckAssertion() {
    static_assert(std::is_same<GNode, Graph::GraphNode>::value,
                  "GNode != Graph::GraphNode");
  }
};

static CheckAssertion init;

struct Prefix {
  unsigned int id;
  unsigned int val;
  Prefix(unsigned int _id, unsigned _val) : id(_id), val(_val) {}
};

Graph graph;

static size_t degree(const GNode& node) {
  return std::distance(graph.edge_begin(node, galois::MethodFlag::UNPROTECTED),
                       graph.edge_end(node, galois::MethodFlag::UNPROTECTED));
}

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out << "(dist: " << n.dist << ")";
  return out;
}

struct GNodeIndexer {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, galois::MethodFlag::UNPROTECTED).dist;
  }
};

struct GNodeSort {
  bool operator()(const GNode& a, const GNode& b) const {
    return degree(a) < degree(b);
  }
};

std::vector<GNode> initial[2];
std::vector<GNode> perm;
GNode source, report;

// std::map<GNode, unsigned int> order;
// std::vector< std::vector<GNode> > bucket;
// galois::gdeque<GNode> bucket;
galois::InsertBag<GNode> bucket;
galois::substrate::SimpleLock dbglock;

std::vector<std::map<GNode, unsigned int>> redbuck;

// debug
galois::GAtomic<unsigned int> loops     = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> sorts     = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> maxbucket = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> minbucket =
    galois::GAtomic<unsigned int>(DIST_INFINITY);
galois::GAtomic<unsigned int> avgbucket   = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> numbucket   = galois::GAtomic<unsigned int>(0);
galois::GAtomic<unsigned int> smallbucket = galois::GAtomic<unsigned int>(0);

struct PartialSum {
  GNode& operator()(const GNode& partial, GNode& item) {
    /*
    if(graph.getData(item).numChildren > 0)
        graph.getData(item).have = true;

    std::cerr << "[" << graph.getData(item).id << "] " <<
    graph.getData(item).numChildren << " have?: " << graph.getData(item).have <<
    "\n";
    */

    // dbglock.lock();
#ifdef SERIAL_SWAP
    graph.getData(item, galois::MethodFlag::UNPROTECTED).numChildren +=
        graph.getData(partial, galois::MethodFlag::UNPROTECTED).numChildren;
#else
    SNode& idata = graph.getData(item, galois::MethodFlag::UNPROTECTED);
    idata.sum    = idata.numChildren;
    idata.numChildren +=
        graph.getData(partial, galois::MethodFlag::UNPROTECTED).numChildren;
#endif
    // std::cerr << "[" << graph.getData(item,
    // galois::MethodFlag::UNPROTECTED).id << "] " << graph.getData(item,
    // galois::MethodFlag::UNPROTECTED).numChildren << "\n"; dbglock.unlock();
    return item;
  }
};

struct SegReduce {
  unsigned int sum;

  SegReduce(unsigned int _sum) : sum(_sum) {}

  void operator()(const GNode& item, galois::UserContext<GNode>& ctx) {
    graph.getData(item, galois::MethodFlag::UNPROTECTED).numChildren += sum;
  }
};

#ifndef SERIAL_SWAP
struct Swap {
  void operator()(const GNode& item, galois::UserContext<GNode>& ctx) const {
    operator()(item);
  }
  void operator()(const GNode& item) const {
    SNode& idata = graph.getData(item, galois::MethodFlag::UNPROTECTED);
    idata.numChildren -= idata.sum;
#ifndef NO_SORT
    idata.startindex = idata.numChildren;
#endif
  }
};
#endif

#ifndef NO_SORT
struct SortChildren {
  unsigned int round;
  // galois::GReduceMax<unsigned int>& maxlen;

  // SortChildren(unsigned int r, galois::GReduceMax<unsigned int>& m) :
  // round(r), maxlen(m) {}
  SortChildren(unsigned int r, galois::GReduceMax<unsigned int>& m)
      : round(r) {}

  void operator()(GNode& parent, galois::UserContext<GNode>& ctx) const {
    operator()(parent);
  }
  void operator()(GNode& parent) const {
    SNode& pdata = graph.getData(parent, galois::MethodFlag::UNPROTECTED);

    if (pdata.sum > 1) {

      // maxlen.update(pdata.sum);

      // dbglock.lock();
      // std::cerr << "[" << pdata.id << "] sorting: " << pdata.sum << "\n";
      // dbglock.unlock();

      unsigned int limit = pdata.startindex + pdata.sum;
      // sort(initial[round].begin()+pdata.startindex,
      // initial[round].begin()+(pdata.startindex + pdata.sum), GNodeSort());
      sort(initial[round].begin() + pdata.startindex,
           initial[round].begin() + limit, GNodeSort());

      for (unsigned int i = pdata.startindex; i < limit; ++i) {
        SNode& cdata =
            graph.getData(initial[round][i], galois::MethodFlag::UNPROTECTED);
        cdata.order = i;
      }
    }
  }
};

struct IndexChildren {
  unsigned int round;
  galois::GReduceMax<unsigned int>& maxlen;

  IndexChildren(unsigned int r, galois::GReduceMax<unsigned int>& m)
      : round(r), maxlen(m) {}

  void operator()(GNode& parent, galois::UserContext<GNode>& ctx) {
    operator()(parent);
  }
  void operator()(GNode& parent) {
    SNode& pdata = graph.getData(parent, galois::MethodFlag::UNPROTECTED);

    if (pdata.sum > 1) {

      maxlen.update(pdata.sum);

      // dbglock.lock();
      // std::cerr << "[" << pdata.id << "] sorting: " << pdata.sum << "\n";
      // dbglock.unlock();

      for (unsigned int i = pdata.startindex; i < pdata.startindex + pdata.sum;
           ++i) {
        SNode& cdata =
            graph.getData(initial[round][i], galois::MethodFlag::UNPROTECTED);
        cdata.order = i;
      }
    }
  }
};
#endif

#ifndef TOTAL_PREFIX
struct LocalPrefix {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  unsigned int round;
  unsigned int chunk;

  LocalPrefix(unsigned int r, unsigned int c) : round(r), chunk(c) {}

  void operator()(unsigned int me, unsigned int tot) {

    // unsigned int len = initial[round].size();
    // unsigned int start = me * ceil((double) len / tot);
    // unsigned int end = (me+1) * ceil((double) len / tot);
    unsigned int start = me * chunk;
    unsigned int end   = (me + 1) * chunk;

    if (me != tot - 1) {
      // dbglock.lock();
      // std::cerr << "On_each thread: " << me << " step: " << ceil(len / tot)
      // << " start: " << start << " end " << end+1 << "\n"; std::cerr <<
      // "On_each thread tot: " << tot << " len: " << len << " ceil: " <<
      // ceil((double) len / tot) << " floor: " << floor((double) len / tot) <<
      // "\n";

      // std::cerr << graph.getData(*(initial[round].begin()+start),
      // galois::MethodFlag::UNPROTECTED).id << " to " <<
      // graph.getData(*(initial[round].begin()+(end+1)),
      // galois::MethodFlag::UNPROTECTED).id << "\n";

#ifndef SERIAL_SWAP
      SNode& idata =
          graph.getData(initial[round][start], galois::MethodFlag::UNPROTECTED);
      idata.sum = idata.numChildren;
#endif
      std::partial_sum(initial[round].begin() + start,
                       initial[round].begin() + end,
                       initial[round].begin() + start, PartialSum());
      // dbglock.unlock();
    } else {
      // dbglock.lock();
      // std::cerr << "On_each thread: " << me << " size: " << len << " start: "
      // << start << " end " << initial[round].size() << "\n"; std::cerr <<
      // "On_each thread tot: " << tot << " len: " << len << " ceil: " <<
      // ceil((double) len / tot) << " floor: " << floor((double) len / tot) <<
      // "\n"; std::cerr << graph.getData(*(initial[round].begin()+start),
      // galois::MethodFlag::UNPROTECTED).id << " to " <<
      // graph.getData(*(initial[round].end()-1),
      // galois::MethodFlag::UNPROTECTED).id << "\n";

#ifndef SERIAL_SWAP
      SNode& idata =
          graph.getData(initial[round][start], galois::MethodFlag::UNPROTECTED);
      idata.sum = idata.numChildren;
#endif
      std::partial_sum(initial[round].begin() + start, initial[round].end(),
                       initial[round].begin() + start, PartialSum());
      // dbglock.unlock();
    }
  }
};

struct DistrPrefix {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  unsigned int round;
  unsigned int chunk;

  DistrPrefix(unsigned int r, unsigned int c) : round(r), chunk(c) {}

  void operator()(unsigned int me, unsigned int tot) {
    if (me > 0) {
      if (me != tot - 1) {

        // unsigned int len = initial[round].size();
        unsigned int start = me * chunk;
        unsigned int end   = (me + 1) * chunk - 1;
        unsigned int val   = graph
                               .getData(initial[round][start - 1],
                                        galois::MethodFlag::UNPROTECTED)
                               .numChildren;

        // dbglock.lock();
        // std::cerr << "On_each thread: " << me << " step: " << ceil(len / tot)
        // << " start: " << start << " end " << end+1 << "\n"; std::cerr <<
        // "On_each thread tot: " << tot << " len: " << len << " ceil: " <<
        // ceil((double) len / tot) << " floor: " << floor((double) len / tot)
        // <<
        // "\n";

        // std::cerr << graph.getData(*(initial[round].begin()+start),
        // galois::MethodFlag::UNPROTECTED).id << " to " <<
        // graph.getData(*(initial[round].begin()+(end+1)),
        // galois::MethodFlag::UNPROTECTED).id << "\n";
        for (unsigned int i = start; i < end; ++i) {
          graph.getData(initial[round][i], galois::MethodFlag::UNPROTECTED)
              .numChildren += val;
          // std::cerr << "Loop: " << i << " size: " << seglen << " start: " <<
          // start << " end " << end << "\n";
        }
        // dbglock.unlock();
      } else {
        // dbglock.lock();
        // std::cerr << "On_each thread: " << me << " size: " << len << " start:
        // " << start << " end " << initial[round].size() << "\n"; std::cerr <<
        // "On_each thread tot: " << tot << " len: " << len << " ceil: " <<
        // ceil((double) len / tot) << " floor: " << floor((double) len / tot)
        // <<
        // "\n"; std::cerr << graph.getData(*(initial[round].begin()+start),
        // galois::MethodFlag::UNPROTECTED).id << " to " <<
        // graph.getData(*(initial[round].end()-1),
        // galois::MethodFlag::UNPROTECTED).id << "\n";

        unsigned int len   = initial[round].size();
        unsigned int start = me * chunk;
        unsigned int val   = graph
                               .getData(initial[round][start - 1],
                                        galois::MethodFlag::UNPROTECTED)
                               .numChildren;

        for (unsigned int i = start; i < len; ++i) {
          graph.getData(initial[round][i], galois::MethodFlag::UNPROTECTED)
              .numChildren += val;
          // std::cerr << "Loop: " << i << " size: " << seglen << " start: " <<
          // start << " end " << end << "\n";
        }
        // dbglock.unlock();
      }
    }
  }
};

#else

struct TotalPrefix {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  unsigned int round;
  unsigned int chunk;
  galois::runtime::PthreadBarrier barrier;

  TotalPrefix(unsigned int r, unsigned int c, galois::runtime::PthreadBarrier b)
      : round(r), chunk(c), barrier(b) {}

  void operator()(unsigned int me, unsigned int tot) {

    unsigned int len = initial[round].size();
    unsigned int start = me * chunk;
    unsigned int end = (me + 1) * chunk;

    if (me != tot - 1) {
#ifndef SERIAL_SWAP
      SNode& idata =
          graph.getData(initial[round][start], galois::MethodFlag::UNPROTECTED);
      idata.sum = idata.numChildren;
#endif
      std::partial_sum(initial[round].begin() + start,
                       initial[round].begin() + end,
                       initial[round].begin() + start, PartialSum());
    } else {
#ifndef SERIAL_SWAP
      SNode& idata =
          graph.getData(initial[round][start], galois::MethodFlag::UNPROTECTED);
      idata.sum = idata.numChildren;
#endif
      std::partial_sum(initial[round].begin() + start, initial[round].end(),
                       initial[round].begin() + start, PartialSum());
    }

    barrier.wait();

    if (me == 0) {
      for (unsigned int i = 1; i < tot - 1; ++i) {
        start = i * chunk;
        end = (i + 1) * chunk - 1;
        graph.getData(initial[round][end], galois::MethodFlag::UNPROTECTED)
            .numChildren += graph
                                .getData(initial[round][start - 1],
                                         galois::MethodFlag::UNPROTECTED)
                                .numChildren;
      }
    }

    barrier.wait();

    if (me != 0) {
      if (me != tot - 1) {
        --end;
        unsigned int val = graph
                               .getData(initial[round][start - 1],
                                        galois::MethodFlag::UNPROTECTED)
                               .numChildren;
        for (unsigned int i = start; i < end; ++i) {
          graph.getData(initial[round][i], galois::MethodFlag::UNPROTECTED)
              .numChildren += val;
        }
      } else {
        unsigned int val = graph
                               .getData(initial[round][start - 1],
                                        galois::MethodFlag::UNPROTECTED)
                               .numChildren;
        for (unsigned int i = start; i < len; ++i) {
          graph.getData(initial[round][i], galois::MethodFlag::UNPROTECTED)
              .numChildren += val;
        }
      }
    }
  }
};

#endif

// Find a good starting node for CM based on minimum degree
static void findStartingNode(GNode& starting) {
  unsigned int mindegree = DIST_INFINITY;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {
    unsigned int nodedegree = degree(*src);

    if (nodedegree < mindegree) {
      mindegree = nodedegree;
      starting  = *src;
    }
  }

  SNode& data = graph.getData(starting);
  std::cerr << "Starting Node: " << data.id << " degree: " << degree(starting)
            << "\n";
}

// Compute variance around mean distance from the source
static void variance(unsigned int mean) {
  unsigned int n = 0;
  double M2      = 0.0;
  double var     = 0.0;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {
    SNode& data = graph.getData(*src);
    M2 += (data.dist - mean) * (data.dist - mean);
  }

  var = M2 / (n - 1);
  std::cout << "var: " << var << " mean: " << mean << "\n";
}

struct not_consistent {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n);
         ii != ei; ++ii) {
      GNode dst          = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(dst).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value for " << graph.getData(dst).id << ": "
                  << ddist << " > " << (dist + 1) << "\n";
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
      std::cerr << "unvisited node " << graph.getData(n).id << ": " << dist
                << " >= INFINITY\n";
      return true;
    }
    // std::cerr << "visited node " << graph.getData(n).id << ": " << dist <<
    // "\n";
    return false;
  }
};

//! Simple verifier
static bool verify(GNode& source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }

  // size_t id = 0;

#ifdef GALOIS_JUNE
  bool okay =
      galois::find_if(graph.begin(), graph.end(), not_consistent()) ==
          graph.end() &&
      galois::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#else
  bool okay = galois::ParallelSTL::find_if(graph.begin(), graph.end(),
                                           not_consistent()) == graph.end() &&
              galois::ParallelSTL::find_if(graph.begin(), graph.end(),
                                           not_visited()) == graph.end();
#endif

  if (okay) {
    galois::GReduceMax<unsigned int> maxDist;
    galois::GAccumulator<unsigned> sum;
    galois::GAccumulator<unsigned> count;

    galois::do_all(galois::iterate(graph), [&](const GNode& n) {
      auto d = graph.getData(n, galois::MethodFlag::UNPROTECTED).dist;
      if (d < INFINITY) {
        maxDist.update(d);
        sum += d;
        count += 1;
      }
    });
    std::cout << "max dist: " << maxDist.reduce() << "\n";
    unsigned mean = sum.reduce() / count.reduce();
    std::cout << "avg dist: " << mean << "\n";

    variance(mean);
  }

  return okay;
}

// Compute maximum bandwidth for a given graph
struct banddiff {

  galois::GAtomic<long int>& maxband;
  galois::GAtomic<long int>& profile;
  std::vector<GNode>& nmap;

  banddiff(galois::GAtomic<long int>& _mb, galois::GAtomic<long int>& _pr,
           std::vector<GNode>& _nm)
      : maxband(_mb), profile(_pr), nmap(_nm) {}

  void operator()(const GNode& source) const {

    long int maxdiff = 0;
    SNode& sdata     = graph.getData(source, galois::MethodFlag::UNPROTECTED);

    for (Graph::edge_iterator
             ii = graph.edge_begin(source, galois::MethodFlag::UNPROTECTED),
             ei = graph.edge_end(source, galois::MethodFlag::UNPROTECTED);
         ii != ei; ++ii) {

      GNode dst    = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      long int diff = abs(static_cast<long int>(sdata.id) -
                          static_cast<long int>(ddata.id));
      // long int diff = (long int) sdata.id - (long int) ddata.id;
      maxdiff = diff > maxdiff ? diff : maxdiff;
    }

    long int globalmax = maxband;
    profile += maxdiff;

    if (maxdiff > globalmax) {
      while (!maxband.cas(globalmax, maxdiff)) {
        globalmax = maxband;
        if (!(maxdiff > globalmax))
          break;
      }
    }
  }
};

// Parallel loop for maximum bandwidth computation
static void bandwidth(std::string msg) {
  galois::GAtomic<long int> bandwidth = galois::GAtomic<long int>(0);
  galois::GAtomic<long int> profile   = galois::GAtomic<long int>(0);
  std::vector<GNode> nodemap;
  std::vector<bool> visited;
  visited.reserve(graph.size());
  ;
  visited.resize(graph.size(), false);
  ;
  nodemap.reserve(graph.size());
  ;

  // static int count = 0;
  // std::cout << graph.size() << "Run: " << count++ << "\n";

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {
    nodemap[graph.getData(*src, galois::MethodFlag::UNPROTECTED).id] = *src;
  }

  // Computation of bandwidth and profile in parallel
  galois::do_all(graph.begin(), graph.end(),
                 banddiff(bandwidth, profile, nodemap));

  unsigned int nactiv = 0;
  unsigned int maxwf  = 0;
  unsigned int curwf  = 0;
  double mswf         = 0.0;

  // Computation of maximum and root-square-mean wavefront. Serial
  for (unsigned int i = 0; i < graph.size(); ++i) {

    for (Graph::edge_iterator
             ii = graph.edge_begin(nodemap[i], galois::MethodFlag::UNPROTECTED),
             ei = graph.edge_end(nodemap[i], galois::MethodFlag::UNPROTECTED);
         ii != ei; ++ii) {

      GNode neigh  = graph.getEdgeDst(ii);
      SNode& ndata = graph.getData(neigh, galois::MethodFlag::UNPROTECTED);

      // std::cerr << "neigh: " << ndata.id << "\n";
      if (visited[ndata.id] == false) {
        visited[ndata.id] = true;
        nactiv++;
        //	std::cerr << "val: " << nactiv<< "\n";
      }
    }

    SNode& idata = graph.getData(nodemap[i], galois::MethodFlag::UNPROTECTED);

    if (visited[idata.id] == false) {
      visited[idata.id] = true;
      curwf             = nactiv + 1;
    } else
      curwf = nactiv--;

    maxwf = curwf > maxwf ? curwf : maxwf;
    mswf += (double)curwf * curwf;
  }

  mswf = mswf / graph.size();

  std::cout << msg << " Bandwidth: " << bandwidth << "\n";
  std::cout << msg << " Profile: " << profile << "\n";
  std::cout << msg << " Max WF: " << maxwf << "\n";
  std::cout << msg << " Mean-Square WF: " << mswf << "\n";
  std::cout << msg << " RMS WF: " << sqrt(mswf) << "\n";

  // nodemap.clear();
}

static void permute(std::vector<GNode>& ordering) {

  std::vector<GNode> nodemap;
  nodemap.reserve(graph.size());
  ;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {

    nodemap[graph.getData(*src, galois::MethodFlag::UNPROTECTED).id] = *src;
  }

  unsigned int N = ordering.size() - 1;

  // std::cout << " ordering size: " << ordering.size() << "\n";

  for (int i = N; i >= 0; --i) {
    // RCM
    graph.getData(ordering[i], galois::MethodFlag::UNPROTECTED).id = N - i;
    // CM
    // graph.getData(ordering[i], galois::MethodFlag::UNPROTECTED).id = i;
  }
}

// Clear node data to re-execute on specific graph
struct resetNode {
  void operator()(const GNode& n) const {
    SNode& node      = graph.getData(n, galois::MethodFlag::UNPROTECTED);
    node.dist        = DIST_INFINITY;
    node.numChildren = 0;
    // node.numChildren = 0;
    // node.rflag = false;
    // node.pflag = false;
    // node.have = false;
    node.parent = n;
    // order[n] = DIST_INFINITY;
    node.order = DIST_INFINITY;
    // node.bucket->clear();
  }
};

void resetGraph() {
  initial[0].clear();
  initial[1].clear();
  bucket.clear();
  galois::do_all(graph.begin(), graph.end(), resetNode());
}

// Read graph from a binary .gr as dirived from a Matrix Market .mtx using
// graph-convert
static void readGraph(GNode& source, GNode& report) {
  galois::graphs::readGraph(graph, filename);

  source = *graph.begin();
  report = *graph.begin();

  size_t nnodes = graph.size();
  std::cout << "Read " << nnodes << " nodes\n";

  size_t id        = 0;
  bool foundReport = false;
  bool foundSource = false;

  // bucket.reserve(nnodes);
  // order.reserve(nnodes);

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei;
       ++src) {

    SNode& node = graph.getData(*src, galois::MethodFlag::UNPROTECTED);
    node.dist   = DIST_INFINITY;
    node.id     = id;
    node.parent = id;
    // node.bucket = new galois::gdeque<GNode>();
    // node.numChildren = 0;
    node.numChildren = galois::GAtomic<unsigned int>(0);
    // node.rflag = false;
    // node.pflag = false;
    // node.have = false;
    // order[*src] = DIST_INFINITY;
    node.order = DIST_INFINITY;

    // std::cout << "Report node: " << reportNode << " (dist: " <<
    // distances[reportNode] << ")\n";

    if (id == startNode) {
      source      = *src;
      foundSource = true;
    }
    if (id == reportNode) {
      foundReport = true;
      report      = *src;
    }
    ++id;
  }

  if (startNode == DIST_INFINITY) {
    findStartingNode(source);
    foundSource = true;
  }

  if (!foundReport || !foundSource) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
}

struct BarrierNoDup {
  std::string name() const { return "Cuthill (Inline) Barrier)"; }

  void operator()(const GNode& source) const {

#ifdef FINE_GRAIN_TIMING
    galois::TimeAccumulator vTmain[6];
    vTmain[0] = galois::TimeAccumulator();
    vTmain[1] = galois::TimeAccumulator();
    vTmain[2] = galois::TimeAccumulator();
    vTmain[3] = galois::TimeAccumulator();
    vTmain[4] = galois::TimeAccumulator();
    vTmain[5] = galois::TimeAccumulator();

    vTmain[0].start();
#endif

    unsigned int round = 0;

    initial[0].reserve(100);
    initial[1].reserve(100);
    perm.reserve(graph.size());

    SNode& sdata = graph.getData(source);
    sdata.dist   = 0;
    // order[source] = 0;
    sdata.order = 0;
    perm.push_back(source);

    // round = (round + 1) & 1;

    for (Graph::edge_iterator ii = graph.edge_begin(source),
                              ei = graph.edge_end(source);
         ii != ei; ++ii) {
      GNode dst         = graph.getEdgeDst(ii);
      SNode& ddata      = graph.getData(dst);
      ddata.dist        = 1;
      ddata.numChildren = 0;
      ddata.parent      = sdata.id;
      initial[round].push_back(dst);
      // sdata.numChildren++;
    }

    sort(initial[round].begin(), initial[round].end(), GNodeSort());

    for (unsigned int i = 0; i < initial[round].size(); ++i) {
      // order[initial[round][i]] = i+1;
      graph.getData(initial[round][i]).order = i + 1;
    }

#ifdef FINE_GRAIN_TIMING
    vTmain[0].stop();
#endif

    // unsigned int added = 0;
    galois::GAtomic<unsigned int> added = galois::GAtomic<unsigned int>(0);
    ;
    galois::GAtomic<unsigned int> temp = galois::GAtomic<unsigned int>(0);
    ;

    // unsigned int depth = 0;
    unsigned int thr = galois::getActiveThreads();
    // galois::runtime::PthreadBarrier barrier(thr);
    __attribute__((unused)) galois::substrate::Barrier& barrier =
        galois::runtime::getBarrier(thr);

    while (true) {
      unsigned next = (round + 1) & 1;

#ifdef FINE_GRAIN_TIMING
      vTmain[1].start();
#endif

      // std::cerr << "Depth: " << ++depth << " ";
      // std::cerr << "Parents: " << initial[round].size() << "\n";
      galois::do_all(initial[round].begin(), initial[round].end(), Expand(next),
                     galois::loopname("expand"));
#ifdef FINE_GRAIN_TIMING
      vTmain[1].stop();
      vTmain[2].start();
#endif

      // std::cerr << "Children: " << bucket.size() << "\n";
      galois::do_all(bucket.begin(), bucket.end(), Children(),
                     galois::loopname("reduction"));
      /*
                  for(galois::InsertBag<GNode>::iterator ii = bucket.begin(), ei
         = bucket.end(); ii != ei; ++ii){ SNode& cdata = graph.getData(*ii,
         galois::MethodFlag::UNPROTECTED); if(!cdata.rflag) {
                          graph.getData(cdata.parent,
         galois::MethodFlag::UNPROTECTED).numChildren++;
                          graph.getData(cdata.parent,
         galois::MethodFlag::UNPROTECTED).have = true; cdata.rflag = true;
                      }
                  }
                  */

#ifdef FINE_GRAIN_TIMING
      vTmain[2].stop();
      vTmain[3].start();
#endif

      added = 0;
      temp  = 0;

      /*
      std::cerr << "Size: " << initial[round].size() << "\n";
      for(std::vector<GNode>::iterator ii = initial[round].begin(), ei =
      initial[round].end(); ii != ei; ++ii){ SNode& data = graph.getData(*ii,
      galois::MethodFlag::UNPROTECTED); std::cerr << data.id << " ";
      }
      std::cerr << "\n";

      for(std::vector<GNode>::iterator ii = initial[round].begin(), ei =
      initial[round].end(); ii != ei; ++ii){ SNode& data = graph.getData(*ii,
      galois::MethodFlag::UNPROTECTED); std::cerr << data.numChildren << " ";
      }
      std::cerr << "\n";
      */

      unsigned int seglen = initial[round].size();
      unsigned int chunk  = (seglen + (thr - 1)) / thr;
      unsigned int start;
      unsigned int end;

      // std::cerr << "Segment : " << seglen / thr << "\n";

      // if(seglen / thr > 2)
      // if(seglen > qlen)
      if (seglen > 1000) {
#ifdef TOTAL_PREFIX
        galois::on_each(TotalPrefix(round, chunk, barrier),
                        galois::loopname("totalprefix"));
#else

        galois::on_each(LocalPrefix(round, chunk),
                        galois::loopname("localprefix"));

        for (unsigned int i = 1; i < thr - 1; ++i) {
          start = i * chunk;
          end = (i + 1) * chunk - 1;
          graph.getData(initial[round][end], galois::MethodFlag::UNPROTECTED)
              .numChildren += graph
                                  .getData(initial[round][start - 1],
                                           galois::MethodFlag::UNPROTECTED)
                                  .numChildren;
        }

        galois::on_each(DistrPrefix(round, chunk),
                        galois::loopname("distrprefix"));
#endif
      } else {
#ifndef SERIAL_SWAP
        SNode& idata =
            graph.getData(initial[round][0], galois::MethodFlag::UNPROTECTED);
        idata.sum = idata.numChildren;
#endif
        std::partial_sum(initial[round].begin(), initial[round].end(),
                         initial[round].begin(), PartialSum());
      }

      // std::partial_sum(initial[round].begin(), initial[round].end(),
      // initial[round].begin(), PartialSum());

      /*
      std::cerr << "Size for prefix sum: " << initial[round].size() << "\n";
      for(std::vector<GNode>::iterator ii = initial[round].begin(), ei =
      initial[round].end(); ii != ei; ++ii){ SNode& data = graph.getData(*ii,
      galois::MethodFlag::UNPROTECTED); std::cerr << data.numChildren << " ";
      }
          std::cerr << "\n";

      std::cerr << "Size for sum: " << initial[round].size() << "\n";
      for(std::vector<GNode>::iterator ii = initial[round].begin(), ei =
      initial[round].end(); ii != ei; ++ii){ SNode& data = graph.getData(*ii,
      galois::MethodFlag::UNPROTECTED); std::cerr << data.sum << " ";
      }
          std::cerr << "\n";
          */

      /*
      #ifdef FINE_GRAIN_TIMING
      vTmain[3].stop();
      vTmain[5].start();
      #endif
      */

#ifdef SERIAL_SWAP
      for (std::vector<GNode>::iterator ii = initial[round].begin(),
                                        ei = initial[round].end();
           ii != ei; ++ii) {
        std::swap(
            graph.getData(*ii, galois::MethodFlag::UNPROTECTED).numChildren,
            added);
      }
#else
      added = graph
                  .getData(initial[round][seglen - 1],
                           galois::MethodFlag::UNPROTECTED)
                  .numChildren;
      galois::do_all(initial[round].begin(), initial[round].end(), Swap(),
                     galois::loopname("swap"));
#endif

      /*
                  std::cerr << "After swap Size for prefix sum: " <<
         initial[round].size() << "\n"; for(std::vector<GNode>::iterator ii =
         initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
                      SNode& data = graph.getData(*ii,
         galois::MethodFlag::UNPROTECTED); std::cerr << data.numChildren << " ";
                  }
                      std::cerr << "\n";

                  std::cerr << "Size for startindex: " << initial[round].size()
         << "\n"; for(std::vector<GNode>::iterator ii = initial[round].begin(),
         ei = initial[round].end(); ii != ei; ++ii){ SNode& data =
         graph.getData(*ii, galois::MethodFlag::UNPROTECTED); std::cerr <<
         data.startindex << " ";
                  }
                      std::cerr << "\n";

                  std::cerr << "After swap Size for sum: " <<
         initial[round].size() << "\n"; for(std::vector<GNode>::iterator ii =
         initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
                      SNode& data = graph.getData(*ii,
         galois::MethodFlag::UNPROTECTED); std::cerr << data.sum << " ";
                  }
                      std::cerr << "\n";
                      */

      // std::cerr << "total: " << added << "\n";

      /*
      for(std::vector<GNode>::iterator ii = initial[round].begin(), ei =
      initial[round].end(); ii != ei; ++ii){ SNode& data = graph.getData(*ii,
      galois::MethodFlag::UNPROTECTED); temp = data.numChildren;
          data.numChildren = added;
          added += temp;
      }
      */

      initial[next].resize(added);

      // std::cerr << "After partial sum: " << added << "\n";

#ifdef FINE_GRAIN_TIMING
      vTmain[3].stop();
#endif

      if (added == 0) {
#ifdef FINE_GRAIN_TIMING
        std::cerr << "Init: " << vTmain[0].get() << "\n";
        std::cerr << "Expand(par): " << vTmain[1].get() << "\n";
        std::cerr << "Reduction(par): " << vTmain[2].get() << "\n";
        std::cerr << "PartialSum(par): " << vTmain[3].get() << "\n";
        // std::cerr << "Swap(ser): " << vTmain[5].get() << "\n";
        std::cerr << "Placement(par): " << vTmain[4].get() << "\n";
        std::cout << "& " << vTmain[1].get() << " & " << vTmain[2].get()
                  << " & " << vTmain[3].get() << " & " << vTmain[4].get()
                  << " & "
                  << vTmain[1].get() + vTmain[2].get() + vTmain[3].get() +
                         vTmain[4].get()
                  << "\n";
#endif
        perm.insert(perm.end(), initial[round].begin(), initial[round].end());
        break;
      }

#ifdef FINE_GRAIN_TIMING
      vTmain[4].start();
#endif
      // galois::for_each<WL>(initial[round].begin(), initial[round].end(),
      // Place(next));
      galois::do_all(bucket.begin(), bucket.end(), Place(next),
                     galois::loopname("placement"));
#ifndef NO_SORT
      galois::GReduceMax<unsigned int> maxlen;
      galois::do_all(initial[round].begin(), initial[round].end(),
                     SortChildren(next, maxlen), galois::loopname("sort"));
// galois::do_all<>(initial[round].begin(), initial[round].end(),
// IndexChildren(next, maxlen), "index"); std::cout << "max sorting len: " <<
// maxlen.get() << "\n";
#endif

      perm.insert(perm.end(), initial[round].begin(), initial[round].end());
      initial[round].clear();
      bucket.clear();
      round = next;

#ifdef FINE_GRAIN_TIMING
      vTmain[4].stop();
#endif
    }

    /*
            std::cerr << "Order: \n";
            for(int i=0; i<order.size(); ++i){
                std::cerr << i << " at: " << order[i] << "\n";
            }
            */
    // std::cerr << "\n";
  }

  struct Expand {
    unsigned int round;

    Expand(unsigned int r) : round(r) {}

    // For fine-grain timing inside foreach threads
    // Millis
    unsigned long tget(unsigned int start_hi, unsigned int start_low,
                       unsigned int stop_hi, unsigned int stop_low) const {
      unsigned long msec = stop_hi - start_hi;
      msec *= 1000;
      if (stop_low > start_low)
        msec += (stop_low - start_low) / 1000;
      else {
        msec -= 1000; // borrow
        msec += (stop_low + 1000000 - start_low) / 1000;
      }
      return msec;
    }

    // Micros
    unsigned long utget(unsigned int start_hi, unsigned int start_low,
                        unsigned int stop_hi, unsigned int stop_low) const {
      unsigned long usec = stop_hi - start_hi;
      usec *= 1000000;
      if (stop_low > start_low)
        usec += (stop_low - start_low);
      else {
        usec -= 1000000; // borrow
        usec += (stop_low + 1000000 - start_low);
      }
      return usec;
    }

    void operator()(GNode& n, galois::UserContext<GNode>& ctx) const {
      operator()(n);
    }
    void operator()(GNode& n) const {
      SNode& sdata         = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      unsigned int newDist = sdata.dist + 1;

      for (Graph::edge_iterator
               ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
               ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED);
           ii != ei; ++ii) {
        GNode dst    = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        if (ddata.dist < newDist)
          continue;

        unsigned int oldDist;

        GNode parent;

        while (true) {
          oldDist = ddata.dist;
          // It actually enters only with equality
          if (oldDist <= newDist) {
            break;
          }
          if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            bucket.push_back(dst);
            break;
          }
        }

        /*
                        if(ddata.dist > newDist){
                            ddata.dist = newDist;
                            bucket.push_back(dst);
                        }
                        */

        // parent = ddata.parent;
        //__sync_bool_compare_and_swap(&ddata.parent, parent, n);

        /*
                        if(graph.getData(ddata.parent).order > sdata.order)
                            ddata.parent = n;
                        if(order[ddata.parent] > order[n])
                            ddata.parent = n;
                        GNode parent;
                        */
        /*
                    dbglock.lock();
                    std::cerr << "[" << sdata.id << "] checking: " << ddata.id
           << " current parent: " << graph.getData(ddata.parent).id << " nc: "
           << graph.getData(ddata.parent).numChildren << " me: " <<
           sdata.numChildren << "\n"; dbglock.unlock();

        */
        while (true) {
          parent = ddata.parent;
          // if(order[parent] > order[n]){
          if (graph.getData(parent, galois::MethodFlag::UNPROTECTED).order >
              sdata.order) {
            // if(graph.getData(ddata.parent).numChildren > sdata.numChildren){
            if (__sync_bool_compare_and_swap(&ddata.parent, parent, n)) {
              break;
            }
            continue;
          }
          break;
        }
      }
    }
  };

  /*
      struct Place {
          unsigned int round;

          Place(unsigned int r) : round(r) {}

          void operator()(GNode& parent, galois::UserContext<GNode>& ctx) {
              SNode& pdata = graph.getData(parent,
     galois::MethodFlag::UNPROTECTED);

              if(!pdata.have)
                  return;

              unsigned int index = pdata.numChildren;

              //unsigned int count = 0;
              //unsigned int actual = 0;
              for(galois::InsertBag<GNode>::iterator ii = bucket.begin(), ei =
     bucket.end(); ii != ei; ++ii){ SNode& cdata = graph.getData(*ii,
     galois::MethodFlag::UNPROTECTED);
                  //count++;
                  if(!cdata.pflag && cdata.parent == parent){
                      //order[*ii] = index;
                      cdata.order = index;
                      initial[round][index++] = *ii;
                      cdata.pflag = true;
                      //actual++;
                  }
              }
          }
      };
      */

  struct Place {
    unsigned int round;

    Place(unsigned int r) : round(r) {}

    void operator()(GNode& child, galois::UserContext<GNode>& ctx) const {
      operator()(child);
    }
    void operator()(GNode& child) const {
      SNode& cdata = graph.getData(child, galois::MethodFlag::UNPROTECTED);
      SNode& pdata =
          graph.getData(cdata.parent, galois::MethodFlag::UNPROTECTED);

      unsigned int index    = pdata.numChildren++;
      cdata.order           = index;
      initial[round][index] = child;

      /*
      dbglock.lock();
      std::cerr << "[" << pdata.id << "] scanned: " << count << " added: " <<
      actual << "\n"; dbglock.unlock();
      */

      /*
           if(sdata.sum > 1) {
           sort(sdata.bucket.begin(), sdata.bucket.end(), GNodeSort());
           }
           */
    }
  };

  struct Children {
    void operator()(GNode& child, galois::UserContext<GNode>& ctx) const {
      operator()(child);
    }
    void operator()(GNode& child) const {
      SNode& cdata = graph.getData(child, galois::MethodFlag::UNPROTECTED);
      // graph.getData(cdata.parent,
      // galois::MethodFlag::UNPROTECTED).mutex.lock();
      graph.getData(cdata.parent, galois::MethodFlag::UNPROTECTED)
          .numChildren++;
      // graph.getData(cdata.parent, galois::MethodFlag::UNPROTECTED).have =
      // true; graph.getData(cdata.parent,
      // galois::MethodFlag::UNPROTECTED).mutex.unlock();
    }
  };

  /*
      struct Children {
          void operator()(GNode& owner, galois::UserContext<GNode>& ctx) {
              SNode& odata = graph.getData(owner,
     galois::MethodFlag::UNPROTECTED);
              //for(std::vector<GNode>::iterator ii = odata.bucket.begin(), ei =
     odata.bucket.end(); ii != ei; ++ii){ for(galois::gdeque<GNode>::iterator ii
     = odata.bucket->begin(), ei = odata.bucket->end(); ii != ei; ++ii){ SNode&
     cdata = graph.getData(*ii, galois::MethodFlag::UNPROTECTED);
                  //I'll make it GAtomic
                  graph.getData(cdata.parent,
     galois::MethodFlag::UNPROTECTED).mutex.lock(); graph.getData(cdata.parent,
     galois::MethodFlag::UNPROTECTED).numChildren++; graph.getData(cdata.parent,
     galois::MethodFlag::UNPROTECTED).mutex.unlock();
              }
          }
      };
      */
};

template <typename AlgoTy>
void run(const AlgoTy& algo) {

  int maxThreads = numThreads;
  std::vector<galois::TimeAccumulator> vT(maxThreads + 20);

  // Measure time to read graph
  // vT[INIT] = galois::TimeAccumulator();
  galois::StatTimer itimer("InitTime");
  itimer.start();

  readGraph(source, report);

  itimer.stop();

  bandwidth("Initial");

  // I've observed cold start. First run takes a few millis more.
  // algo(source);
  // resetGraph();

  galois::StatTimer T;
  T.start();
  galois::StatTimer Tcuthill("CuthillTime");
  Tcuthill.start();

  algo(source);

  Tcuthill.stop();
  T.stop();

  permute(perm);
  bandwidth("Permuted");
  std::cout << "done!\n";

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

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  using namespace galois::worklists;
  typedef PerSocketChunkLIFO<8> BSWL_LIFO;
  typedef PerSocketChunkFIFO<8> BSWL_FIFO;

  switch (algo) {
  case barrierCM:
    run(BarrierNoDup());
    break;
  default:
    std::cerr << "Unknown algorithm" << algo << "\n";
    abort();
  }

  return 0;
}

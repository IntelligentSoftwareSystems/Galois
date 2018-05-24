/** Cuthull-McKee -*- C++ -*-
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "galois/Galois.h"
#include "galois/Atomic.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <deque>
#include <numeric>

static const char* name = "Cuthill-McKee Reordering";
static const char* desc =
  "Computes a reordering of matrix rows and columns (or a relabeling of graph nodes) "
  "according to the Cuthill-McKee heuristic";
static const char* url = 0;

enum PseudoAlgo {
  simplePseudo,
  eagerPseudo,
  fullPseudo
};

enum OutputType {
  permutedgraph,
  permutation
};

enum OutputEdgeType {
  void_,
  int32,
  int64
};

typedef unsigned int DistType;
static const DistType DIST_INFINITY = std::numeric_limits<DistType>::max() - 1;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional, cll::desc("[output file]"), cll::init(""));
static cll::opt<OutputType> writeType("output", cll::desc("Output type:"),
    cll::values(
      clEnumValN(OutputType::permutedgraph, "graph", "Permuted graph"),
      clEnumValN(OutputType::permutation, "perm", "Permutation (default)"),
      clEnumValEnd), cll::init(OutputType::permutation));
static cll::opt<OutputEdgeType> writeEdgeType("edgeType", cll::desc("Input/Output edge type:"),
    cll::values(
      clEnumValN(OutputEdgeType::void_, "void", "no edge values"),
      clEnumValN(OutputEdgeType::int32, "int32", "32 bit edge values"),
      clEnumValN(OutputEdgeType::int64, "int64", "64 bit edge values"),
      clEnumValEnd), cll::init(OutputEdgeType::void_));
static cll::opt<PseudoAlgo> pseudoAlgo(cll::desc("Psuedo-Peripheral algorithm:"),
    cll::values(
      clEnumVal(simplePseudo, "Simple"),
      clEnumVal(eagerPseudo, "Eager terminiation of full algorithm"),
      clEnumVal(fullPseudo, "Full algorithm"),
      clEnumValEnd), cll::init(fullPseudo));
static cll::opt<bool> anyBFS("anyBFS", cll::desc("Use Any BFS ordering"), cll::init(false));
static cll::opt<bool> printBandwidth("printBandwidth", cll::desc("Print bandwidth"), cll::init(false));
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from. Overrides pseudo-peripheral search."), cll::init(DIST_INFINITY));
static cll::opt<unsigned int> pseudoStartNode("pseudoStartNode",
    cll::desc("Node to pseudo-peripheral search from"), cll::init(0));

namespace {

//****** Work Item and Node Data Definitions ******
struct SNode {
  DistType dist;
  unsigned int degree;
  unsigned int id;
  bool done;
};

typedef galois::graphs::LC_CSR_Graph<SNode, void>
  ::with_no_lockable<true>::type
  ::with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode;

Graph graph;

std::vector<GNode> perm;

std::ostream& operator<<(std::ostream& os, const SNode& n) {
  os << "(id: " << n.id;
  os << ", dist: ";
  if (n.dist == DIST_INFINITY)
    os << "Inf";
  else
    os << n.dist;
  os << ", degree: " << n.degree << ")";
  return os;
}

struct GNodeIndexer: public std::unary_function<GNode,unsigned int> {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, galois::MethodFlag::UNPROTECTED).dist;// >> 2;
  }
};

struct sortDegFn {
  bool operator()(const GNode& lhs, const GNode& rhs) const {
    return
      std::distance(graph.edge_begin(lhs, galois::MethodFlag::UNPROTECTED),
        graph.edge_end(lhs, galois::MethodFlag::UNPROTECTED))
      <
      std::distance(graph.edge_begin(rhs, galois::MethodFlag::UNPROTECTED),
        graph.edge_end(rhs, galois::MethodFlag::UNPROTECTED))
      ;
  }
};

struct UnsignedIndexer: public std::unary_function<unsigned,unsigned> {
  unsigned operator()(unsigned x) const { return x;}
};

struct default_reduce {
  template<typename T>
  void operator()(T& dest, T& src) {
    T::reduce(dest,src);
  }
};

struct BFS {
  struct Result {
    galois::gstl::Vector<size_t> counts;
    size_t max_width;
    GNode source;
    bool complete;

    size_t ecc() { return counts.size() - 1; }
  };

private:
  struct EmptyFunction {
    template<typename T>
    void operator()(T t) { }
    template<typename T>
    void push(T t) { }
  };

  struct BagWorklist {
    galois::InsertBag<GNode>* bag;
    BagWorklist(galois::InsertBag<GNode>* b): bag(b) { }
    void push(const GNode& x) {
      bag->push(x);
    }
  };

  struct AccumUpdate {
    galois::GAccumulator<size_t>* accum;
    AccumUpdate(galois::GAccumulator<size_t>* a): accum(a) { }
    void operator()(DistType x) {
      accum += 1;
    }
  };

  //! Compute BFS levels
  template<bool useContext, typename Worklist, typename Updater>
  struct Process {
    typedef int tt_does_not_need_aborts;

    Worklist wl;
    Updater updater;

    Process(const Worklist& w = Worklist(), const Updater& u = Updater()): wl(w), updater(u) { }

    template<typename Context>
    void operator()(const GNode& n, Context& ctx) {
      SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      DistType newDist = data.dist + 1;
      for (Graph::edge_iterator ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
             ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        DistType oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            updater(newDist);
            if (useContext)
              ctx.push(dst);
            else
              wl.push(dst);
            break;
          }
        }
      }
    }
  };

  typedef Process<true, EmptyFunction, EmptyFunction> UnorderedProcess;
  typedef Process<false, BagWorklist, AccumUpdate> OrderedProcess;

  //! Compute histogram of levels
  template<typename GR>
  struct CountLevels {
    GR& counts;
    bool reset;

    explicit CountLevels(bool r, GR& c): counts(c), reset(r) { }

    void operator()(const GNode& n) const {
    }

    static void reduce(std::deque<size_t>& dest, std::deque<size_t>& src) {
      if (dest.size() < src.size())
        dest.resize(src.size());
      std::transform(src.begin(), src.end(),
          dest.begin(), dest.begin(), std::plus<size_t>());
    }
  };

  static Result unorderedAlgo(GNode source, bool reset) {
    using namespace galois::worklists;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;
    
    Result res;
    res.source = source;

    graph.getData(source).dist = 0;
    galois::for_each(source, UnorderedProcess(), galois::wl<dChunk>());

    struct updater {
      void operator()(std::deque<size_t>& lhs, size_t rhs) {
        if (lhs.size() <= rhs)
          lhs.resize(rhs + 1);
        ++lhs[rhs];
      }
    };



    galois::GVectorPerItemReduce<size_t, std::plus<size_t> > counts;
    galois::do_all(galois::iterate(graph), 
        [&] (const GNode& n) {
          SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          
          assert(data.dist != DIST_INFINITY);
          if (data.dist == DIST_INFINITY)
            return;
          counts.update(data.dist, 1);
          if (reset)
            data.dist = DIST_INFINITY;
          
        });


    res.counts = counts.reduce();
    res.max_width = *std::max_element(res.counts.begin(), res.counts.end());
    res.complete = true;
    return res;
  }

  static Result orderedAlgo(GNode source, bool reset, size_t limit) {
    galois::InsertBag<GNode> wls[2];
    Result res;

    galois::InsertBag<GNode>* cur = &wls[0];
    galois::InsertBag<GNode>* next = &wls[1];

    res.source = source;
    graph.getData(source).dist = 0;
    cur->push(source);

    while (!cur->empty()) {
      galois::GAccumulator<size_t> count;

      galois::for_each(*cur, OrderedProcess(BagWorklist(next), AccumUpdate(&count)));
      res.counts.push_back(count.reduce());
      if (res.counts.back() >= limit) {
        res.complete = next->empty();
        break;
      }
      cur->clear();
      std::swap(cur, next);
    }

    res.max_width = *std::max_element(res.counts.begin(), res.counts.end());
    return res;
  }

public:
  static void initNode(const GNode& n) {
    SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
    data.degree = std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
        graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
    resetNode(n);
  }

  static void resetNode(const GNode& n) {
    SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
    data.dist = DIST_INFINITY;
    data.done = false;
  }

  static void init() {
    galois::do_all(graph, &initNode);
  }

  static void reset() {
    galois::do_all(graph, &resetNode);
  }

    // Compute maximum bandwidth for a given graph
  struct banddiff {
    galois::GAtomic<long int>& maxband;
    galois::GAtomic<long int>& profile;
    std::vector<GNode>& nmap; 

    banddiff(galois::GAtomic<long int>& _mb, galois::GAtomic<long int>& _pr, std::vector<GNode>& _nm) : maxband(_mb), profile(_pr), nmap(_nm) { }

    void operator()(const GNode& source) const {
      long int maxdiff = 0;
      SNode& sdata = graph.getData(source, galois::MethodFlag::UNPROTECTED);

      for (Graph::edge_iterator ii = graph.edge_begin(source, galois::MethodFlag::UNPROTECTED), 
          ei = graph.edge_end(source, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        long int diff = abs(static_cast<long int>(sdata.id) - static_cast<long int>(ddata.id));
        //long int diff = (long int) sdata.id - (long int) ddata.id;
        maxdiff = diff > maxdiff ? diff : maxdiff;
      }

      long int globalmax = maxband;
      profile += maxdiff;

      if (maxdiff > globalmax){
        while (!maxband.cas(globalmax, maxdiff)){
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
    galois::GAtomic<long int> profile = galois::GAtomic<long int>(0);
    std::vector<GNode> nodemap;
    std::vector<bool> visited;
    visited.resize(graph.size(), false);
    nodemap.resize(graph.size());

    //static int count = 0;
    //std::cout << graph.size() << "Run: " << count++ << "\n";

    for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei; ++src) {
      nodemap[graph.getData(*src, galois::MethodFlag::UNPROTECTED).id] = *src;
    }

    //Computation of bandwidth and profile in parallel
    galois::do_all(graph.begin(), graph.end(), banddiff(bandwidth, profile, nodemap));

    unsigned int nactiv = 0;
    unsigned int maxwf = 0;
    unsigned int curwf = 0;
    double mswf = 0.0;

    //Computation of maximum and root-square-mean wavefront. Serial
    for (unsigned int i = 0; i < graph.size(); ++i) {
      for (Graph::edge_iterator ii = graph.edge_begin(nodemap[i], galois::MethodFlag::UNPROTECTED), 
          ei = graph.edge_end(nodemap[i], galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

        GNode neigh = graph.getEdgeDst(ii);
        SNode& ndata = graph.getData(neigh, galois::MethodFlag::UNPROTECTED);

        //std::cerr << "neigh: " << ndata.id << "\n";
        if (visited[ndata.id] == false){
          visited[ndata.id] = true;
          nactiv++;
          //  std::cerr << "val: " << nactiv<< "\n";
        }
      }

      SNode& idata = graph.getData(nodemap[i], galois::MethodFlag::UNPROTECTED);

      if (visited[idata.id] == false) {
        visited[idata.id] = true;
        curwf = nactiv+1;
      } else {
        curwf = nactiv--;
      }

      maxwf = curwf > maxwf ? curwf : maxwf;
      mswf += (double) curwf * curwf;
    }

    mswf = mswf / graph.size();

    std::cout << msg << " Bandwidth: " << bandwidth << "\n";
    std::cout << msg << " Profile: " << profile << "\n";
    std::cout << msg << " Max WF: " << maxwf << "\n";
    std::cout << msg << " Mean-Square WF: " << mswf << "\n";
    std::cout << msg << " RMS WF: " << sqrt(mswf) << "\n";

    //nodemap.clear();
  }

  static void permute(std::vector<GNode>& ordering) {
    std::vector<GNode> nodemap;
    nodemap.resize(graph.size());

    for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei; ++src) {
      nodemap[graph.getData(*src, galois::MethodFlag::UNPROTECTED).id] = *src;
    }

    unsigned int N = ordering.size() - 1;

    for (int i = N; i >= 0; --i) {
      // RCM
      graph.getData(ordering[i], galois::MethodFlag::UNPROTECTED).id = N - i;
      // CM
      //graph.getData(ordering[i], galois::MethodFlag::UNPROTECTED).id = i;
    }
  }

  static Result go(GNode source, bool reset) {
    return unorderedAlgo(source, reset);
  }

  static Result go(GNode source, bool reset, size_t limit) {
    return orderedAlgo(source, reset, limit);
  }
};


/**
 * The eccentricity of vertex v, ecc(v), is the greatest distance from v to any vertex.
 * A peripheral vertex v is one whose distance from some other vertex u is the
 * diameter of the graph: \exists u : dist(v, u) = D. A pseudo-peripheral vertex is a 
 * vertex v that satisfies: \forall u : dist(v, u) = ecc(v) ==> ecc(v) = ecc(u).
 *
 * Simple pseudo-peripheral algorithm:
 *  1. Choose v
 *  2. Among the vertices dist(v, u) = ecc(v), select u with minimal degree
 *  3. If ecc(u) > ecc(v) then
 *       v = u and go to step 2
 *     otherwise
 *       u is a pseudo-peripheral vertex
 */
struct SimplePseudoPeripheral {
  struct min_degree {
    template<typename T>
    T operator()(const T& a, const T& b) const {
      if (!a) return b;
      if (!b) return a;
      if (graph.getData(*a).degree < graph.getData(*b).degree)
        return a;
      else
        return b;
    }
  };

  struct has_dist {
    DistType dist;
    explicit has_dist(DistType d): dist(d) { }
    galois::optional<GNode> operator()(const GNode& a) const {
      if (graph.getData(a).dist == dist)
        return galois::optional<GNode>(a);
      return galois::optional<GNode>();
    }
  };

  static std::pair<BFS::Result, GNode> search(const GNode& start) {
    BFS::Result res = BFS::go(start, false);
    GNode candidate =
      *galois::ParallelSTL::map_reduce(graph.begin(), graph.end(),
          has_dist(res.ecc()), galois::optional<GNode>(), min_degree());
    return std::make_pair(res, candidate);
  }

  static BFS::Result go(GNode source) {
    int searches = 0;

    ++searches;
    std::pair<BFS::Result, GNode> v = search(source);
    while (true) {
      // NB: Leaves graph BFS state for last iteration
      ++searches;
      BFS::reset();
      std::pair<BFS::Result, GNode> u = search(v.second);

      std::cout << "ecc(v) = " << v.first.ecc() << " ecc(u) = " << u.first.ecc() << "\n";
      
      bool better = u.first.ecc() > v.first.ecc();
      v = u;
      if (!better)
        break;
    }

    std::cout << "Selected source: " << graph.getData(v.first.source)
      << " ("
      << "searches: " << searches 
      << ")\n";
    return v.first;
  }
};

/**
 * A more complicated pseudo-peripheral algorithm.
 *
 * Let the width of vertex v be the maximum number of nodes with the same
 * distance from v.
 *
 * Unlike the simple one, instead of picking a minimal degree candidate u,
 * select among some number of candidates U. Here, we select the top n
 * lowest degree nodes who do not share neighborhoods.
 *
 * If there exists a vertex u such that ecc(u) > ecc(v) proceed as in the
 * simple algorithm. 
 *
 * Otherwise, select the u that has least maximum width.
 */
struct PseudoPeripheral {
  struct order_by_degree {
    bool operator()(const GNode& a, const GNode& b) const {
      return graph.getData(a).degree < graph.getData(b).degree;
    }
  };

  //! Collect nodes with dist == d
  struct collect_nodes {
    galois::InsertBag<GNode>& bag;
    size_t dist;
    
    collect_nodes(galois::InsertBag<GNode>& b, size_t d): bag(b), dist(d) { }

    void operator()(const GNode& n) const {
      if (graph.getData(n).dist == dist)
        bag.push(n);
    }
  };

  struct select_candidates {
    static std::deque<GNode> go(unsigned topn, size_t dist) {
      galois::InsertBag<GNode> bag;
      galois::do_all(graph, collect_nodes(bag, dist));

      // Incrementally sort nodes until we find least N who are not neighbors
      // of each other
      std::deque<GNode> nodes;
      std::deque<GNode> result;
      std::copy(bag.begin(), bag.end(), std::back_inserter(nodes));
      size_t cur = 0;
      size_t size = nodes.size();
      size_t delta = topn * 5;

      for (std::deque<GNode>::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ) {
        std::deque<GNode>::iterator mi = ii;
        if (cur + delta < size) {
          std::advance(mi, delta);
          cur += delta;
        } else {
          mi = ei;
          cur = size;
        }

        std::partial_sort(ii, mi, ei, order_by_degree());

        for (std::deque<GNode>::iterator jj = ii; jj != mi; ++jj) {
          GNode n = *jj;

          // Ignore marked neighbors
          if (graph.getData(n).done)
            continue;

          result.push_back(n);
          
          if (result.size() == topn) {
            return result;
          }

          // Mark neighbors
          for (Graph::edge_iterator nn = graph.edge_begin(n), en = graph.edge_end(n); nn != en; ++nn)
            graph.getData(graph.getEdgeDst(nn)).done = true;
        }

        ii = mi;
      }

      return result;
    }
  };

  static std::pair<BFS::Result,std::deque<GNode> > search(const GNode& start, size_t limit, bool computeCandidates) {
    BFS::Result res;
    std::deque<GNode> candidates;

    if (limit == (size_t)-1 || pseudoAlgo == fullPseudo) {
      res = BFS::go(start, false);
      if (computeCandidates)
        candidates = select_candidates::go(5, res.ecc());
      if (res.max_width >= limit)
        res.complete = false;
    } else {
      res = BFS::go(start, false, limit);
      if (res.complete && computeCandidates)
        candidates = select_candidates::go(5, res.ecc());  
    }

    BFS::reset();

    return std::make_pair(res, candidates);
  }

  static BFS::Result go(GNode source) {
    int skips = 0;
    int searches = 0;
    galois::optional<BFS::Result> terminal;

    ++searches;
    std::pair<BFS::Result, std::deque<GNode> > v = search(source, ~0, true);

    while (true) {
      std::cout 
        << "(ecc(v), max_width) =" 
        << " (" << v.first.ecc() << ", " << v.first.max_width << ")"
        << " (ecc(u), max_width(u)) =";

      size_t last = ~0;
      for (auto ii = v.second.begin(), ei = v.second.end(); ii != ei; ++ii) {
        ++searches;
        std::pair<BFS::Result,std::deque<GNode> > u = search(*ii, last, false);

        std::cout << " (" << u.first.ecc() << ", " << u.first.max_width << ")";

        if (!u.first.complete) {
          ++skips;
          continue;
        } else if (u.first.ecc() > v.first.ecc()) {
          v = u;
          terminal = galois::optional<BFS::Result>();
          break;
        } else if (u.first.max_width < last) {
          last = u.first.max_width;
          terminal = galois::optional<BFS::Result>(u.first);
        }
      }

      std::cout << "\n";

      if (terminal)
        break;
      v = search(v.first.source, ~0, true);
    }

    BFS::Result res;
    if (v.first.max_width > terminal->max_width) {
      res = *terminal;
    } else {
      res = v.first;
    }

    std::cout << "Selected source: " << graph.getData(res.source)
      << " (skips: " << skips
      << ", searches: " << searches 
      << ")\n";
    return res;
  }
};


struct CuthillUnordered {
  template<typename C,typename RO, typename WO>
  struct PlaceFn {
    C& counts;
    RO& read_offset;
    WO& write_offset;

    PlaceFn(C& c, RO& r, WO& w): counts(c), read_offset(r), write_offset(w) { }

    void operator()(unsigned me, unsigned int tot) const {
      DistType n = me;
      while (n < counts.size()) {
        unsigned start = read_offset[n];
        unsigned t_wo = write_offset[n+1].data;
        volatile unsigned* endp = (volatile unsigned*)&write_offset[n].data;
        unsigned cend;
        unsigned todo = counts[n];
        while (todo) {
          //spin
          while (start == (cend = *endp)) { galois::substrate::asmPause(); }
          while (start != cend) {
            GNode next = perm[start];
            unsigned t_worig = t_wo;
            //find eligible nodes
            for (Graph::edge_iterator ii = graph.edge_begin(next, galois::MethodFlag::UNPROTECTED),
             ei = graph.edge_end(next, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
              GNode dst = graph.getEdgeDst(ii);
              SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              if (!ddata.done && (ddata.dist == n + 1)) {
                ddata.done = true;
                perm[t_wo] = dst;
                ++t_wo;
              }
            }
            //sort to get cuthill ordering
            std::sort(&perm[t_worig], &perm[t_wo], sortDegFn());
            //output nodes
            galois::substrate::compilerBarrier();
            write_offset[n+1].data = t_wo;
            //  ++read_offset[n];
            //  --level_count[n];
            ++start;
            --todo;
          }
        }
        n += tot;
      }
    }
  };

  template<typename C, typename RO, typename WO>
  static void place_nodes(C& c, RO& read_offset, WO& write_offset) {
    galois::on_each(PlaceFn<C,RO,WO>(c, read_offset, write_offset), galois::loopname("place"));
  }

  template<typename C>
  static void place_nodes(GNode source, C& counts) {
    std::deque<unsigned int> read_offset;
    std::deque<galois::substrate::CacheLineStorage<unsigned int> > write_offset;

    read_offset.push_back(0);
    std::partial_sum(counts.begin(), counts.end(), back_inserter(read_offset));
    write_offset.insert(write_offset.end(), read_offset.begin(), read_offset.end());

    perm[0] = source;
    write_offset[0].data = 1;

    place_nodes(counts, read_offset, write_offset);
  }

  template<typename C>
  static void go(GNode source, C& counts) {
    place_nodes(source, counts);
  }

  static void go(GNode source) {
    BFS::Result res = BFS::go(source, false);
    go(source, res.counts);
  }
};

struct AnyBFSUnordered {
  template<typename C, typename WO>
  struct PlaceFn {
    C& counts;
    WO& write_offset;

    PlaceFn(C& c, WO& w): counts(c), write_offset(w) { }

    void operator()(const GNode& src) const {
      unsigned d = graph.getData(src).dist;
      unsigned wloc = __sync_fetch_and_add(&write_offset[d].data, 1);
      perm[wloc] = src;
    }
  };

  template<typename C, typename RO, typename WO>
  static void place_nodes(C& c, RO& read_offset, WO& write_offset) {
    galois::do_all(graph.begin(), graph.end(), PlaceFn<C,WO>(c, write_offset), galois::loopname("place"));
  }

  template<typename C>
  static void place_nodes(GNode source, C& counts) {
    std::deque<unsigned int> read_offset;
    std::deque<galois::substrate::CacheLineStorage<unsigned int> > write_offset;

    read_offset.push_back(0);
    std::partial_sum(counts.begin(), counts.end(), back_inserter(read_offset));
    write_offset.insert(write_offset.end(), read_offset.begin(), read_offset.end());

    place_nodes(counts, read_offset, write_offset);
  }

  template<typename C>
  static void go(GNode source, C& counts) {
    place_nodes(source, counts);
  }

  static void go(GNode source) {
    BFS::Result res = BFS::go(source, false);
    go(source, res.counts);
  }
};

template<typename EdgeTy>
void writeGraph() {
  galois::graphs::FileGraph origGraph;
  origGraph.fromFileInterleaved<EdgeTy>(filename);
  std::vector<size_t> perm;
  perm.resize(graph.size());
  for (GNode n : graph)
    perm[n] = graph.getData(n).id;

  galois::graphs::FileGraph out;
  galois::graphs::permute<EdgeTy>(origGraph, perm, out);
  std::cout 
    << "Writing permuted graph to " << outputFilename 
    << " (nodes: " << out.size() << " edges: " << out.sizeEdges() << ")\n";
  out.toFile(outputFilename);
}

void writePermutation() {
  std::ofstream file(outputFilename.c_str());
  std::vector<unsigned int> transpose;
  transpose.resize(graph.size());
  for (GNode n : graph)
    transpose[graph.getData(n).id] = n;
  std::cout << "Writing permutation to " << outputFilename << "\n";
  for (unsigned int id : transpose)
    file << id << "\n";
  file.close();
}

} // end anonymous

int main(int argc, char **argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  BFS::Result result; 

  galois::StatTimer itimer("InitTime");
  itimer.start();
  galois::graphs::readGraph(graph, filename); 
  {
    size_t id = 0;
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii){
      if (id == startNode) {
        result.source = *ii;
      }
      graph.getData(*ii).id = id++;
    }
  }
  BFS::init();
  itimer.stop();

  std::cout << "Read " << std::distance(graph.begin(), graph.end()) << " nodes\n";
  perm.resize(std::distance(graph.begin(), graph.end()));

  if (printBandwidth)
    BFS::bandwidth("Initial");

  galois::StatTimer T;
  T.start();

  if (startNode == DIST_INFINITY) {
    galois::StatTimer Tpseudo("PseudoTime");
    Tpseudo.start();
    Graph::iterator ii = graph.begin();
    std::advance(ii, pseudoStartNode);
    result = pseudoAlgo == simplePseudo ? 
      SimplePseudoPeripheral::go(*ii) : PseudoPeripheral::go(*ii);
    Tpseudo.stop();
  }

  galois::StatTimer Tcuthill("CuthillTime");
  Tcuthill.start();
  if (pseudoAlgo == simplePseudo) {
    if (anyBFS)
      AnyBFSUnordered::go(result.source, result.counts);
    else
      CuthillUnordered::go(result.source, result.counts);
  } else {
    if (anyBFS)
      AnyBFSUnordered::go(result.source);
    else
      CuthillUnordered::go(result.source);
  }
  Tcuthill.stop();
  T.stop();

  BFS::permute(perm);
  if (printBandwidth)
    BFS::bandwidth("Permuted");

  if (outputFilename != "") {
    switch (writeType) {
      case OutputType::permutation: writePermutation(); break;
      case OutputType::permutedgraph:
        switch (writeEdgeType) {
          case OutputEdgeType::void_: writeGraph<void>(); break;
          case OutputEdgeType::int32: writeGraph<uint32_t>(); break;
          case OutputEdgeType::int64: writeGraph<uint64_t>(); break;
          default: abort();
        }
        break;
      default: abort();
    }
  }

  std::cout << "done!\n";
  return 0;
}

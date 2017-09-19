#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

//kik 
#include "Galois/Atomic.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <algorithm>

#include <sys/time.h>

//#define FINE_GRAIN_TIMING 
//#define GALOIS_JUNE
//#define PRINT_DEGREE_DISTR

static const char* name = "Sloan's reordering algorithm";
static const char* desc = "Computes a permutation of a matrix according to Sloan's algorithm";
static const char* url = 0;

//****** Command Line Options ******

enum ExecPhase {
  INIT,
  RUN,
  TOTAL,
};

enum Statuses {
  INACTIVE,
  PREACTIVE,
  ACTIVE,
  NUMBERED,
};

enum PseudoAlgo {
  simplePseudo,
  eagerPseudo,
  fullPseudo
};

typedef unsigned int DistType;

static const DistType DIST_INFINITY = std::numeric_limits<DistType>::max() - 1;

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(DIST_INFINITY));
static cll::opt<unsigned int> terminalNode("terminalnode",
    cll::desc("Terminal Node to find distance to"),
    cll::init(DIST_INFINITY));
static cll::opt<int> niter("iter",
    cll::desc("Number of benchmarking iterations"),
    cll::init(1));
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);
static cll::opt<unsigned int> W1("w1",
    cll::desc("First weight"), cll::init(16));
static cll::opt<unsigned int> W2("w2",
    cll::desc("Second weight"), cll::init(1));
static cll::opt<PseudoAlgo> pseudoAlgo(cll::desc("Psuedo-Peripheral algorithm:"),
    cll::values(
      clEnumVal(simplePseudo, "Simple"),
      clEnumVal(eagerPseudo, "Eager terminiation of full algorithm"),
      clEnumVal(fullPseudo, "Full algorithm"),
      clEnumValEnd), cll::init(fullPseudo));

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int id;
  DistType dist;
  unsigned int degree;
  unsigned int status;
  int prio;
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out <<  "(id: " << n.id << " dist: " << n.dist << ")";
  return out;
}

typedef Galois::Graph::LC_Linear_Graph<SNode, void>
  ::with_no_lockable<true>::type
  ::with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode;

Graph graph;

static size_t degree(const GNode& node) { 
  return std::distance(graph.edge_begin(node), graph.edge_end(node));
}

struct UpdateRequest {
  GNode node;
  int prio;

  UpdateRequest(): prio(0) { }
  UpdateRequest(const GNode& N, int P): node(N), prio(P) { }
};

struct UpdateRequestIndexer: public std::unary_function<UpdateRequest,int> {
  int operator()(const UpdateRequest& val) const {
    int p = val.prio;
    return -p;
  }
};

struct UpdateRequestLess {
  bool operator()(const UpdateRequest& a, const UpdateRequest& b) const {
    return a.prio <= b.prio;
  }
};

struct GNodeIndexer: public std::unary_function<GNode,int> {
  int operator()(const GNode& node) const {
    return graph.getData(node, Galois::MethodFlag::UNPROTECTED).prio;
  }
};

struct GNodeLess {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, Galois::MethodFlag::UNPROTECTED).prio < graph.getData(b, Galois::MethodFlag::UNPROTECTED).prio;
  }
};

struct GNodeGreater {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, Galois::MethodFlag::UNPROTECTED).prio > graph.getData(b, Galois::MethodFlag::UNPROTECTED).prio;
  }
};

struct GNodeBefore {
  bool operator()(const GNode& a, const GNode& b) const {
    return (degree(a) < degree(b));
  }
};

struct default_reduce {
  template<typename T>
  void operator()(T& dest, T& src) {
    T::reduce(dest,src);
  }
};

struct BFS {
  struct Result {
    std::deque<size_t> counts;
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
    Galois::InsertBag<GNode>* bag;
    BagWorklist(Galois::InsertBag<GNode>* b): bag(b) { }
    void push(const GNode& x) {
      bag->push(x);
    }
  };

  struct AccumUpdate {
    Galois::GAccumulator<size_t>* accum;
    AccumUpdate(Galois::GAccumulator<size_t>* a): accum(a) { }
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
      SNode& data = graph.getData(n, Galois::MethodFlag::UNPROTECTED);
      DistType newDist = data.dist + 1;
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::UNPROTECTED),
             ei = graph.edge_end(n, Galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
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
      SNode& data = graph.getData(n, Galois::MethodFlag::UNPROTECTED);
      
      assert(data.dist != DIST_INFINITY);
      counts.update(data.dist);
      if (reset)
        data.dist = DIST_INFINITY;
    }

    static void reduce(std::deque<size_t>& dest, std::deque<size_t>& src) {
      if (dest.size() < src.size())
        dest.resize(src.size());
      std::transform(src.begin(), src.end(),
          dest.begin(), dest.begin(), std::plus<size_t>());
    }
  };

  static Result unorderedAlgo(GNode source, bool reset) {
    using namespace Galois::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;
    
    Result res;
    res.source = source;

    graph.getData(source).dist = 0;
    Galois::for_each(source, UnorderedProcess(), Galois::wl<dChunk>());

    struct updater {
      void operator()(std::deque<size_t>& lhs, size_t rhs) {
        if (lhs.size() <= rhs)
          lhs.resize(rhs + 1);
        ++lhs[rhs];
      }
    };
    Galois::GReducible<std::deque<size_t>, updater> counts{updater()};
    Galois::do_all_local(graph, CountLevels<decltype(counts)>(reset, counts));
    res.counts = counts.reduce(CountLevels<decltype(counts)>::reduce);
    res.max_width = *std::max_element(res.counts.begin(), res.counts.end());
    res.complete = true;
    return res;
  }

  static Result orderedAlgo(GNode source, bool reset, size_t limit) {
    Galois::InsertBag<GNode> wls[2];
    Result res;

    Galois::InsertBag<GNode>* cur = &wls[0];
    Galois::InsertBag<GNode>* next = &wls[1];

    res.source = source;
    graph.getData(source).dist = 0;
    cur->push(source);

    while (!cur->empty()) {
      Galois::GAccumulator<size_t> count;

      Galois::for_each_local(*cur, OrderedProcess(BagWorklist(next), AccumUpdate(&count)));
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
    SNode& data = graph.getData(n, Galois::MethodFlag::UNPROTECTED);
    data.degree = std::distance(graph.edge_begin(n, Galois::MethodFlag::UNPROTECTED),
        graph.edge_end(n, Galois::MethodFlag::UNPROTECTED));
    resetNode(n);
  }

  static void resetNode(const GNode& n) {
    SNode& data = graph.getData(n, Galois::MethodFlag::UNPROTECTED);
    data.dist = DIST_INFINITY;
    data.status = INACTIVE;
  }

  static void init() {
    Galois::do_all_local(graph, &initNode);
  }

  static void reset() {
    Galois::do_all_local(graph, &resetNode);
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
    Galois::optional<GNode> operator()(const GNode& a) const {
      if (graph.getData(a).dist == dist)
        return Galois::optional<GNode>(a);
      return Galois::optional<GNode>();
    }
  };

  static std::pair<BFS::Result, GNode> search(const GNode& start) {
    BFS::Result res = BFS::go(start, false);
    GNode candidate =
      *Galois::ParallelSTL::map_reduce(graph.begin(), graph.end(),
          has_dist(res.ecc()), Galois::optional<GNode>(), min_degree());
    return std::make_pair(res, candidate);
  }

  static std::pair<BFS::Result,GNode> go(GNode source) {
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
      source = v.first.source;
      v = u;
      if (!better)
        break;
    }

    std::cout << "Selected source: " << graph.getData(v.first.source)
      << " ("
      << "searches: " << searches 
      << ")\n";
    return std::make_pair(v.first, source);
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
    Galois::InsertBag<GNode>& bag;
    size_t dist;
    
    collect_nodes(Galois::InsertBag<GNode>& b, size_t d): bag(b), dist(d) { }

    void operator()(const GNode& n) const {
      if (graph.getData(n).dist == dist)
        bag.push(n);
    }
  };

  struct select_candidates {
    static std::deque<GNode> go(unsigned topn, size_t dist) {
      Galois::InsertBag<GNode> bag;
      Galois::do_all_local(graph, collect_nodes(bag, dist));

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
          if (graph.getData(n).status != INACTIVE)
            continue;

          result.push_back(n);
          
          if (result.size() == topn) {
            return result;
          }

          // Mark neighbors
          for (Graph::edge_iterator nn = graph.edge_begin(n), en = graph.edge_end(n); nn != en; ++nn)
            graph.getData(graph.getEdgeDst(nn)).status = INACTIVE;
        }

        ii = mi;
      }

      return result;
    }
  };

  static std::pair<BFS::Result,std::deque<GNode> > search(const GNode& start, size_t limit, bool computeCandidates) {
    BFS::Result res;
    std::deque<GNode> candidates;

    if (limit == ~(size_t)0 || pseudoAlgo == fullPseudo) {
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

  static std::pair<BFS::Result,GNode> go(GNode source) {
    int skips = 0;
    int searches = 0;
    Galois::optional<BFS::Result> terminal;

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
          terminal = Galois::optional<BFS::Result>();
          break;
        } else if (u.first.max_width < last) {
          last = u.first.max_width;
          terminal = Galois::optional<BFS::Result>(u.first);
        }
      }

      std::cout << "\n";

      if (terminal)
        break;
      v = search(v.first.source, ~0, true);
    }

    BFS::Result res;
    GNode end;
    if (v.first.max_width > terminal->max_width) {
      res = *terminal;
      end = v.first.source;
    } else {
      res = v.first;
      end = terminal->source;
    }

    std::cout << "Selected source: " << graph.getData(res.source)
      << " (skips: " << skips
      << ", searches: " << searches 
      << ")\n";
    return std::make_pair(res, end);
  }
};

struct Permutation {
  size_t cur;
  std::vector<GNode> perm;

  Permutation(): cur(0) { }

  void print() {
    std::cerr << "Sloan Permutation:\n";

    for (std::vector<GNode>::iterator nit = perm.begin(); nit != perm.end(); nit++) {
      SNode& data = graph.getData(*nit);
      //std::cerr << "[" << data.id << "] level: " << data.dist << " degree: " << data.degree << "\n";
      //std::cerr << data.id + 1 << " (" << data.degree << ") level: " << data.dist << "\n";
      //std::cerr << data.id + 1 << "\n";
      std::cerr << data.id << "\n";
    }
    std::cerr << "\n";
  }

  bool add(GNode x) {
    size_t c = __sync_fetch_and_add(&cur, 1);
    if (c >= perm.size())
      return false;
    perm[c] = x;
    return true;
  }

  void init(size_t x) {
    perm.resize(x);
  }

  void reset() {
    perm.clear();
    cur = 0;
  }

  void rprint() {
    std::cerr << "Reverse Sloan Permutation:\n";
    for(std::vector<GNode>::reverse_iterator nit = perm.rbegin(); nit != perm.rend(); nit++){
      SNode& data = graph.getData(*nit);
      //std::cerr << "[" << data.id << "] level: " << data.dist << " degree: " << data.degree << "\n";
      //std::cerr << data.id + 1 << " (" << data.degree << ") level: " << data.dist << "\n";

      std::cerr << data.id << " (" << degree(*nit) << ") level: " << data.dist << "\n";
    }
    std::cerr << "\n";
  }

  void permute() {
    std::vector<GNode> nodemap;
    nodemap.reserve(graph.size());;

    for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei; ++src) {
      nodemap[graph.getData(*src).id] = *src;
    }

    unsigned int N = perm.size() - 1;

    for(int i = N; i >= 0; --i) {
      //std::cerr << perm[i] << " " << graph.getData(nodemap[permid[i]]).id << " changes to: " << N - i << "\n";
      graph.getData(perm[i]).id = N - i;
    }
  }
};

Permutation perm;

//debugging 
void printAccess(std::string msg){
  std::cerr << msg << " Access Pattern:\n";

  std::vector<unsigned int> temp;

  for (Graph::iterator src = graph.begin(), ei =
      graph.end(); src != ei; ++src) {

    SNode& sdata = graph.getData(*src);

    std::cerr << sdata.id << " connected with (" << degree(*src) << "): ";

    for (Graph::edge_iterator ii = graph.edge_begin(*src, Galois::MethodFlag::UNPROTECTED), 
        ei = graph.edge_end(*src, Galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);

      long int diff = abs(static_cast<long int>(sdata.id) - static_cast<long int>(ddata.id));

      std::cerr << ddata.id << " (" << diff << "), ";
    }

    std::cerr << "\n";
    //std::cerr << data.id << " (" << degree(*src) << ") level: " << data.dist << " reads: " << data.read << " writes: " << data.write << "\n";
    //std::cerr << data.id << " (" << degree(*src) << ") level: " << data.dist << "\n";

    temp.push_back(sdata.id);
  }

  //for(std::vector<unsigned int>::reverse_iterator nit = temp.rbegin(); nit != temp.rend(); nit++)
  for(std::vector<unsigned int>::iterator nit = temp.begin(); nit != temp.end(); nit++){
    std::cerr << *nit + 1 << "\n";
  }
  std::cerr << "\n";
}

void findStartingNode(GNode& starting) {
  unsigned int mindegree = DIST_INFINITY; 

  for (Graph::iterator src = graph.begin(), ei =
      graph.end(); src != ei; ++src) {
    unsigned int nodedegree = degree(*src);

    if(nodedegree < mindegree){
      mindegree = nodedegree;
      starting = *src;
    }
  }

  SNode& data = graph.getData(starting);
  std::cerr << "Starting Node: " << data.id << " degree: " << degree(starting) << "\n";
}

template<typename T>
class GReduceAverage {
  typedef std::pair<T, unsigned> TP;
  struct AVG {
    void operator() (TP& lhs, const TP& rhs) const {
      lhs.first += rhs.first;
      lhs.second += rhs.second;
    }
  };
  Galois::GReducible<std::pair<T, unsigned>, AVG> data;

public:
  void update(const T& _newVal) {
    data.update(std::make_pair(_newVal, 1));
  }

  /**
   * returns the thread local value if in a parallel loop or
   * the final reduction if in serial mode
   */
  const T reduce() {
#ifdef GALOIS_JUNE
    const TP& d = data.get();
#else
    const TP& d = data.reduce();
#endif
    return d.first / d.second;
  }

  void reset(const T& d) {
    data.reset(std::make_pair(d, 0));
  }

  GReduceAverage& insert(const T& rhs) {
#ifdef GALOIS_JUNE
    TP& d = data.get();
#else
    TP& d = data.reduce();
#endif
    d.first += rhs;
    d.second++;
    return *this;
  }
};

//Compute mean distance from the source
struct avg_dist {
  GReduceAverage<unsigned int>& m;
  avg_dist(GReduceAverage<unsigned int>& _m): m(_m) { }

  void operator()(const GNode& n) const {
    if(graph.getData(n).dist < DIST_INFINITY)
      m.update(graph.getData(n).dist);
  }
};

//Compute variance around mean distance from the source
static void variance(unsigned long int mean) {
  unsigned long int n = 0;
  long double M2 = 0.0;
  long double var = 0.0;

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei; ++src) {
    SNode& data = graph.getData(*src);
    if(data.dist < DIST_INFINITY){
      M2 += (data.dist - mean)*(data.dist - mean);
      ++n;
    }
  }

  var = M2/(n-1);
  std::cout << "var: " << var << " mean: " << mean << "\n";
}

struct not_consistent {
  bool operator()(GNode n) const {
    DistType dist = graph.getData(n).dist;
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      DistType ddist = graph.getData(dst).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value for " << graph.getData(dst).id << ": " << ddist << " > " << (dist + 1) << "\n";
  return true;
      }
    }
    return false;
  }
};

struct not_visited {
  bool operator()(GNode n) const {
    DistType dist = graph.getData(n).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr << "unvisited node " << graph.getData(n).id << ": " << dist << " >= INFINITY\n";
      return true;
    }
    //std::cerr << "visited node " << graph.getData(n).id << ": " << dist << "\n";
    return false;
  }
};

struct max_dist {
  Galois::GReduceMax<unsigned int>& m;
  max_dist(Galois::GReduceMax<unsigned int>& _m): m(_m) { }

  void operator()(const GNode& n) const {
    if(graph.getData(n).dist < DIST_INFINITY)
      m.update(graph.getData(n).dist);
  }
};

//! Simple verifier
static bool verify(GNode& source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source: " << graph.getData(source) << " has non-zero dist value\n";
    return false;
  }
  
#ifdef GALOIS_JUNE
  bool okay = Galois::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && Galois::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#else
  bool okay = Galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && Galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#endif

  //if (okay) {
    Galois::GReduceMax<unsigned int> m;
    GReduceAverage<unsigned int> mean;
    Galois::do_all(graph.begin(), graph.end(), max_dist(m));
#ifdef GALOIS_JUNE
    std::cout << "max dist: " << m.get() << "\n";
#else
    std::cout << "max dist: " << m.reduce() << "\n";
#endif
    Galois::do_all(graph.begin(), graph.end(), avg_dist(mean));
    Galois::do_all(graph.begin(), graph.end(), avg_dist(mean));
    std::cout << "avg dist: " << mean.reduce() << "\n";

    variance(mean.reduce());
  //}
  
  return okay;
}

		// Compute maximum bandwidth for a given graph
struct banddiff {

	Galois::GAtomic<long int>& maxband;
	Galois::GAtomic<long int>& profile;
	std::vector<GNode>& nmap; 

	banddiff(Galois::GAtomic<long int>& _mb, Galois::GAtomic<long int>& _pr, std::vector<GNode>& _nm) : maxband(_mb), profile(_pr), nmap(_nm) { }

	void operator()(const GNode& source) const {

		long int maxdiff = 0;
		SNode& sdata = graph.getData(source, Galois::MethodFlag::UNPROTECTED);

		for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::MethodFlag::UNPROTECTED), 
				ei = graph.edge_end(source, Galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

			GNode dst = graph.getEdgeDst(ii);
			SNode& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);

			long int diff = abs(static_cast<long int>(sdata.id) - static_cast<long int>(ddata.id));
			//long int diff = (long int) sdata.id - (long int) ddata.id;
			maxdiff = diff > maxdiff ? diff : maxdiff;
		}

		long int globalmax = maxband;
		profile += maxdiff;

		if(maxdiff > globalmax){
			while(!maxband.cas(globalmax, maxdiff)){
				globalmax = maxband;
				if(!(maxdiff > globalmax))
					break;
			}
		}
	}
};

// Parallel loop for maximum bandwidth computation
static void bandwidth(std::string msg) {
	Galois::GAtomic<long int> bandwidth = Galois::GAtomic<long int>(0);
	Galois::GAtomic<long int> profile = Galois::GAtomic<long int>(0);
	std::vector<GNode> nodemap;
	std::vector<bool> visited;
	visited.reserve(graph.size());;
	visited.resize(graph.size(), false);;
	nodemap.reserve(graph.size());;

	//static int count = 0;
	//std::cout << graph.size() << "Run: " << count++ << "\n";

	for (Graph::iterator src = graph.begin(), ei =
			graph.end(); src != ei; ++src) {
		nodemap[graph.getData(*src, Galois::MethodFlag::UNPROTECTED).id] = *src;
	}

	//Computation of bandwidth and profile in parallel
	Galois::do_all(graph.begin(), graph.end(), banddiff(bandwidth, profile, nodemap));

	unsigned int nactiv = 0;
	unsigned int maxwf = 0;
	unsigned int curwf = 0;
	double mswf = 0.0;

	//Computation of maximum and root-square-mean wavefront. Serial
	for(unsigned int i = 0; i < graph.size(); ++i){

		for (Graph::edge_iterator ii = graph.edge_begin(nodemap[i], Galois::MethodFlag::UNPROTECTED), 
				ei = graph.edge_end(nodemap[i], Galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

			GNode neigh = graph.getEdgeDst(ii);
			SNode& ndata = graph.getData(neigh, Galois::MethodFlag::UNPROTECTED);

			//std::cerr << "neigh: " << ndata.id << "\n";
			if(visited[ndata.id] == false){
				visited[ndata.id] = true;
				nactiv++;
				//	std::cerr << "val: " << nactiv<< "\n";
			}
		}

		SNode& idata = graph.getData(nodemap[i], Galois::MethodFlag::UNPROTECTED);

		if(visited[idata.id] == false){
			visited[idata.id] = true;
			curwf = nactiv+1;
		}
		else
			curwf = nactiv--;

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

//Clear node data to re-execute on specific graph
struct resetNode {
  void operator()(const GNode& n) const {
    graph.getData(n).dist = DIST_INFINITY;
    //graph.getData(n).flag = false;
    //graph.getData(n).bucket.clear();
  }
};

static void resetGraph() {
  Galois::do_all(graph.begin(), graph.end(), resetNode());
  perm.reset();
}

void printDegreeDistribution() {
  std::map<unsigned int, unsigned int> distr;

  for (Graph::iterator n = graph.begin(), ei = graph.end(); n != ei; ++n) {
      distr[degree(*n)]++;
      //std::cerr << graph.getData(*n, Galois::MethodFlag::UNPROTECTED).id << "  " << graph.getData(*n, Galois::MethodFlag::UNPROTECTED).dist << "\n";
  }

  std::cerr << "Degree  Count\n";
  for (std::map<unsigned int, unsigned int>::iterator slot = distr.begin(), ei = distr.end(); slot != ei; ++slot) {
    std::cerr << slot->first << "  " << slot->second << "\n";
  }
}

// Read graph from a binary .gr as dirived from a Matrix Market .mtx using graph-convert
static void readGraph() {
  Galois::Graph::readGraph(graph, filename);

  size_t nnodes = graph.size();
  std::cout << "Read " << nnodes << " nodes\n";
  
  size_t id = 0;
  //bool foundTerminal = false;
  //bool foundSource = false;

  perm.init(nnodes);

  for (Graph::iterator src = graph.begin(), ei = graph.end(); src != ei; ++src, ++id) {
    SNode& node = graph.getData(*src, Galois::MethodFlag::UNPROTECTED);
    node.dist = DIST_INFINITY;
    node.status = INACTIVE;
    node.id = id;
  }
}

struct Sloan {
  std::string name() const { return "Sloan"; }

  struct bfsFn {
    typedef int tt_does_not_need_aborts;

    void operator()(GNode& n, Galois::UserContext<GNode>& ctx) const {
      SNode& data = graph.getData(n, Galois::MethodFlag::UNPROTECTED);

      DistType newDist = data.dist + 1;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::UNPROTECTED),
          ei = graph.edge_end(n, Galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);

        DistType oldDist;
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

    static void go(GNode source) {
      using namespace Galois::WorkList;
      typedef dChunkedFIFO<64> dChunk;
      typedef ChunkedFIFO<64> Chunk;
      typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;

      graph.getData(source).dist = 0;
      Galois::for_each(source, bfsFn(), Galois::loopname("BFS"), Galois::wl<OBIM>());
    }
  };
  
  struct initFn {
    void operator()(const GNode& n) const {
      SNode& data = graph.getData(n, Galois::MethodFlag::UNPROTECTED);
      data.status = INACTIVE;
      data.degree = degree(n);
      data.prio = W1 * data.dist - W2 * (data.degree + 1);
    }

    static void go() {
      Galois::do_all(graph.begin(), graph.end(), initFn());
    }
  };

  struct sloanFn {
    typedef int tt_does_not_need_aborts;
    typedef int tt_needs_parallel_break;

    void operator()(UpdateRequest& next, Galois::UserContext<UpdateRequest>& ctx) const {
      unsigned int prev_status;
      GNode parent = next.node;
      SNode& pdata = graph.getData(parent, Galois::MethodFlag::UNPROTECTED);

      while ((prev_status = pdata.status) != NUMBERED) {
        if (__sync_bool_compare_and_swap(&pdata.status, prev_status, NUMBERED)) {
          break;
        }
      }

      if (prev_status == NUMBERED)
        return;

      if (!perm.add(parent)) {
        ctx.breakLoop();
        return;
      }

      if (prev_status == PREACTIVE) {
        for (Graph::edge_iterator ii = graph.edge_begin(parent, Galois::MethodFlag::UNPROTECTED),
            ei = graph.edge_end(parent, Galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

          GNode child = graph.getEdgeDst(ii);
          SNode& cdata = graph.getData(child, Galois::MethodFlag::UNPROTECTED);

          if (cdata.status == NUMBERED)
            continue;

          if (cdata.status == INACTIVE && __sync_bool_compare_and_swap(&cdata.status, INACTIVE, PREACTIVE)) {
            ; // TODO
          }
          int prio = __sync_add_and_fetch(&cdata.prio, W2);
          ctx.push(UpdateRequest(child, prio));
        }
      }

      for (Graph::edge_iterator ii = graph.edge_begin(parent, Galois::MethodFlag::UNPROTECTED),
          ei = graph.edge_end(parent, Galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

        GNode child = graph.getEdgeDst(ii);
        SNode& cdata = graph.getData(child, Galois::MethodFlag::UNPROTECTED);

        if (cdata.status != PREACTIVE)
          continue;

        if (cdata.status == PREACTIVE && __sync_bool_compare_and_swap(&cdata.status, PREACTIVE, ACTIVE)) {
          ; // TODO
        }
        int prio = __sync_add_and_fetch(&cdata.prio, W2);

        ctx.push(UpdateRequest(child, prio));

        for (Graph::edge_iterator ij = graph.edge_begin(child, Galois::MethodFlag::UNPROTECTED),
            ej = graph.edge_end(child, Galois::MethodFlag::UNPROTECTED); ij != ej; ++ij) {
          GNode grandchild = graph.getEdgeDst(ij);
          SNode& gdata = graph.getData(grandchild, Galois::MethodFlag::UNPROTECTED);

          if (gdata.status == NUMBERED)
            continue;

          if (gdata.status == INACTIVE && __sync_bool_compare_and_swap(&gdata.status, INACTIVE, PREACTIVE)) {
            ; // TODO
          }
          int prio = __sync_add_and_fetch(&gdata.prio, W2);
          ctx.push(UpdateRequest(grandchild, prio));
        }
      }
    }

    static void go(GNode source) {
      using namespace Galois::WorkList;
      typedef dChunkedLIFO<64> dChunk;
      typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

      graph.getData(source).status = PREACTIVE;

      Galois::for_each(UpdateRequest(source, graph.getData(source).prio), sloanFn(), Galois::loopname("Sloan"), Galois::wl<OBIM>());
    }
  };

  static void go(GNode source, GNode terminal, bool skipBfs) {
#ifdef FINE_GRAIN_TIMING
    Galois::TimeAccumulator vTmain[6]; 
    vTmain[0] = Galois::TimeAccumulator();
    vTmain[1] = Galois::TimeAccumulator();
    vTmain[2] = Galois::TimeAccumulator();
    vTmain[3] = Galois::TimeAccumulator();

    vTmain[0].start();
#endif

    if (!skipBfs)
      bfsFn::go(terminal);
    //verify(terminal);
#ifdef FINE_GRAIN_TIMING
    vTmain[0].stop();
    vTmain[1].start();
#endif
    initFn::go();
#ifdef FINE_GRAIN_TIMING
    vTmain[1].stop();
    vTmain[2].start();
#endif
    Galois::StatTimer T;
    T.start();
    sloanFn::go(source);
    T.stop();
#ifdef FINE_GRAIN_TIMING
    vTmain[2].stop();
#endif

#ifdef FINE_GRAIN_TIMING
    std::cerr << "bfsFn: " << vTmain[0].get() << "\n";
    std::cerr << "initFn: " << vTmain[1].get() << "\n";
    std::cerr << "sloanFn: " << vTmain[2].get() << "\n";
    //std::cout << "& " << vTmain[0].get() << " & \\multicolumn{2} {c|} {" << vTmain[1].get() << "} & " << vTmain[2].get() << " & " << vTmain[0].get() + vTmain[1].get()  + vTmain[2].get() << "\n";
#endif
    //printSloan();
  }
};

template<typename AlgoTy>
void run() {
  AlgoTy algo;
  GNode source, terminal;

  int maxThreads = numThreads; 
  std::vector<Galois::TimeAccumulator> vT(maxThreads+2); 

  //Measure time to read graph
  vT[INIT] = Galois::TimeAccumulator();
  vT[INIT].start();

  readGraph();

  bandwidth("Initial");

  vT[INIT].stop();

  std::cout << "Init: " << vT[INIT].get() 
    << " ( " << (double) vT[INIT].get() / 1000 << " seconds )\n";

  //Measure total computation time to read graph
  vT[TOTAL].start();

  if ((startNode < DIST_INFINITY && startNode >= graph.size()) 
      || (terminalNode < DIST_INFINITY && terminalNode >= graph.size())) {
    std::cerr 
      << "failed to set terminal: " << terminalNode 
      << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  if (startNode < DIST_INFINITY) {
    source = graph.begin()[startNode];
  } 

  if (terminalNode < DIST_INFINITY) {
    terminal = graph.begin()[terminalNode];
  }
  
  bool skipFirstBfs = false;

  if (startNode == DIST_INFINITY || terminalNode == DIST_INFINITY) {
    if (startNode == DIST_INFINITY)
      source = *graph.begin();

    Galois::StatTimer Tpseudo("PseudoTime");
    Tpseudo.start();
    std::pair<BFS::Result, GNode> result;
    if (pseudoAlgo == simplePseudo) { 
      result = SimplePseudoPeripheral::go(source);
      skipFirstBfs = true;
      source = result.second;
      terminal = result.first.source;
    } else {
      result = PseudoPeripheral::go(source);
      source = result.first.source;
      terminal = result.second;
    }
    Tpseudo.stop();
  }
  
  // Execution with the specified number of threads
  vT[RUN] = Galois::TimeAccumulator();

  std::cout << "Running " << algo.name() << " version with "
    << numThreads << " threads for " << niter << " iterations\n";

  for(int i = 0; i < niter; i++){
    vT[RUN].start();

    algo.go(source, terminal, i == 0 && skipFirstBfs);

    vT[RUN].stop();

    perm.permute();
    bandwidth("Permuted");

    std::cout << "Iteration " << i
      << " numthreads: " << numThreads
      << " " << vT[RUN].get() << "\n";

    if(i != niter-1)
      resetGraph();
  }

  std::cout << "Final time numthreads: " << numThreads << " " << vT[RUN].get() << "\n";
  std::cout << "Avg time numthreads: " << numThreads << " " << vT[RUN].get() / niter << "\n";

#ifdef PRINT_DEGREE_DISTR
  printDegreeDistribution();
#endif

  vT[TOTAL].stop();

  std::cout << "Total with threads: " << numThreads
    << " " << vT[TOTAL].get() << " ( " << (double) vT[TOTAL].get() / 1000 << " seconds )\n";

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

  run<Sloan>();

  return 0;
}

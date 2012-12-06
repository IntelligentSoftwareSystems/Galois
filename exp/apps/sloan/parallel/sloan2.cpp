#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/Graph.h"
#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
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

#define FINE_GRAIN_TIMING 
//#define GALOIS_JUNE
//#define PRINT_DEGREE_DISTR

#define W1 1               //default weight for the distance in the Sloan algorithm
#define W2 2               //default weight for the degree in the Sloan algorithm

static const char* name = "Sloan's reordering algorithm";
static const char* desc = "Computes a permutation of a matrix according to Sloan's algorithm\n";
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

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(0));
static cll::opt<unsigned int> terminalNode("terminalnode",
    cll::desc("Terminal Node to find distance to"),
    cll::init(0));
static cll::opt<int> niter("iter",
    cll::desc("Number of benchmarking iterations"),
    cll::init(5));
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int id;
  unsigned int dist;
  unsigned int degree;
  unsigned int status;
  int prio;
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out <<  "(dist: " << n.dist << ")";
  return out;
}

typedef Galois::Graph::LC_Linear_Graph<SNode, void> Graph;
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
    return graph.getData(node, Galois::NONE).prio;
  }
};

struct GNodeLess {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, Galois::NONE).prio < graph.getData(b, Galois::NONE).prio;
  }
};

struct GNodeGreater {
  bool operator()(const GNode& a, const GNode& b) const {
    return graph.getData(a, Galois::NONE).prio > graph.getData(b, Galois::NONE).prio;
  }
};

struct GNodeBefore {
  bool operator()(const GNode& a, const GNode& b) const {
    return (degree(a) < degree(b));
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
std::priority_queue<UpdateRequest, std::vector<UpdateRequest>, UpdateRequestLess> pq;
//std::set<UpdateRequest, std::greater<UpdateRequest> > pq;
//std::multiset<UpdateRequest, UpdateRequestGreater> pq;


//debugging 
static void printAccess(std::string msg){
  std::cerr << msg << " Access Pattern:\n";

  std::vector<unsigned int> temp;

  for (Graph::iterator src = graph.begin(), ei =
      graph.end(); src != ei; ++src) {

    SNode& sdata = graph.getData(*src);

    std::cerr << sdata.id << " connected with (" << degree(*src) << "): ";

    for (Graph::edge_iterator ii = graph.edge_begin(*src, Galois::NONE), 
        ei = graph.edge_end(*src, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int diff = abs(sdata.id - ddata.id);

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

static void findStartingNode(GNode& starting) {
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
  GReduceAverage<unsigned long int>& m;
  avg_dist(GReduceAverage<unsigned long int>& _m): m(_m) { }

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
    unsigned int dist = graph.getData(n).dist;
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(dst).dist;
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
    unsigned int dist = graph.getData(n).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr << "unvisited node " << graph.getData(n).id << ": " << dist << " >= INFINITY\n";
      return true;
    }
    //std::cerr << "visited node " << graph.getData(n).id << ": " << dist << "\n";
    return false;
  }
};

struct max_dist {
  Galois::GReduceMax<unsigned long int>& m;
  max_dist(Galois::GReduceMax<unsigned long int>& _m): m(_m) { }

  void operator()(const GNode& n) const {
    if(graph.getData(n).dist < DIST_INFINITY)
      m.update(graph.getData(n).dist);
  }
};

//! Simple verifier
static bool verify(GNode& source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
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
    Galois::GReduceMax<unsigned long int> m;
    GReduceAverage<unsigned long int> mean;
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

  Galois::GAtomic<unsigned int>& maxband;
  banddiff(Galois::GAtomic<unsigned int>& _mb): maxband(_mb) { }

  void operator()(const GNode& source) const {

    SNode& sdata = graph.getData(source, Galois::NONE);
    for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::NONE), 
         ei = graph.edge_end(source, Galois::NONE); ii != ei; ++ii) {

      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int diff = abs(sdata.id - ddata.id);
      unsigned int max = maxband;

      if(diff > max){
        while(!maxband.cas(max, diff)){
          max = maxband;
          if(!diff > max)
            break;
        }
      }
    }
  }
};

// Compute maximum bandwidth for a given graph
struct profileFn {

  Galois::GAtomic<unsigned int>& sum;
  profileFn(Galois::GAtomic<unsigned int>& _s): sum(_s) { }

  void operator()(const GNode& source) const {

    unsigned int max = 0;
    SNode& sdata = graph.getData(source, Galois::NONE);

    for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::NONE), 
        ei = graph.edge_end(source, Galois::NONE); ii != ei; ++ii) {

      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int diff = abs(sdata.id - ddata.id);

      max = (diff > max) ? diff : max;
    }

    sum += (max + 1);
  }
};

// Parallel loop for maximum bandwidth computation
static void bandwidth(std::string msg) {
    Galois::GAtomic<unsigned int> maxband = Galois::GAtomic<unsigned int>(0);
    Galois::do_all(graph.begin(), graph.end(), banddiff(maxband));
    std::cout << msg << " Bandwidth: " << maxband << "\n";
}

// Parallel loop for maximum bandwidth computation
static void profile(std::string msg) {
    Galois::GAtomic<unsigned int> prof = Galois::GAtomic<unsigned int>(0);
    Galois::do_all(graph.begin(), graph.end(), profileFn(prof));
    std::cout << msg << " Profile: " << prof << "\n";
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

static void printDegreeDistribution() {
  std::map<unsigned int, unsigned int> distr;

  for (Graph::iterator n = graph.begin(), ei = graph.end(); n != ei; ++n) {
      distr[degree(*n)]++;
      //std::cerr << graph.getData(*n, Galois::NONE).id << "  " << graph.getData(*n, Galois::NONE).dist << "\n";
  }

  std::cerr << "Degree  Count\n";
  for (std::map<unsigned int, unsigned int>::iterator slot = distr.begin(), ei = distr.end(); slot != ei; ++slot) {
    std::cerr << slot->first << "  " << slot->second << "\n";
  }
}

// Read graph from a binary .gr as dirived from a Matrix Market .mtx using graph-convert
static void readGraph(GNode& source, GNode& terminal) {
  graph.structureFromFile(filename);

  source = *graph.begin();
  terminal = *graph.begin();

  size_t nnodes = graph.size();
  std::cout << "Read " << nnodes << " nodes\n";
  
  size_t id = 0;
  bool foundTerminal = false;
  bool foundSource = false;

  perm.init(nnodes);

  for (Graph::iterator src = graph.begin(), ei =
      graph.end(); src != ei; ++src) {

    SNode& node = graph.getData(*src, Galois::NONE);
    node.dist = DIST_INFINITY;
    node.id = id;

    if (id == startNode) {
      source = *src;
      foundSource = true;
    } 
    if (id == terminalNode) {
      foundTerminal = true;
      terminal = *src;
    }
    ++id;
  }

/*
  if(startNode == DIST_INFINITY){
    findStartingNode(source);
    foundSource = true;
  }
  */

  if (!foundTerminal || !foundSource) {
    std::cerr 
      << "failed to set terminal: " << terminalNode 
      << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
}

struct Sloan {
  std::string name() const { return "Sloan"; }

  struct bfsFn {
    typedef int tt_does_not_need_aborts;

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

    static void go(GNode source) {
      using namespace GaloisRuntime::WorkList;
      typedef dChunkedFIFO<64> dChunk;
      typedef ChunkedFIFO<64> Chunk;
      typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;

      graph.getData(source).dist = 0;
      Galois::for_each<OBIM>(source, bfsFn(), "BFS");
    }
  };
  
  struct initFn {
    void operator()(const GNode& n) const {
      SNode& data = graph.getData(n, Galois::NONE);
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
      SNode& pdata = graph.getData(parent, Galois::NONE);

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
        for (Graph::edge_iterator ii = graph.edge_begin(parent, Galois::NONE),
            ei = graph.edge_end(parent, Galois::NONE); ii != ei; ++ii) {

          GNode child = graph.getEdgeDst(ii);
          SNode& cdata = graph.getData(child, Galois::NONE);

          if (cdata.status == NUMBERED)
            continue;

          if (cdata.status == INACTIVE && __sync_bool_compare_and_swap(&cdata.status, INACTIVE, PREACTIVE)) {
            ; // TODO
          }
          int prio = __sync_add_and_fetch(&cdata.prio, W2);
          ctx.push(UpdateRequest(child, prio));
        }
      }

      for (Graph::edge_iterator ii = graph.edge_begin(parent, Galois::NONE),
          ei = graph.edge_end(parent, Galois::NONE); ii != ei; ++ii) {

        GNode child = graph.getEdgeDst(ii);
        SNode& cdata = graph.getData(child, Galois::NONE);

        if (cdata.status != PREACTIVE)
          continue;

        if (cdata.status == PREACTIVE && __sync_bool_compare_and_swap(&cdata.status, PREACTIVE, ACTIVE)) {
          ; // TODO
        }
        int prio = __sync_add_and_fetch(&cdata.prio, W2);

        ctx.push(UpdateRequest(child, prio));

        for (Graph::edge_iterator ij = graph.edge_begin(child, Galois::NONE),
            ej = graph.edge_end(child, Galois::NONE); ij != ej; ++ij) {
          GNode grandchild = graph.getEdgeDst(ij);
          SNode& gdata = graph.getData(grandchild, Galois::NONE);

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
      using namespace GaloisRuntime::WorkList;
      typedef dChunkedLIFO<4> dChunk;
      typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;

      graph.getData(source).status = PREACTIVE;

      Galois::for_each<OBIM>(UpdateRequest(source, graph.getData(source).prio), sloanFn(), "Sloan");
    }
  };

  static void go(GNode source, GNode terminal) {
#ifdef FINE_GRAIN_TIMING
    Galois::TimeAccumulator vTmain[6]; 
    vTmain[0] = Galois::TimeAccumulator();
    vTmain[1] = Galois::TimeAccumulator();
    vTmain[2] = Galois::TimeAccumulator();
    vTmain[3] = Galois::TimeAccumulator();

    vTmain[0].start();
#endif

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
    sloanFn::go(source);
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
  Galois::TimeAccumulator vT[maxThreads+2]; 

  //Measure time to read graph
  vT[INIT] = Galois::TimeAccumulator();
  vT[INIT].start();

  readGraph(source, terminal);

  bandwidth("Initial");
  profile("Initial");

  //std::cout << "original bandwidth: " << boost::bandwidth(*bgraph) << std::endl;
  //std::cout << "original profile: " << boost::profile(*bgraph) << std::endl;
  //std::cout << "original max_wavefront: " << boost::max_wavefront(*bgraph) << std::endl;
  //std::cout << "original aver_wavefront: " << boost::aver_wavefront(*bgraph) << std::endl;
  //std::cout << "original rms_wavefront: " << boost::rms_wavefront(*bgraph) << std::endl;

  vT[INIT].stop();

  std::cout << "Init: " << vT[INIT].get() 
    << " ( " << (double) vT[INIT].get() / 1000 << " seconds )\n";

  //Measure total computation time to read graph
  vT[TOTAL].start();

  // Execution with the specified number of threads
  vT[RUN] = Galois::TimeAccumulator();

  std::cout << "Running " << algo.name() << " version with "
    << numThreads << " threads for " << niter << " iterations\n";

  for(int i = 0; i < niter; i++){
    vT[RUN].start();

    algo.go(source, terminal);

    vT[RUN].stop();

    perm.permute();
    bandwidth("Permuted");
    profile("Permuted");

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

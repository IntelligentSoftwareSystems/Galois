/** Maximum Cardinality Matching in Bipartite Graphs -*- C++ -*-
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
 * Maximum cardinality matching in bipartite graphs. For more information see
 * 
 * K. Mehlhorn and S. Naeher. LEDA: A Platform for Combinatorial and Geometric
 * Computing. Cambridge University Press, 1999.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/FileGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace cll = llvm::cl;

static const char* name = "Maximum cardinality matching in bipartite graphs";
static const char* desc =
  "Computes maximum cardinality matching in bipartite graphs. "
  "A matching of G is a subset of edges that do not share an endpoint. "
  "The maximum cardinality matching is the matching with the most number of edges.";
static const char* url = "bipartite_mcm";

enum MatchingAlgo {
  ffAlgo,
  abmpAlgo
};

enum ExecutionType {
  serial,
  parallel
};

enum InputType {
  generated,
  fromFile
};

static cll::opt<MatchingAlgo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(ffAlgo, "Ford-Fulkerson augmenting paths"),
      clEnumVal(abmpAlgo, "Alt-Blum-Mehlhorn-Paul"),
      clEnumValEnd), cll::init(abmpAlgo));
static cll::opt<ExecutionType> executionType(cll::desc("Choose execution type:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(parallel, "Parallel"),
      clEnumValEnd), cll::init(parallel));
static cll::opt<InputType> inputType("inputType", cll::desc("Input type:"),
    cll::values(
      clEnumVal(generated, "Generated"),
      clEnumVal(fromFile, "From file"),
      clEnumValEnd), cll::init(fromFile));
static cll::opt<int> N("n", cll::desc("Size of each set of nodes in generated input"), cll::init(100));
static cll::opt<int> numEdges("numEdges", cll::desc("Number of edges in generated input"), cll::init(1000));
static cll::opt<int> numGroups("numGroups", cll::desc("Number of groups in generated input"), cll::init(10));
static cll::opt<int> seed("seed", cll::desc("Random seed for generated input"), cll::init(0));
static cll::opt<std::string> inputFilename("file", cll::desc("Input graph"), cll::init(""));
static cll::opt<bool> runIteratively("runIteratively", cll::desc("After finding matching, removed matched edges and repeat"), cll::init(false));

//******************************** Common ************************

typedef std::atomic<int> sharedEdgeData;

template<typename G>
struct GraphTypes {
  typedef typename G::GraphNode GraphNode;
  typedef std::pair<GraphNode,GraphNode> Edge;
  typedef std::vector<Edge> Matching;
};

struct BaseNode {
  size_t id;
  int degree;
  bool covered;
  bool free;
  bool reachable;  // for preparing node cover
  BaseNode(size_t i = -1): id(i), degree(0), covered(false), free(true), reachable(false) { }
  void reset() {
    degree = 0;
    covered = false;
    free = true;
    reachable = false;
  }
};

template <typename G>
void markReachable(G& g, const typename G::GraphNode& root, const size_t numA) {
  typedef typename G::GraphNode GraphNode;

  std::deque<GraphNode> queue;
  queue.push_back(root);

  while (!queue.empty()) {
    GraphNode cur = queue.front();
    queue.pop_front();
    if (g.getData(cur).reachable)
      continue;
    g.getData(cur).reachable = true;
    for (auto ii : g.edges(cur)) {
      if ((cur < numA) != *g.getEdgeData(ii)) {
        GraphNode dst = g.getEdgeDst(ii);
        queue.push_back(dst);
      }
    }
  }
}

template<typename G, template<typename, bool> class Algo>
void prepareForVerifier(G &g, typename GraphTypes<G>::Matching *matching, size_t numA) {
  typedef typename GraphTypes<G>::Edge Edge;
  typedef typename GraphTypes<G>::Matching Matching;
  typedef typename G::GraphNode GraphNode;

  for (size_t src = numA; src < g.size(); src++) {
    for (auto ii : g.edges(src)) {
      if (*g.getEdgeData(ii)){
        GraphNode dst = g.getEdgeDst(ii);
        matching->push_back(Edge(src, dst));
      }
    }
  }

  for (size_t i = 0; i < numA; i++) {
    if (g.getData(i).free) {
      markReachable<G>(g, i, numA);
    }
  }

  for (typename Matching::iterator ii = matching->begin(), ei = matching->end(); ii != ei; ++ii) {
    if (g.getData(ii->first).reachable) {
      // Reachable from a free node in A
      g.getData(ii->first).covered = true;
    } else {
      g.getData(ii->second).covered = true;
    }
  }
}

//********************** FF Algorithm **************************

struct FFNode: public BaseNode {
  int pred;
  bool reached;
  FFNode(size_t i = -1): BaseNode(i), pred(-1), reached(false) { }
  void reset() {
    BaseNode::reset();
    reached = false;
    pred = -1;
  }
};

//! Switch between concurrent and serial instances
template<typename T1,typename T2,bool B> struct InstanceWrapper;
template<typename T1,typename T2>
struct InstanceWrapper<T1,T2,true> {
  T1& m_t1;
  T2& m_t2;
  typedef T2 Type;
  InstanceWrapper(T1& t1, T2& t2): m_t1(t1), m_t2(t2) { }
  T2& get() { return m_t2; }
};
template<typename T1,typename T2>
struct InstanceWrapper<T1,T2,false> {
  T1& m_t1;
  T2& m_t2;
  typedef T1 Type;
  InstanceWrapper(T1& t1, T2& t2): m_t1(t1), m_t2(t2) { }
  T1& get() { return m_t1; }
};

//! Switch between concurrent and serial types
template<typename T1,typename T2,bool B> struct TypeWrapper;
template<typename T1,typename T2>
struct TypeWrapper<T1,T2,true> {
  typedef T2 Type;
};
template<typename T1,typename T2>
struct TypeWrapper<T1,T2,false> {
  typedef T1 Type;
};


//! Matching algorithm of Ford and Fulkerson
template<typename G, bool Concurrent>
struct MatchingFF {
  typedef typename G::GraphNode GraphNode;
  typedef typename G::node_data_type node_data_type;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename GraphTypes<G>::Edge Edge;

  typedef std::vector<Edge> SerialRevs;
  typedef std::vector<GraphNode> SerialReached;

  typedef std::vector<Edge, typename galois::PerIterAllocTy::rebind<Edge>::other> ParallelRevs;
  typedef std::vector<GraphNode,
          typename galois::PerIterAllocTy::rebind<GraphNode>::other> ParallelReached;

  typedef InstanceWrapper<SerialRevs,ParallelRevs,Concurrent> RevsWrapper;
  typedef InstanceWrapper<SerialReached,ParallelReached,Concurrent> ReachedWrapper;

  typedef std::deque<GraphNode, typename galois::PerIterAllocTy::rebind<GraphNode>::other> Queue;
  typedef std::vector<GraphNode, typename galois::PerIterAllocTy::rebind<GraphNode>::other> Preds;
  
  static const galois::MethodFlag flag = Concurrent ? galois::MethodFlag::WRITE : galois::MethodFlag::UNPROTECTED;

  static const bool canRunIteratively = true;

  std::string name() { return std::string(Concurrent ? "Concurrent" : "Serial") + " Ford-Fulkerson"; }

  template <typename C>
  bool findAugmentingPath(G& g, const GraphNode& root, C& ctx,
      typename RevsWrapper::Type& revs, typename ReachedWrapper::Type& reached, size_t numA) {
    Queue queue(ctx.getPerIterAlloc());
    Preds preds(ctx.getPerIterAlloc());

    // Order matters between (1) and (2)
    g.getData(root, flag).reached = true; // (1)
    reached.push_back(root); // (2)

    queue.push_back(root);

    while (!queue.empty()) {
      GraphNode src = queue.front();
      queue.pop_front();

      for (auto ii : g.edges(src, galois::MethodFlag::UNPROTECTED)) {
        if ((src < numA) == *g.getEdgeData(ii)) {
          // This is an incoming edge, so there's no need to process it.
          continue;
        }

        GraphNode dst = g.getEdgeDst(ii);
        node_data_type& ddst = g.getData(dst, flag);
        if (ddst.reached)
          continue;

        ddst.reached = true;
        reached.push_back(dst);

        ddst.pred = preds.size();
        preds.push_back(src);

        if (ddst.free) {
          // Fail-safe point modulo ``reached'' which is handled separately
          ddst.free = false;
          GraphNode cur = dst;
          while (cur != root) {
            auto pred = preds[g.getData(cur, galois::MethodFlag::UNPROTECTED).pred];
            revs.push_back(Edge(pred, cur));
            cur = pred;
          }
          return true;
        } else {
          //assert(std::distance(g.edge_begin(dst), g.edge_end(dst)) == 1);
          for (auto jj : g.edges(dst, galois::MethodFlag::UNPROTECTED)) {
            auto edge = g.getEdgeData(jj);
            if ((dst < numA) == *edge) {
              continue;
            }

            GraphNode cur = g.getEdgeDst(jj);

            g.getData(cur, flag).pred = preds.size();
            preds.push_back(dst);

            g.getData(cur, galois::MethodFlag::UNPROTECTED).reached = true;
            reached.push_back(cur);
 
            queue.push_back(cur);
          }
        }
      }
    }
    return false;
  }

  //! Makes sure that ``reached'' to properly reset even if we get aborted
  struct ReachedCleanup {
    G& g;
    typename ReachedWrapper::Type& reached;

    ReachedCleanup(G& g, typename ReachedWrapper::Type& r): g(g), reached(r) { }
    
    ~ReachedCleanup() {
      cleanup();
    }

    virtual void release() { cleanup(); }

    void cleanup() {
      // In non-concurrent case, we can continue reusing reached
      if (Concurrent)
        clear();
    }

    void clear() {
      for (typename ReachedWrapper::Type::iterator ii = reached.begin(), ei = reached.end(); ii != ei; ++ii) {
        assert(g.getData(*ii, galois::MethodFlag::UNPROTECTED).reached);
        g.getData(*ii, galois::MethodFlag::UNPROTECTED).reached = false;
      }
      reached.clear();
    }
  };

  template <typename C>
  void propagate(G& g, const GraphNode& src, C& ctx,
      typename RevsWrapper::Type& revs, typename ReachedWrapper::Type& reached, size_t numA) {

    ReachedCleanup cleanup(g, reached);

    if (findAugmentingPath(g, src, ctx, revs, reached, numA)) {
      g.getData(src, galois::MethodFlag::UNPROTECTED).free = false;

      // Reverse edges in augmenting path
      for (typename RevsWrapper::Type::iterator jj = revs.begin(), ej = revs.end(); jj != ej; ++jj) {
        sharedEdgeData *edge_data;
        bool found = false;
        for (auto kk : g.edges(jj->first, galois::MethodFlag::UNPROTECTED)) {
          if (g.getEdgeDst(kk) == jj->second) {
            edge_data = g.getEdgeData(kk);
            found = true;
            break;
          }
        }
        assert(found);
        assert((jj->first < numA) != *edge_data);
        // Reverse the edge by flipping the shared flag.
        edge_data->fetch_xor(true, std::memory_order_acq_rel);
      }
      revs.clear();

      cleanup.clear();
    }
  }


  void operator()(G& g, size_t numA) {
    SerialRevs revs;
    SerialReached reached;

    galois::setActiveThreads(Concurrent ? numThreads : 1);
    galois::for_each(galois::iterate(size_t(0), numA),
        [&, outer=this] (const GraphNode& node, auto& ctx) {
          if (!g.getData(node, flag).free)
            return;

          ParallelRevs parallelRevs(ctx.getPerIterAlloc());
          ParallelReached parallelReached(ctx.getPerIterAlloc());

          outer->propagate(g, node, ctx,
              RevsWrapper(revs, parallelRevs).get(),
              ReachedWrapper(reached, parallelReached).get(),
              numA);
        },
        galois::loopname("MatchingFF"),
        galois::per_iter_alloc(),
        galois::wl<galois::worklists::dChunkedFIFO<32> >());
  }
};


//********************** ABMP Algorithm **************************

struct ABMPNode: public FFNode {
  unsigned layer;
  int next;
  ABMPNode(size_t i = -1): FFNode(i), layer(0), next(0) { }
  void reset() {
    FFNode::reset();
    layer = 0;
    next = 0;
  }
};

//! Matching algorithm of Alt, Blum, Mehlhorn and Paul
template<typename G, bool Concurrent>
struct MatchingABMP {
  typedef typename G::GraphNode GraphNode;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename G::node_data_type node_data_type;
  typedef typename GraphTypes<G>::Edge Edge;
  typedef std::vector<Edge, typename galois::PerIterAllocTy::rebind<Edge>::other> Revs;
  typedef std::pair<GraphNode,unsigned> WorkItem;

  static const galois::MethodFlag flag = Concurrent ? galois::MethodFlag::WRITE : galois::MethodFlag::UNPROTECTED;

  static const bool canRunIteratively = true;

  std::string name() { 
    return std::string(Concurrent ? "Concurrent" : "Serial") + " Alt-Blum-Mehlhorn-Paul"; 
  }

  sharedEdgeData *nextEdge(G& g, const GraphNode& src, GraphNode& next, size_t numA) {
    node_data_type& dsrc = g.getData(src, galois::MethodFlag::UNPROTECTED);
    unsigned l = dsrc.layer - 1;

    // Start search where we last left off
    edge_iterator ii = g.edge_begin(src, galois::MethodFlag::UNPROTECTED);
    edge_iterator ei = g.edge_end(src, galois::MethodFlag::UNPROTECTED);
    assert(dsrc.next <= std::distance(ii, ei));
    std::advance(ii, dsrc.next);
    for (; ii != ei && (src < numA == *g.getEdgeData(ii) || g.getData(g.getEdgeDst(ii), flag).layer != l); ++ii, ++dsrc.next) { ; }

    if (ii == ei) {
      return nullptr;
    } else {
      next = g.getEdgeDst(ii);
      return g.getEdgeData(ii);
    }
  }

  //! Returns true if we've added a new element
  //TODO: better name here
  template <typename C>
  bool propagate(G& g, const GraphNode& root, C& ctx, size_t numA) {
    Revs revs(ctx.getPerIterAlloc());

    GraphNode cur = root;

    g.getData(root, flag);

    while (true) {
      GraphNode next;
      if (g.getData(cur, galois::MethodFlag::UNPROTECTED).free && g.getData(cur, galois::MethodFlag::UNPROTECTED).layer == 0) {
        assert(g.getData(root, galois::MethodFlag::UNPROTECTED).free);
        // (1) Breakthrough
        g.getData(cur, galois::MethodFlag::UNPROTECTED).free = g.getData(root, galois::MethodFlag::UNPROTECTED).free = false;

        // Reverse edges in augmenting path
        for (auto ii = revs.begin(), ei = revs.end(); ii != ei; ++ii) {
          // Reverse directions of edges in revs.
          // No need to lock shared edge data here.
          // It was already locked previously.
          bool found = false;
          for (auto kk : g.edges(ii->first, galois::MethodFlag::UNPROTECTED)) {
            if (g.getEdgeDst(kk) == ii->second) {
              found = true;
              auto edge_flag = g.getEdgeData(kk);
              edge_flag->fetch_xor(true, std::memory_order_acq_rel);
              break;
            }
          }
          assert(found);
        }
        //revs.clear();
        if (revs.size() > 1024) {
          std::cout << "WARNING: allocating large amounts in parallel: " 
            << revs.size() << "elements\n";
        }
        return false;
      }
      sharedEdgeData *edge = nextEdge(g, cur, next, numA);
      if (edge != nullptr) {
        // (2) Advance
        revs.push_back(Edge(cur, next));
        cur = next;
      } else {
        // (3) Retreat
        unsigned& layer = g.getData(cur, galois::MethodFlag::UNPROTECTED).layer;
        layer += 2;
        g.getData(cur, galois::MethodFlag::UNPROTECTED).next = 0;
        if (revs.empty()) {
          ctx.push(std::make_pair(cur, layer));
          return true;
        }
        cur = revs.back().first;
        revs.pop_back();
      }
    }
  }

  void operator()(G& g, size_t numA) {
    galois::StatTimer t("serial");
    t.start();
    std::vector<WorkItem> initial;
    for (size_t i = 0; i < numA; i++) {
      g.getData(i).layer = 1;
      if (g.getData(i).free) {
        initial.push_back(std::make_pair(i, 1));
      }
    }
    t.stop();

    unsigned maxLayer = (unsigned) (0.1*sqrt(std::distance(g.begin(), g.end())));
    // size_t size = initial.size();
    galois::setActiveThreads(Concurrent ? numThreads : 1);
    
    using namespace galois::worklists;

    auto indexer = [] (const WorkItem& n) {
      return n.second;
    };


    typedef dChunkedFIFO<1024> dChunk;
    typedef OrderedByIntegerMetric<decltype(indexer),dChunk> OBIM;
    
    galois::for_each(galois::iterate(initial),
        [&, outer=this] (const WorkItem& item, auto& ctx) {
          unsigned curLayer = item.second;
          if (curLayer > maxLayer) {
            //std::cout << "Reached max layer: " << curLayer << "\n";
            ctx.breakLoop();
            return;
          }
          //if (size <= 50 * curLayer) {
          //  std::cout << "Reached min size: " << size << "\n";
          //  ctx.breakLoop();
          //}
          if (!outer->propagate(g, item.first, ctx, numA)) {
            //__sync_fetch_and_add(&size, -1);
          }
        }, 
        galois::per_iter_alloc(),
        galois::parallel_break(),
        galois::loopname("MatchingABMP"),
        galois::wl<OBIM>(indexer));

    t.start();
    MatchingFF<G,false> algo;
    //std::cout << "Switching to " << algo.name() << "\n";
    algo(g, numA);
    t.stop();
  }
};

// ******************* Verification ***************************

template<typename G>
struct Verifier {
  typedef typename G::GraphNode GraphNode;
  typedef typename G::node_data_type node_data_type;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename GraphTypes<G>::Matching Matching;

  bool hasCoveredNeighbors(G& g, const GraphNode& src) {
    for (auto ii : g.edges(src)) {
      GraphNode dst = g.getEdgeDst(ii);
      if (!g.getData(dst).covered)
        return false;
    }
    return true;
  }

  bool operator()(G& g, const Matching& matching, size_t numA) {
    for (auto ii : matching) {
      g.getData(ii.first).degree++;
      g.getData(ii.second).degree++;
    }
    size_t count = 0;
    for (auto n : g) {
      auto &dii = g.getData(n);
      if (dii.degree > 1) {
        std::cerr << "Error: not a matching, node " << dii.id << " incident to " << dii.degree << " edges\n";
        return false;
      }
      if (dii.covered) {
        count ++;
      }
      if (!(dii.covered || hasCoveredNeighbors(g, n))) {
        std::cerr << "Error: not a node cover, node " << dii.id
          << " with degree " << dii.degree << " not covered nor incident to covered node\n";
        return false;
      }
    }
    if (count != matching.size()) {
      std::cerr << "Error: matching is different than node cover " << matching.size() << " vs " << count << "\n";
      return false;
    }
    return true;
  }
};

/**
 * Generate a random bipartite graph as used in LEDA evaluation and
 * refererenced in [CGM+97]. Nodes are divided into numGroups groups of size
 * numA/numGroups each. Each node in A has degree d = numEdges/numA and the
 * edges out of a node in group i of A go to random nodes in groups i+1 and
 * i-1  of B. If numGroups == 0, just randomly assign nodes of A to nodes of
 * B.
 */
template<typename G>
decltype(auto) generateRandomInput(int &numA, int &numB, int numEdges, int numGroups, int seed, G& g) {
  typedef typename G::edge_data_type edge_data_type;
  
  std::cout 
    << "numGroups: " << numGroups
    << " seed: " << seed
    << "\n";

  galois::graphs::FileGraphWriter p;
  p.setNumNodes(numA + numB);
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(galois::LargeArray<edge_data_type>::size_of::value);

  // Build an array for the shared edge data between nodes.
  // LargeArray default initializes its contents when create is called.
  galois::LargeArray<sharedEdgeData> bidirectional_edge_data;
  bidirectional_edge_data.create(numEdges);
  // Create an array of addresses to the shared edge data.
  galois::LargeArray<edge_data_type> edgeData;
  edgeData.create(numEdges);

  int current_bidirectional_index = 0;

  for (int phase = 0; phase < 2; ++phase) {
    if (phase == 0)
      p.phase1();
    else
      p.phase2();

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(0, 1);

    assert(numA > 0 && numB > 0);

    int d = numEdges/numA;
    if (numGroups > numA)
      numGroups = numA;
    if (numGroups > numB)
      numGroups = numB;

    int count = 0;
    if (numGroups > 0) {
      int aSize = numA/numGroups;
      int bSize = numB/numGroups;

      for (int ii = 0; ii < numA; ++ii, ++count) {
        int group = count/aSize;
        if (group == numGroups)
          break;
        int base1 = group == 0 ? (numGroups-1)*bSize : (group-1)*bSize;
        int base2 = group == numGroups-1 ? 0 : (group+1)*bSize;
        for (int i = 0; i < d; ++i) {
          int b = dist(gen) < 0.5 ? base1 : base2;
          int off = (int)(dist(gen) * (bSize-1));
          if (phase == 0) {
            p.incrementDegree(ii);
          } else {
            assert(!bidirectional_edge_data[current_bidirectional_index]);
            edgeData.set(p.addNeighbor(ii, b + off + numA), &bidirectional_edge_data[current_bidirectional_index++]);
          }
        }
      }
    }

    int r = numEdges - count*d;
    while (r--) {
      int ind_a = (int)(dist(gen)*(numA-1));
      int ind_b = (int)(dist(gen)*(numB-1));
      if (phase == 0) {
        p.incrementDegree(ind_a);
      } else {
        assert(!bidirectional_edge_data[current_bidirectional_index]);
        edgeData.set(p.addNeighbor(ind_a, ind_b + numA), &bidirectional_edge_data[current_bidirectional_index++]);
      }
    }
  }
  // Check that the bidirectional edge data has been exhausted.
  // Otherwise, something has gone wrong and the wrong number of edges was created/initialized somehow.
  assert(current_bidirectional_index == numEdges);

  auto *rawEdgeData = p.finish<edge_data_type>();
  std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), rawEdgeData);
  galois::graphs::FileGraphWriter q;
  galois::graphs::makeSymmetric<edge_data_type>(p, q);
  galois::graphs::readGraph(g, q);
  return bidirectional_edge_data;
}

/**
 * Read bipartite graph from file.
 *
 * Assumes
 *  (1) nodes in set A have edges while nodes in set B don't
 *  (2) nodes in set A are the first numA nodes (followed by nodes in set B)
 */
template<typename G>
void readInput(const std::string& filename, G& g) {
  galois::graphs::readGraph(g, filename);
}

template<template<typename,bool> class Algo, typename G>
size_t countMatching(G& g, size_t numA) {
  size_t count = 0;
  for (size_t n = numA; n < g.size(); n++) {
    for (auto edge : g.edges(n)) {
      if (*g.getEdgeData(edge)) {
        count += 1;
      }
    }
  }
  return count;
}


template<template<typename,bool> class Algo, typename G, bool Concurrent>
void start(int N, int numEdges, int numGroups) {
  typedef Algo<G,Concurrent> A;

  A algo;
  G g;
  galois::LargeArray<sharedEdgeData> bidirectional_edge_data;
  size_t numA, numB;

  if (runIteratively && !algo.canRunIteratively)
    GALOIS_DIE("algo does not support iterative execution");

  switch (inputType) {
    case generated: bidirectional_edge_data = generateRandomInput(N, N, numEdges, numGroups, seed, g);
         numA = N; numB = N; break;
    //case fromFile: readInput(inputFilename, g); break;
    default: GALOIS_DIE("unknown input type");
  }

  /*size_t id = 0;
  for (auto n : g) {
    g.getData(n).id = id++;
    if (g.edge_begin(n) != g.edge_end(n))
      g.A.push_back(n);
    else
      g.B.push_back(n);
  }
  */

  std::cout 
    << "numA: " << numA
    << " numB: " << numB
    << "\n";

  std::cout << "Starting " << algo.name() << "\n";

  galois::StatTimer t;

  while (true) {
    t.start();
    algo(g, numA);
    t.stop();

    if (!skipVerify) {
      typename GraphTypes<G>::Matching matching;
      prepareForVerifier<G,Algo>(g, &matching, numA);
      if (!Verifier<G>()(g, matching, numA)) {
        GALOIS_DIE("Verification failed");
      } else {
        std::cout << "Verification successful.\n";
      }
    }

    size_t matchingSize = countMatching<Algo>(g, numA);
    std::cout << "Matching of cardinality: " << matchingSize << "\n";

    if (!runIteratively || matchingSize == 0)
      break;
  }
}


template<bool Concurrent>
void start() {
  switch (algo) {
    case abmpAlgo:
      start<MatchingABMP, galois::graphs::LC_CSR_Graph<ABMPNode, sharedEdgeData*>, Concurrent>(N, numEdges, numGroups); break;
    case ffAlgo:
      start<MatchingFF, galois::graphs::LC_CSR_Graph<FFNode, sharedEdgeData*>, Concurrent>(N, numEdges, numGroups); break;
    default:
      GALOIS_DIE("unknown algo");
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  switch (executionType) {
    case serial: start<false>(); break;
    case parallel: start<true>(); break;
    default:
      GALOIS_DIE("unknown execution type");
  }

  return 0;
}

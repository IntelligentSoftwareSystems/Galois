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

// TODO(ddn): Needs a graph implementation that supports reversing edges more efficiently

#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/FileGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#ifdef GALOIS_USE_EXP
#include "Galois/PriorityScheduling.h"
#endif

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Maximum cardinality matching in bipartite graphs";
static const char* desc =
  "A matching of G is a subset of edges that do not share an endpoint. The "
  "maximum cardinality matching is the matching with the most number of edges.";
static const char* url = "bipartite_mcm";

enum MatchingAlgo {
  pfpAlgo,
  ffAlgo,
  abmpAlgo
};

enum ExecutionType {
  serial,
  parallel
};

static cll::opt<MatchingAlgo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(pfpAlgo, "Preflow-push"),
      clEnumVal(ffAlgo, "Ford-Fulkerson augmenting paths"),
      clEnumVal(abmpAlgo, "Alt-Blum-Mehlhorn-Paul"),
      clEnumValEnd), cll::init(abmpAlgo));
static cll::opt<ExecutionType> executionType(cll::desc("Choose execution type:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(parallel, "Parallel"),
      clEnumValEnd), cll::init(parallel));
static cll::opt<int> N(cll::Positional, cll::desc("<N>"), cll::Required);
static cll::opt<int> numEdges(cll::Positional, cll::desc("<numEdges>"), cll::Required);
static cll::opt<int> numGroups(cll::Positional, cll::desc("<numGroups>"), cll::Required);
static cll::opt<int> seed(cll::Positional, cll::desc("<seed>"), cll::Required);

template<typename NodeTy,typename EdgeTy>
struct BipartiteGraph: public Galois::Graph::FirstGraph<NodeTy,EdgeTy,true> {
  typedef Galois::Graph::FirstGraph<NodeTy,EdgeTy,true> Super;
  typedef std::vector<typename Super::GraphNode> NodeList;
  typedef NodeTy node_type;
  typedef EdgeTy edge_type;

  NodeList A;
  NodeList B;

  void addNode(const typename Super::GraphNode& n, bool isA, Galois::MethodFlag mflag = Galois::ALL) {
    if (isA) {
      A.push_back(n);
    } else {
      B.push_back(n);
    }
    Super::addNode(n, mflag);
  }

  void addNode(const typename Super::GraphNode& n, Galois::MethodFlag mflag = Galois::ALL) {
    Super::addNode(n, mflag);
  }
};

//******************************** Common ************************

template<typename G, template<typename,bool> class Algo> struct Exists {
  bool operator()(G& g, const typename G::edge_iterator&) { return true; }
};

template<typename G>
struct GraphTypes {
  typedef typename G::GraphNode GraphNode;
  typedef std::pair<GraphNode,GraphNode> Edge;
  typedef std::vector<Edge> Matching;
  template<template<typename,bool> class Algo> struct Functions {
    typedef Exists<G,Algo> ExistsFn;
  };
};

struct BaseNode {
  size_t id;
  int degree;
  bool covered;
  bool free;
  bool reachable;  // for preparing node cover
  BaseNode(): id(-1) { }
  BaseNode(size_t i): id(i), degree(0), covered(false), free(true), reachable(false) { }
};

template<typename G>
struct MarkReachable {
  typedef typename G::GraphNode GraphNode;
  typedef typename G::edge_iterator edge_iterator;

  void operator()(G& g, const GraphNode& root) {
    std::deque<GraphNode> queue;
    queue.push_back(root);

    while (!queue.empty()) {
      GraphNode cur = queue.front();
      queue.pop_front();
      if (g.getData(cur).reachable)
        continue;
      g.getData(cur).reachable = true;
      for (edge_iterator ii = g.edge_begin(cur), ei = g.edge_end(cur); ii != ei; ++ii) {
        GraphNode dst = g.getEdgeDst(ii);
        queue.push_back(dst);
      }
    }
  }
};

template<typename G,template<typename,bool> class Algo>
struct PrepareForVerifier {
  typedef typename GraphTypes<G>::Edge Edge;
  typedef typename GraphTypes<G>::Matching Matching;
  typedef typename G::GraphNode GraphNode;
  typedef typename G::NodeList NodeList;
  typedef typename G::node_type node_type;
  typedef typename G::edge_iterator edge_iterator;

  void operator()(G& g, Matching* matching) {
    typename GraphTypes<G>::template Functions<Algo>::ExistsFn exists;

    for (typename NodeList::iterator src = g.B.begin(), esrc = g.B.end(); src != esrc; ++src) {
      for (edge_iterator ii = g.edge_begin(*src), ei = g.edge_end(*src); ii != ei; ++ii) {
        GraphNode dst = g.getEdgeDst(ii);
        if (exists(g, ii)) {
          matching->push_back(Edge(*src, dst));
        }
      }
    }

    for (typename NodeList::iterator ii = g.A.begin(), ei = g.A.end(); ii != ei; ++ii) {
      if (g.getData(*ii).free)
        MarkReachable<G>()(g, *ii);
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
};

//********************** FF Algorithm **************************

struct FFNode: public BaseNode {
  int pred;
  bool reached;
  FFNode(): BaseNode() { }
  FFNode(size_t i): BaseNode(i), pred(-1), reached(false)  { }
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
  typedef typename G::NodeList NodeList;
  typedef typename G::node_type node_type;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename GraphTypes<G>::Edge Edge;

  typedef std::vector<Edge> SerialRevs;
  typedef std::vector<GraphNode> SerialReached;

  typedef std::vector<Edge, typename Galois::PerIterAllocTy::rebind<Edge>::other> ParallelRevs;
  typedef std::vector<GraphNode,
          typename Galois::PerIterAllocTy::rebind<GraphNode>::other> ParallelReached;

  typedef InstanceWrapper<SerialRevs,ParallelRevs,Concurrent> RevsWrapper;
  typedef InstanceWrapper<SerialReached,ParallelReached,Concurrent> ReachedWrapper;

  typedef std::deque<GraphNode, typename Galois::PerIterAllocTy::rebind<GraphNode>::other> Queue;
  typedef std::vector<GraphNode, typename Galois::PerIterAllocTy::rebind<GraphNode>::other> Preds;
  
  static const Galois::MethodFlag flag = Concurrent ? Galois::CHECK_CONFLICT : Galois::NONE;

  std::string name() { return std::string(Concurrent ? "Concurrent" : "Serial") + " Ford-Fulkerson"; }

  bool findAugmentingPath(G& g, const GraphNode& root, Galois::UserContext<GraphNode>& ctx,
      typename RevsWrapper::Type& revs, typename ReachedWrapper::Type& reached) {
    Queue queue(ctx.getPerIterAlloc());
    Preds preds(ctx.getPerIterAlloc());

    // Order matters between (1) and (2)
    g.getData(root, flag).reached = true; // (1)
    reached.push_back(root); // (2)

    queue.push_back(root);

    while (!queue.empty()) {
      GraphNode src = queue.front();
      queue.pop_front();

      for (edge_iterator ii = g.edge_begin(src, flag), ei = g.edge_end(src, flag); ii != ei; ++ii) {
        GraphNode dst = g.getEdgeDst(ii);
        node_type& ddst = g.getData(dst, Galois::NONE);
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
            GraphNode pred = preds[g.getData(cur, Galois::NONE).pred];
            revs.push_back(Edge(pred, cur));
            cur = pred;
          }
          return true;
        } else {
          assert(std::distance(g.edge_begin(dst), g.edge_end(dst)) == 1);
          for (edge_iterator jj = g.edge_begin(dst, flag), ej = g.edge_end(dst, flag); jj != ej; ++jj) {
            GraphNode cur = g.getEdgeDst(jj);

            g.getData(cur, Galois::NONE).pred = preds.size();
            preds.push_back(dst);

            g.getData(cur, Galois::NONE).reached = true;
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

    ReachedCleanup(G& _g, typename ReachedWrapper::Type& r): g(_g), reached(r) { }
    
    ~ReachedCleanup() {
      // In non-concurrent case, we can continue reusing reached
      if (Concurrent)
        clear();
    }

    void clear() {
      for (typename ReachedWrapper::Type::iterator ii = reached.begin(), ei = reached.end(); ii != ei; ++ii) {
        assert(g.getData(*ii, Galois::NONE).reached);
        g.getData(*ii, Galois::NONE).reached = false;
      }
      reached.clear();
    }
  };

  void operator()(G& g, const GraphNode& src, Galois::UserContext<GraphNode>& ctx,
      typename RevsWrapper::Type& revs, typename ReachedWrapper::Type& reached) {

    ReachedCleanup cleanup(g, reached);

    if (findAugmentingPath(g, src, ctx, revs, reached)) {
      g.getData(src, Galois::NONE).free = false;

      // Reverse edges in augmenting path
      for (typename RevsWrapper::Type::iterator jj = revs.begin(), ej = revs.end(); jj != ej; ++jj) {
        g.removeEdge(jj->first, g.findEdge(jj->first, jj->second, Galois::NONE), Galois::NONE);
        g.addEdge(jj->second, jj->first, Galois::NONE);
      }
      revs.clear();

      cleanup.clear();
    }
  }

  //! Main entry point for Galois::for_each
  struct Process {
    typedef int tt_needs_per_iter_alloc;
    MatchingFF<G,Concurrent>& parent;
    G& g;
    SerialRevs& serialRevs;
    SerialReached& serialReached;

    Process(MatchingFF<G,Concurrent>& _parent, G& _g, SerialRevs& revs, SerialReached& reached):
      parent(_parent), g(_g), serialRevs(revs), serialReached(reached) { }

    void operator()(const GraphNode& node, Galois::UserContext<GraphNode>& ctx) {
      if (!g.getData(node, flag).free)
        return;

      ParallelRevs parallelRevs(ctx.getPerIterAlloc());
      ParallelReached parallelReached(ctx.getPerIterAlloc());

      parent(g, node, ctx,
          RevsWrapper(serialRevs, parallelRevs).get(),
          ReachedWrapper(serialReached, parallelReached).get());
    }
  };

  void operator()(G& g) {
    SerialRevs revs;
    SerialReached reached;

    Galois::setActiveThreads(Concurrent ? numThreads : 1);
    Galois::for_each(g.A.begin(), g.A.end(), Process(*this, g, revs, reached));
  }
};


//********************** ABMP Algorithm **************************

struct ABMPNode: public FFNode {
  unsigned layer;
  int next;
  ABMPNode(): FFNode() { }
  ABMPNode(size_t i): FFNode(i), layer(0), next(0) { }
};

//! Matching algorithm of Alt, Blum, Mehlhorn and Paul
template<typename G, bool Concurrent>
struct MatchingABMP {
  typedef typename G::NodeList NodeList;
  typedef typename G::GraphNode GraphNode;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename G::node_type node_type;
  typedef typename GraphTypes<G>::Edge Edge;
  typedef std::vector<Edge, typename Galois::PerIterAllocTy::rebind<Edge>::other> Revs;
  typedef std::pair<GraphNode,unsigned> WorkItem;

  static const Galois::MethodFlag flag = Concurrent ? Galois::CHECK_CONFLICT : Galois::NONE;

  struct Indexer: public std::unary_function<const WorkItem&,unsigned> {
    unsigned operator()(const WorkItem& n) const {
      return n.second;
    }
  };

  struct Less: public std::binary_function<const WorkItem&,const WorkItem&,bool> {
    bool operator()(const WorkItem& n1, const WorkItem& n2) const {
      if (n1.second < n2.second) return true;
      if (n1.second > n2.second) return false;
      return n1.first < n2.first;
    }
  };

  struct Greater: public std::binary_function<const WorkItem&,const WorkItem&,bool> {
    bool operator()(const WorkItem& n1, const WorkItem& n2) const {
      if (n1.second > n2.second) return true;
      if (n1.second < n2.second) return false;
      return n1.first > n2.first;
    }
  };

  std::string name() { 
    return std::string(Concurrent ? "Concurrent" : "Serial") + " Alt-Blum-Mehlhorn-Paul"; 
  }

  bool nextEdge(G& g, const GraphNode& src, GraphNode& next) {
    node_type& dsrc = g.getData(src, Galois::NONE);
    unsigned l = dsrc.layer - 1;

    // Start search where we last left off
    edge_iterator ii = g.edge_begin(src, flag);
    std::advance(ii, dsrc.next);
    edge_iterator ei = g.edge_end(src, flag);
    for (; ii != ei && g.getData(g.getEdgeDst(ii), Galois::NONE).layer != l;
        ++ii, ++dsrc.next) {
      ;
    }

    if (ii == ei) {
      return false;
    } else {
      next = g.getEdgeDst(ii);
      return true;
    }
  }

  //! Returns true if we've added a new element
  bool operator()(G& g, const GraphNode& root, Galois::UserContext<WorkItem>& ctx) {
    Revs revs(ctx.getPerIterAlloc());

    GraphNode cur = root;

    while (true) {
      GraphNode next;
      if (g.getData(cur, Galois::NONE).free && g.getData(cur, Galois::NONE).layer == 0) {
        assert(g.getData(root, Galois::NONE).free);
        // (1) Breakthrough
        g.getData(cur, Galois::NONE).free = g.getData(root, Galois::NONE).free = false;
        
        // Reverse edges in augmenting path
        for (typename Revs::iterator ii = revs.begin(), ei = revs.end(); ii != ei; ++ii) {
          g.removeEdge(ii->first, g.findEdge(ii->first, ii->second, Galois::NONE), Galois::NONE);
          g.addEdge(ii->second, ii->first, Galois::NONE);
        }
        //revs.clear();
        if (revs.size() > 1024) {
          std::cout << "WARNING: allocating large amounts in parallel: " 
            << revs.size() << "elements\n";
        }
        return false;
      } else if (nextEdge(g, cur, next)) {
        // (2) Advance
        revs.push_back(Edge(cur, next));
        cur = next;
      } else {
        // (3) Retreat
        unsigned& layer = g.getData(cur, Galois::NONE).layer;
        layer += 2;
        g.getData(cur, Galois::NONE).next = 0;
        if (revs.empty()) {
          ctx.push(std::make_pair(cur, layer));
          return true;
        }
        cur = revs.back().first;
        revs.pop_back();
      }
    }
  }

  struct Process {
    typedef int tt_needs_parallel_break;
    typedef int tt_needs_per_iter_alloc;
    MatchingABMP<G,Concurrent>& parent;
    G& g;
    unsigned& maxLayer;
    size_t& size;

    Process(MatchingABMP<G,Concurrent>& p, G& _g, unsigned& m, size_t& s):
      parent(p), g(_g), maxLayer(m), size(s) { }
    
    void operator()(const WorkItem& item, Galois::UserContext<WorkItem>& ctx) {
      unsigned curLayer = item.second;
      if (curLayer > maxLayer) {
        std::cout << "Reached max layer: " << curLayer << "\n";
        ctx.breakLoop();
        return;
      }
      //if (size <= 50 * curLayer) {
      //  std::cout << "Reached min size: " << size << "\n";
      //  ctx.breakLoop();
      //}
      if (!parent(g, item.first, ctx)) {
        //__sync_fetch_and_add(&size, -1);
      }
    }
  };

  void operator()(G& g) {
    Galois::StatTimer t("serial");
    t.start();
    std::vector<WorkItem> initial;
    for (typename NodeList::iterator ii = g.A.begin(), ei = g.A.end(); ii != ei; ++ii) {
      g.getData(*ii).layer = 1;
      if (g.getData(*ii).free)
        initial.push_back(std::make_pair(*ii, 1));
    }
    t.stop();

    unsigned maxLayer = (unsigned) (0.1*sqrt(g.size()));
    size_t size = initial.size();
    Galois::setActiveThreads(Concurrent ? numThreads : 1);
    
    using namespace GaloisRuntime::WorkList;

    typedef ChunkedFIFO<1024> Chunk;
    typedef dChunkedFIFO<1024> dChunk;
    typedef OrderedByIntegerMetric<Indexer,dChunk> OBIM;
    
#ifdef GALOIS_USE_EXP
    Exp::PriAuto<1024,Indexer,OBIM,Less,Greater>::for_each(
	 initial.begin(), initial.end(), Process(*this, g, maxLayer, size));
#else
    Galois::for_each<OBIM>(initial.begin(), initial.end(), Process(*this, g, maxLayer, size));
#endif
    
    t.start();
    MatchingFF<G,false> algo;
    std::cout << "Switching to " << algo.name() << "\n";
    algo(g);
    t.stop();
  }
};

// *************************** MaxFlow Algorithm *******************************
struct MFNode: public BaseNode {
  size_t excess;
  unsigned height;
  int current;
  MFNode(): BaseNode() { }
  MFNode(size_t i): BaseNode(i), excess(0), height(1), current(0) { }
};

struct MFEdge {
  int cap;
  MFEdge(): cap(1) { }
  MFEdge(int c): cap(c) { }
};


//! Matching via reduction to maxflow
template<typename G, bool Concurrent>
struct MatchingMF {
  typedef typename G::NodeList NodeList;
  typedef typename G::GraphNode GraphNode;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename G::iterator iterator;
  typedef typename G::node_type node_type;
  typedef typename G::edge_type edge_type;
  static const Galois::MethodFlag flag = Concurrent ? Galois::CHECK_CONFLICT : Galois::NONE;
  /**
   * Beta parameter the original Goldberg algorithm to control when global
   * relabeling occurs. For comparison purposes, we keep them the same as
   * before, but it is possible to achieve much better performance by adjusting
   * the global relabel frequency.
   */
  static const int BETA = 12;
  /**
   * Alpha parameter the original Goldberg algorithm to control when global
   * relabeling occurs. For comparison purposes, we keep them the same as
   * before, but it is possible to achieve much better performance by adjusting
   * the global relabel frequency.
   */
  static const int ALPHA = 6;

  std::string name() { return std::string(Concurrent ? "Concurrent" : "Serial") + " Max Flow"; }

  void reduceCapacity(edge_type& edge1, edge_type& edge2, int amount) {
    edge1.cap -= amount;
    edge2.cap += amount;
  }

  bool discharge(G& g, const GraphNode& src, Galois::UserContext<GraphNode>& ctx,
      const GraphNode& source, const GraphNode& sink, unsigned numNodes) {
    node_type& node = g.getData(src, flag);
    //unsigned prevHeight = node.height;
    bool relabeled = false;

    if (node.excess == 0) {
      return false;
    }

    while (true) {
      Galois::MethodFlag f = relabeled ? Galois::NONE : flag;
      bool finished = false;
      int current = 0;

      for (edge_iterator ii = g.edge_begin(src, f), ei = g.edge_end(src, f); ii != ei; ++ii, ++current) {
        GraphNode dst = g.getEdgeDst(ii);
        edge_type& edge = g.getEdgeData(ii);
        if (edge.cap == 0 || current < node.current) 
          continue;

        node_type& dnode = g.getData(dst, Galois::NONE);
        if (node.height - 1 != dnode.height) 
          continue;

        // Push flow
        int amount = std::min(static_cast<int>(node.excess), edge.cap);
        reduceCapacity(edge, g.getEdgeData(g.findEdge(dst, src, Galois::NONE)), amount);

        // Only add once
        if (dst != sink && dst != source && dnode.excess == 0) 
          ctx.push(dst);
        
        node.excess -= amount;
        dnode.excess += amount;
        
        if (node.excess == 0) {
          finished = true;
          node.current = current;
          break;
        }
      }

      if (finished)
        break;

      relabel(g, src, numNodes);
      relabeled = true;

      //prevHeight = node.height;
    }

    return relabeled;
  }

  void relabel(G& g, const GraphNode& src, unsigned numNodes) {
    unsigned minHeight = std::numeric_limits<unsigned>::max();
    int minEdge;

    int current = 0;
    for (edge_iterator ii = g.edge_begin(src, Galois::NONE), ei = g.edge_end(src, Galois::NONE); ii != ei; ++ii, ++current) {
      GraphNode dst = g.getEdgeDst(ii);
      int cap = g.getEdgeData(ii).cap;
      if (cap > 0) {
        node_type& dnode = g.getData(dst, Galois::NONE);
        if (dnode.height < minHeight) {
          minHeight = dnode.height;
          minEdge = current;
        }
      }
    }

    assert(minHeight != std::numeric_limits<unsigned>::max());
    ++minHeight;

    node_type& node = g.getData(src, Galois::NONE);
    node.height = minHeight;
    node.current = minEdge;
  }

  struct Process {
    typedef int tt_needs_parallel_break;

    MatchingMF<G,Concurrent>& parent;
    G& g;
    const GraphNode& source;
    const GraphNode& sink;
    unsigned numNodes;
    unsigned globalRelabelInterval;
    bool& shouldGlobalRelabel;
    unsigned counter;

    Process(MatchingMF<G,Concurrent>& p,
        G& _g,
        const GraphNode& _source,
        const GraphNode& _sink,
        unsigned _numNodes,
        unsigned i,
        bool& s):
      parent(p), g(_g), source(_source), sink(_sink), numNodes(_numNodes),
      globalRelabelInterval(i), shouldGlobalRelabel(s), counter(0) { }

    void operator()(const GraphNode& src, Galois::UserContext<GraphNode>& ctx) {
      int increment = 1;
      if (parent.discharge(g, src, ctx, source, sink, numNodes)) {
        increment += BETA;
      }

      counter += increment;
      if (globalRelabelInterval && counter >= globalRelabelInterval) {
        shouldGlobalRelabel = true;
        ctx.breakLoop();
        return;
      }
    }
  };

  template<bool useCAS>
  struct UpdateHeights {
    typedef int tt_does_not_need_stats;
    G& g;

    UpdateHeights(G& _g): g(_g) { }
    //! Do reverse BFS on residual graph.
    void operator()(const GraphNode& src, Galois::UserContext<GraphNode>& ctx) {
      for (edge_iterator
          ii = g.edge_begin(src, useCAS ? Galois::NONE : flag),
          ei = g.edge_end(src, useCAS ? Galois::NONE : flag);
          ii != ei; ++ii) {
        GraphNode dst = g.getEdgeDst(ii);
        if (g.getEdgeData(g.findEdge(dst, src, Galois::NONE)).cap > 0) {
          node_type& node = g.getData(dst, Galois::NONE);
          unsigned newHeight = g.getData(src, Galois::NONE).height + 1;
          if (useCAS) {
            unsigned oldHeight;
            while (newHeight < (oldHeight = node.height)) {
              if (__sync_bool_compare_and_swap(&node.height, oldHeight, newHeight)) {
                ctx.push(dst);
                break;
              }
            }
          } else {
            if (newHeight < node.height) {
              node.height = newHeight;
              ctx.push(dst);
            }
          }
        }
      }
    }
  };

  void globalRelabel(G& g, const GraphNode& source, const GraphNode& sink, unsigned numNodes,
      std::vector<GraphNode>& incoming) {

    for (iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
      GraphNode src = *ii;
      node_type& node = g.getData(src, Galois::NONE);
      node.height = numNodes;
      node.current = 0;
      if (src == sink)
        node.height = 0;
    }

    Galois::StatTimer T("BfsTime");
    T.start();
    Galois::for_each(sink, UpdateHeights<false>(g));
    T.stop();

    for (iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
      GraphNode src = *ii;
      node_type& node = g.getData(src, Galois::NONE);
      if (src == sink || src == source)
        continue;
      if (node.excess > 0) 
        incoming.push_back(src);
    }
  }

  void initializePreflow(G& g, const GraphNode& source, std::vector<GraphNode>& initial) {
    for (edge_iterator ii = g.edge_begin(source), ei = g.edge_end(source); ii != ei; ++ii) {
      GraphNode dst = g.getEdgeDst(ii);
      edge_type& edge = g.getEdgeData(ii);
      int cap = edge.cap;
      if (cap > 0)
        initial.push_back(dst);
      reduceCapacity(edge, g.getEdgeData(g.findEdge(dst, source)), cap);
      g.getData(dst).excess += cap;
    }
  }

  //! Adds reverse edges,
  void initializeGraph(G& g, GraphNode& source, GraphNode& sink, unsigned& numNodes,
      unsigned& interval) {
    size_t numEdges = 0;

    numNodes = g.size();
    source = g.createNode(node_type(numNodes++));
    sink = g.createNode(node_type(numNodes++));
    g.getData(source).height = numNodes;
    g.addNode(source);
    g.addNode(sink);

    // Add reverse edge
    for (typename NodeList::iterator src = g.A.begin(), esrc = g.A.end(); src != esrc; ++src) {
      for (edge_iterator ii = g.edge_begin(*src), ei = g.edge_end(*src);
          ii != ei; ++ii) {
        GraphNode dst = g.getEdgeDst(ii);
        g.getEdgeData(g.addMultiEdge(dst, *src)) = edge_type(0);
        ++numEdges;
      }
    }

    // Add edge from source to each node in A
    for (typename NodeList::iterator src = g.A.begin(), esrc = g.A.end(); src != esrc; ++src) {
      g.getEdgeData(g.addMultiEdge(source, *src)) = edge_type();
      g.getEdgeData(g.addMultiEdge(*src, source)) = edge_type(0);
      ++numEdges;
    }

    // Add edge to sink from each node in B
    for (typename NodeList::iterator src = g.B.begin(), esrc = g.B.end(); src != esrc; ++src) {
      g.getEdgeData(g.addMultiEdge(*src, sink)) = edge_type();
      g.getEdgeData(g.addMultiEdge(sink, *src)) = edge_type(0);
      ++numEdges;
    }

    interval = numNodes * ALPHA + numEdges;
  }

  //! Extract matching from saturated edges
  void extractMatching(G& g) {
    for (typename NodeList::iterator src = g.A.begin(), esrc = g.A.end(); src != esrc; ++src) {
      for (edge_iterator ii = g.edge_begin(*src), ei = g.edge_end(*src); ii != ei; ++ii) {
        GraphNode dst = g.getEdgeDst(ii);
        if (g.getEdgeData(ii).cap == 0) {
          g.getData(*src).free = g.getData(dst).free = false;
        }
      }
    }
  }

  void operator()(G& g) {
    Galois::StatTimer t("serial");

    t.start();
    GraphNode source;
    GraphNode sink;
    unsigned numNodes;
    unsigned interval;
    initializeGraph(g, source, sink, numNodes, interval);

    std::vector<GraphNode> initial;
    initializePreflow(g, source, initial);
    t.stop();

    bool shouldGlobalRelabel = false;
    Galois::setActiveThreads(Concurrent ? numThreads : 1);
    while (!initial.empty()) {
      Galois::for_each(initial.begin(), initial.end(), 
          Process(*this, g, source, sink, numNodes, interval, shouldGlobalRelabel));

      if (!shouldGlobalRelabel)
        break;

      t.start();
      std::cout << "Starting global relabel, current excess at sink " 
        << g.getData(sink).excess << "\n";
      initial.clear();
      globalRelabel(g, source, sink, numNodes, initial);
      shouldGlobalRelabel = false;
      t.stop();
    }

    t.start();
    std::cout << "Final excess at sink " << g.getData(sink).excess << "\n";
    g.removeNode(sink);
    g.removeNode(source);
    extractMatching(g);
    t.stop();
  }
};

template<typename G>
struct Exists<G,MatchingMF> {
  typedef typename G::edge_iterator edge_iterator;

  bool operator()(G& g, const edge_iterator& ii) { 
    //assert(g.getEdgeData(src, dst).cap + g.getEdgeData(dst, src).cap == 1);
    //assert(g.getEdgeData(src, dst).cap != g.getEdgeData(dst, src).cap);
    return g.getEdgeData(ii).cap == 1;
  }
};

// ******************* Verification ***************************

template<typename G>
struct Verifier {
  typedef typename G::GraphNode GraphNode;
  typedef typename G::node_type node_type;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename G::NodeList NodeList;
  typedef typename GraphTypes<G>::Matching Matching;

  bool hasCoveredNeighbors(G& g, const GraphNode& src) {
    for (edge_iterator ii = g.edge_begin(src), ei = g.edge_end(src); ii != ei; ++ii) {
      GraphNode dst = g.getEdgeDst(ii);
      if (!g.getData(dst).covered)
        return false;
    }
    return true;
  }

  void check(G& g, typename NodeList::iterator ii, typename NodeList::iterator ei,
      size_t& count, bool& retval) {
    for (; ii != ei; ++ii) {
      node_type& dii = g.getData(*ii);
      if (dii.degree > 1) {
        std::cerr << "Error: not a matching, node " << dii.id << " incident to " << dii.degree << " edges\n";
        retval = false;
      }

      if (dii.covered) {
        count++;
      }

      if (dii.covered || hasCoveredNeighbors(g, *ii)) {
        // Good
      } else {
        std::cerr << "Error: not a node cover, node " << dii.id 
          << " with degree " << dii.degree << " not covered nor incident to covered node\n";
        retval = false;
      }
    }
  }

  bool operator()(G& g, const Matching& matching) {
    for (typename Matching::const_iterator ii = matching.begin(),
        ei = matching.end(); ii != ei; ++ii) {
      g.getData(ii->first).degree++;
      g.getData(ii->second).degree++;
    }

    bool retval = true;
    size_t count = 0;
    check(g, g.A.begin(), g.A.end(), count, retval);
    check(g, g.B.begin(), g.B.end(), count, retval);

    if (count != matching.size()) {
      std::cerr << "Error: matching is different than node cover " << matching.size() << " vs " << count << "\n";
      retval = false;
    }

    return retval;
  }
};


static double nextRand() {
  return rand() / (double) RAND_MAX;
}

/**
 * Generate a random bipartite graph as used in LEDA evaluation and
 * refererenced in [CGM+97]. Nodes are divided into numGroups groups of size
 * numA/numGroups each. Each node in A has degree d = numEdges/numA and the
 * edges out of a node in group i of A go to random nodes in groups i+1 and
 * i-1  of B. If numGroups == 0, just randomly assign nodes of A to nodes of
 * B.
 */
template<typename G>
void generateInput(int numA, int numB, int numEdges, int numGroups, G* g) {
  typedef typename G::node_type node_type;

  assert(numA > 0 && numB > 0);

  size_t id = 0;

  for (int i = 0; i < numA; ++i)
    g->addNode(g->createNode(node_type(id++)), true);
  for (int i = 0; i < numB; ++i)
    g->addNode(g->createNode(node_type(id++)), false);

  int d = numEdges/numA;
  if (numGroups > numA)
    numGroups = numA;
  if (numGroups > numB)
    numGroups = numB;

  int count = 0;
  if (numGroups > 0) {
    int aSize = numA/numGroups;
    int bSize = numB/numGroups;

    for (typename G::NodeList::iterator ii = g->A.begin(), ei = g->A.end();
        ii != ei; ++ii, ++count) {
      int group = count/aSize;
      if (group == numGroups)
        break;
      int base1 = group == 0 ? (numGroups-1)*bSize : (group-1)*bSize;
      int base2 = group == numGroups-1 ? 0 : (group+1)*bSize;
      for (int i = 0; i < d; ++i) {
        int b = nextRand() < 0.5 ? base1 : base2;
        int off = (int)(nextRand() * (bSize-1));
        g->addEdge(*ii, g->B[b+off]);
      }
    }
  }

  int r = numEdges - count*d;
  while (r--) {
    int ind_a = (int)(nextRand()*(numA-1));
    int ind_b = (int)(nextRand()*(numB-1));
    g->addEdge(g->A[ind_a], g->B[ind_b]);
  }
}


template<template<typename,bool> class Algo, typename G, bool Concurrent>
void start(int N, int numEdges, int numGroups) {
  typedef Algo<G,Concurrent> A;

  G g;
  generateInput(N, N, numEdges, numGroups, &g);

  A algo;
  std::cout << "Starting " << algo.name() << "\n";

  Galois::StatTimer t;
  t.start();
  algo(g);
  t.stop();

  if (!skipVerify) {
    typename GraphTypes<G>::Matching matching;
    PrepareForVerifier<G,Algo>()(g, &matching);
    if (!Verifier<G>()(g, matching)) {
      std::cerr << "Verification failed.\n";
      //assert(0 && "Verification failed");
      //abort();
    } else {
      std::cout << "Verification succeeded.\n";
    }
    std::cout << "Algorithm produced matching of cardinality: " << matching.size() << "\n";
  }
}


template<bool Concurrent>
void start() {
  switch (algo) {
    case pfpAlgo:
      start<MatchingMF, BipartiteGraph<MFNode,MFEdge>, Concurrent>(N, numEdges, numGroups); break;
    case ffAlgo:
      start<MatchingFF, BipartiteGraph<FFNode,void>, Concurrent>(N, numEdges, numGroups); break;
    default:
    case abmpAlgo:
      start<MatchingABMP, BipartiteGraph<ABMPNode,void>, Concurrent>(N, numEdges, numGroups); break;
  }
}

int main(int argc, char** argv) {
  Galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);

  std::cout << "N: " << N 
    << " numEdges: " << numEdges 
    << " numGroups: " << numGroups
    << " seed: " << seed << "\n";

  srand(seed);

  switch (executionType) {
    case serial: start<false>(); break;
    default:
    case parallel: start<true>(); break;
  }

  return 0;
}

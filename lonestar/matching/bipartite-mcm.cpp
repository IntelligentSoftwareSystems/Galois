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
  pfpAlgo,
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
      clEnumVal(pfpAlgo, "Preflow-push"),
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

// TODO(ddn): switch to this graph for FF and ABMP algos when we fix reading
// graphs
template<typename NodeTy,typename EdgeTy>
struct BipartiteGraph: public galois::graphs::LC_Morph_Graph<NodeTy,EdgeTy> {
  typedef galois::graphs::LC_Morph_Graph<NodeTy,EdgeTy> Super;
  typedef std::vector<typename Super::GraphNode> NodeList;

  NodeList A;
  NodeList B;
};

template<typename NodeTy,typename EdgeTy>
struct MFBipartiteGraph: public galois::graphs::FirstGraph<NodeTy,EdgeTy,true> {
  typedef galois::graphs::FirstGraph<NodeTy,EdgeTy,true> Super;
  typedef std::vector<typename Super::GraphNode> NodeList;

  NodeList A;
  NodeList B;
};

//******************************** Common ************************

template<typename G, template<typename,bool> class Algo>
struct Exists {
  bool operator()(G& g, const typename G::edge_iterator&) { return true; }
};

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
      for (auto ii : g.edges(cur)) {
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
  typedef typename G::node_data_type node_data_type;
  typedef typename G::edge_iterator edge_iterator;

  void operator()(G& g, Matching* matching) {
    Exists<G,Algo> exists;

    for (auto src : g.B) {
      for (auto ii : g.edges(src)) {
        GraphNode dst = g.getEdgeDst(ii);
        if (exists(g, ii)) {
          matching->push_back(Edge(src, dst));
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
  typedef typename G::NodeList NodeList;
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

      for (auto ii : g.edges(src, flag)) {
        GraphNode dst = g.getEdgeDst(ii);
        node_data_type& ddst = g.getData(dst, galois::MethodFlag::UNPROTECTED);
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
            GraphNode pred = preds[g.getData(cur, galois::MethodFlag::UNPROTECTED).pred];
            revs.push_back(Edge(pred, cur));
            cur = pred;
          }
          return true;
        } else {
          assert(std::distance(g.edge_begin(dst), g.edge_end(dst)) == 1);
          for (auto jj : g.edges(dst, flag)) {
            GraphNode cur = g.getEdgeDst(jj);

            g.getData(cur, galois::MethodFlag::UNPROTECTED).pred = preds.size();
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
      typename RevsWrapper::Type& revs, typename ReachedWrapper::Type& reached) {

    ReachedCleanup cleanup(g, reached);

    if (findAugmentingPath(g, src, ctx, revs, reached)) {
      g.getData(src, galois::MethodFlag::UNPROTECTED).free = false;

      // Reverse edges in augmenting path
      for (typename RevsWrapper::Type::iterator jj = revs.begin(), ej = revs.end(); jj != ej; ++jj) {
        auto edge = g.findEdge(jj->first, jj->second, galois::MethodFlag::UNPROTECTED);
        assert(edge != g.edge_end(jj->first));
        g.removeEdge(jj->first, edge, galois::MethodFlag::UNPROTECTED);
        g.addEdge(jj->second, jj->first, galois::MethodFlag::UNPROTECTED);
      }
      revs.clear();

      cleanup.clear();
    }
  }


  void operator()(G& g) {
    SerialRevs revs;
    SerialReached reached;

    galois::setActiveThreads(Concurrent ? numThreads : 1);
    galois::for_each(galois::iterate(g.A),
        [&, outer=this] (const GraphNode& node, auto& ctx) {
          if (!g.getData(node, flag).free)
            return;

          ParallelRevs parallelRevs(ctx.getPerIterAlloc());
          ParallelReached parallelReached(ctx.getPerIterAlloc());

          outer->propagate(g, node, ctx,
              RevsWrapper(revs, parallelRevs).get(),
              ReachedWrapper(reached, parallelReached).get());
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
  typedef typename G::NodeList NodeList;
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

  bool nextEdge(G& g, const GraphNode& src, GraphNode& next) {
    node_data_type& dsrc = g.getData(src, galois::MethodFlag::UNPROTECTED);
    unsigned l = dsrc.layer - 1;

    // Start search where we last left off
    edge_iterator ii = g.edge_begin(src, flag);
    edge_iterator ei = g.edge_end(src, flag);
    assert(dsrc.next <= std::distance(ii, ei));
    std::advance(ii, dsrc.next);
    for (; ii != ei && g.getData(g.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED).layer != l;
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
  //TODO: better name here
  template <typename C>
  bool propagate(G& g, const GraphNode& root, C& ctx) {
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
        for (typename Revs::iterator ii = revs.begin(), ei = revs.end(); ii != ei; ++ii) {
          auto edge = g.findEdge(ii->first, ii->second, galois::MethodFlag::UNPROTECTED);
          assert(edge != g.edge_end(ii->first));
          g.removeEdge(ii->first, edge, galois::MethodFlag::UNPROTECTED);
          g.addEdge(ii->second, ii->first, galois::MethodFlag::UNPROTECTED);
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

  void operator()(G& g) {
    galois::StatTimer t("serial");
    t.start();
    std::vector<WorkItem> initial;
    for (typename NodeList::iterator ii = g.A.begin(), ei = g.A.end(); ii != ei; ++ii) {
      g.getData(*ii).layer = 1;
      if (g.getData(*ii).free)
        initial.push_back(std::make_pair(*ii, 1));
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
          if (!outer->propagate(g, item.first, ctx)) {
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
    algo(g);
    t.stop();
  }
};

// *************************** MaxFlow Algorithm *******************************
struct MFNode: public BaseNode {
  size_t excess;
  unsigned height;
  int current;
  MFNode(size_t i = -1): BaseNode(i), excess(0), height(1), current(0) { }
  void reset() {
    BaseNode::reset();
    excess = 0;
    height = 1;
    current = 0;
  }
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
  typedef typename G::node_data_type node_data_type;
  typedef typename G::edge_data_type edge_data_type;
  static const galois::MethodFlag flag = Concurrent ? galois::MethodFlag::WRITE : galois::MethodFlag::UNPROTECTED;
  static const bool canRunIteratively = false;

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

  void reduceCapacity(edge_data_type& edge1, edge_data_type& edge2, int amount) {
    edge1.cap -= amount;
    edge2.cap += amount;
  }

  template <typename C>
  bool discharge(G& g, const GraphNode& src, C& ctx,
      const GraphNode& source, const GraphNode& sink, unsigned numNodes) {
    node_data_type& node = g.getData(src, flag);
    //unsigned prevHeight = node.height;
    bool relabeled = false;

    if (node.excess == 0) {
      return false;
    }

    while (true) {
      galois::MethodFlag f = relabeled ? galois::MethodFlag::UNPROTECTED : flag;
      bool finished = false;
      int current = -1;

      for (auto ii : g.edges(src, f)) {
        ++current;
        GraphNode dst = g.getEdgeDst(ii);
        edge_data_type& edge = g.getEdgeData(ii);
        if (edge.cap == 0 || current < node.current) 
          continue;

        node_data_type& dnode = g.getData(dst, galois::MethodFlag::UNPROTECTED);
        if (node.height - 1 != dnode.height) 
          continue;

        // Push flow
        int amount = std::min(static_cast<int>(node.excess), edge.cap);
        reduceCapacity(edge, g.getEdgeData(g.findEdge(dst, src, galois::MethodFlag::UNPROTECTED)), amount);

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
    int minEdge = 0; // TODO: not sure of initial value

    int current = -1;
    for (auto ii : g.edges(src, galois::MethodFlag::UNPROTECTED)) {
      ++current;
      GraphNode dst = g.getEdgeDst(ii);
      int cap = g.getEdgeData(ii).cap;
      if (cap > 0) {
        node_data_type& dnode = g.getData(dst, galois::MethodFlag::UNPROTECTED);
        if (dnode.height < minHeight) {
          minHeight = dnode.height;
          minEdge = current;
        }
      }
    }

    assert(minHeight != std::numeric_limits<unsigned>::max());
    ++minHeight;

    node_data_type& node = g.getData(src, galois::MethodFlag::UNPROTECTED);
    node.height = minHeight;
    node.current = minEdge;
  }

  void globalRelabel(G& g, const GraphNode& source, const GraphNode& sink, unsigned numNodes,
      std::vector<GraphNode>& incoming) {

    for (iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
      GraphNode src = *ii;
      node_data_type& node = g.getData(src, galois::MethodFlag::UNPROTECTED);
      node.height = numNodes;
      node.current = 0;
      if (src == sink)
        node.height = 0;
    }

    constexpr static const bool useCAS = false;

    galois::StatTimer T("BfsTime");
    T.start();
    galois::for_each(galois::iterate({ sink }), 
        [&] (const GraphNode& src, auto& ctx) {
          for (auto ii : g.edges(src, useCAS ? galois::MethodFlag::UNPROTECTED : flag)) {
            GraphNode dst = g.getEdgeDst(ii);
            if (g.getEdgeData(g.findEdge(dst, src, galois::MethodFlag::UNPROTECTED)).cap > 0) {
              node_data_type& node = g.getData(dst, galois::MethodFlag::UNPROTECTED);
              unsigned newHeight = g.getData(src, galois::MethodFlag::UNPROTECTED).height + 1;
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
        }, 
        galois::wl<galois::worklists::dChunkedFIFO<32> >(),
        galois::no_stats());
    T.stop();

    for (iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
      GraphNode src = *ii;
      node_data_type& node = g.getData(src, galois::MethodFlag::UNPROTECTED);
      if (src == sink || src == source)
        continue;
      if (node.excess > 0) 
        incoming.push_back(src);
    }
  }

  void initializePreflow(G& g, const GraphNode& source, std::vector<GraphNode>& initial) {
    for (auto ii : g.edges(source)) {
      GraphNode dst = g.getEdgeDst(ii);
      edge_data_type& edge = g.getEdgeData(ii);
      int cap = edge.cap;
      if (cap > 0)
        initial.push_back(dst);
      reduceCapacity(edge, g.getEdgeData(g.findEdge(dst, source)), cap);
      g.getData(dst).excess += cap;
    }
  }

  //! Adds reverse edges
  void initializeGraph(G& g, GraphNode& source, GraphNode& sink, unsigned& numNodes,
      unsigned& globalRelabelInterval) {
    size_t numEdges = 0;

    numNodes = std::distance(g.begin(), g.end());
    source = g.createNode(node_data_type(numNodes++));
    sink = g.createNode(node_data_type(numNodes++));
    g.getData(source).height = numNodes;
    g.addNode(source);
    g.addNode(sink);

    // Add reverse edge
    for (auto src : g.A) {
      for (auto ii : g.edges(src)) {
        GraphNode dst = g.getEdgeDst(ii);
        g.getEdgeData(g.addMultiEdge(dst, src, galois::MethodFlag::WRITE)) = edge_data_type(0);
        ++numEdges;
      }
    }

    // Add edge from source to each node in A
    for (typename NodeList::iterator src = g.A.begin(), esrc = g.A.end(); src != esrc; ++src) {
      g.getEdgeData(g.addMultiEdge(source, *src, galois::MethodFlag::WRITE)) = edge_data_type();
      g.getEdgeData(g.addMultiEdge(*src, source, galois::MethodFlag::WRITE)) = edge_data_type(0);
      ++numEdges;
    }

    // Add edge to sink from each node in B
    for (typename NodeList::iterator src = g.B.begin(), esrc = g.B.end(); src != esrc; ++src) {
      g.getEdgeData(g.addMultiEdge(*src, sink, galois::MethodFlag::WRITE)) = edge_data_type();
      g.getEdgeData(g.addMultiEdge(sink, *src, galois::MethodFlag::WRITE)) = edge_data_type(0);
      ++numEdges;
    }

    globalRelabelInterval = numNodes * ALPHA + numEdges;
  }

  //! Extract matching from saturated edges
  void extractMatching(G& g) {
    for (auto src : g.A) {
      for (auto ii : g.edges(src)) {
        GraphNode dst = g.getEdgeDst(ii);
        if (g.getEdgeData(ii).cap == 0) {
          g.getData(src).free = g.getData(dst).free = false;
        }
      }
    }
  }

  void operator()(G& g) {
    galois::StatTimer t("serial");

    t.start();
    GraphNode source;
    GraphNode sink;
    unsigned numNodes;
    unsigned globalRelabelInterval;
    initializeGraph(g, source, sink, numNodes, globalRelabelInterval);

    std::vector<GraphNode> initial;
    initializePreflow(g, source, initial);
    t.stop();

    bool shouldGlobalRelabel = false;
    unsigned counter = 0;
    galois::setActiveThreads(Concurrent ? numThreads : 1);

    while (!initial.empty()) {
      galois::for_each(galois::iterate(initial),
          [&, outer=this] (const GraphNode& src, auto& ctx) {
            int increment = 1;
            if (outer->discharge(g, src, ctx, source, sink, numNodes)) {
              increment += BETA;
            }

            counter += increment;
            if (globalRelabelInterval && counter >= globalRelabelInterval) {
              shouldGlobalRelabel = true;
              ctx.breakLoop();
              return;
            }
          }, 
          galois::loopname("MatchingMF"),
          galois::parallel_break(),
          galois::wl<galois::worklists::dChunkedFIFO<32> >());

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
  typedef typename G::node_data_type node_data_type;
  typedef typename G::edge_iterator edge_iterator;
  typedef typename G::NodeList NodeList;
  typedef typename GraphTypes<G>::Matching Matching;

  bool hasCoveredNeighbors(G& g, const GraphNode& src) {
    for (auto ii : g.edges(src)) {
      GraphNode dst = g.getEdgeDst(ii);
      if (!g.getData(dst).covered)
        return false;
    }
    return true;
  }

  void check(G& g, typename NodeList::iterator ii, typename NodeList::iterator ei,
      size_t& count, bool& retval) {
    for (; ii != ei; ++ii) {
      node_data_type& dii = g.getData(*ii);
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


/**
 * Generate a random bipartite graph as used in LEDA evaluation and
 * refererenced in [CGM+97]. Nodes are divided into numGroups groups of size
 * numA/numGroups each. Each node in A has degree d = numEdges/numA and the
 * edges out of a node in group i of A go to random nodes in groups i+1 and
 * i-1  of B. If numGroups == 0, just randomly assign nodes of A to nodes of
 * B.
 */
template<typename G>
void generateRandomInput(int numA, int numB, int numEdges, int numGroups, int seed, G& g) {
  typedef typename G::edge_data_type edge_data_type;
  typedef typename G::GraphNode GNode;
  
  std::cout 
    << "numGroups: " << numGroups
    << " seed: " << seed
    << "\n";

  galois::graphs::FileGraphWriter p;
  p.setNumNodes(numA + numB);
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(galois::LargeArray<edge_data_type>::size_of::value);

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
          if (phase == 0)
            p.incrementDegree(ii);
          else
            p.addNeighbor(ii, b+off+numA);
        }
      }
    }

    int r = numEdges - count*d;
    while (r--) {
      int ind_a = (int)(dist(gen)*(numA-1));
      int ind_b = (int)(dist(gen)*(numB-1));
      if (phase == 0)
        p.incrementDegree(ind_a);
      else
        p.addNeighbor(ind_a, ind_b + numA);
    }
  }

  // Leave edge data uninitialized
  p.finish<edge_data_type>();
  galois::graphs::readGraph(g, p);
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
size_t countMatching(G& g) {
  Exists<G,Algo> exists;
  size_t count = 0;
  for (auto n : g.B) {
    for (auto edge : g.out_edges(n)) {
      if (exists(g, edge)) {
        count += 1;
      }
    }
  }
  return count;
}

template<template<typename,bool> class Algo, typename G>
void removeMatchedEdges(G& g) {
  Exists<G,Algo> exists;
  for (auto n : g.B) {
    assert(std::distance(g.edge_begin(n), g.edge_end(n)) <= 1);
    for (auto edge : g.out_edges(n)) {
      if (exists(g, edge)) {
        g.removeEdge(n, edge);
        break;
      }
    }
  }
}

template<template<typename,bool> class Algo, typename G, bool Concurrent>
void start(int N, int numEdges, int numGroups) {
  typedef Algo<G,Concurrent> A;

  A algo;
  G g;

  if (runIteratively && !algo.canRunIteratively)
    GALOIS_DIE("algo does not support iterative execution");

  switch (inputType) {
    case generated: generateRandomInput(N, N, numEdges, numGroups, seed, g); break;
    case fromFile: readInput(inputFilename, g); break;
    default: GALOIS_DIE("unknown input type");
  }

  size_t id = 0;
  for (auto n : g) {
    g.getData(n).id = id++;
    if (g.edge_begin(n) != g.edge_end(n))
      g.A.push_back(n);
    else
      g.B.push_back(n);
  }

  std::cout 
    << "numA: " << g.A.size()
    << " numB: " << g.B.size()
    << "\n";

  std::cout << "Starting " << algo.name() << "\n";

  galois::StatTimer t;

  while (true) {
    t.start();
    algo(g);
    t.stop();

    if (!skipVerify) {
      typename GraphTypes<G>::Matching matching;
      PrepareForVerifier<G,Algo>()(g, &matching);
      if (!Verifier<G>()(g, matching)) {
        GALOIS_DIE("Verification failed");
      } else {
        std::cout << "Verification successful.\n";
      }
    }

    size_t matchingSize = countMatching<Algo>(g);
    std::cout << "Matching of cardinality: " << matchingSize << "\n";

    if (!runIteratively || matchingSize == 0)
      break;

    removeMatchedEdges<Algo>(g);
    for (auto n : g)
      g.getData(n).reset();
  }
}


template<bool Concurrent>
void start() {
  switch (algo) {
    case abmpAlgo:
      start<MatchingABMP, MFBipartiteGraph<ABMPNode,void>, Concurrent>(N, numEdges, numGroups); break;
    case pfpAlgo:
      start<MatchingMF, MFBipartiteGraph<MFNode,MFEdge>, Concurrent>(N, numEdges, numGroups); break;
    case ffAlgo:
      start<MatchingFF, MFBipartiteGraph<FFNode,void>, Concurrent>(N, numEdges, numGroups); break;
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

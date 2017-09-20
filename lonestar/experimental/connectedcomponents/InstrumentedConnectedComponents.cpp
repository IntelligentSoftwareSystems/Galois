/** Connected components -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Compute the connect components of a graph and optionally write out the largest
 * component to file.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/DomainSpecificExecutors.h"
#include "Galois/Timer.h"
#include "Galois/UnionFind.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/OCGraph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "Galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

#ifdef GALOIS_USE_EXP
#include "LigraAlgo.h"
#include "GraphLabAlgo.h"
#include "GraphChiAlgo.h"
#include "Galois/WorkList/WorkSet.h"
#include "Galois/WorkList/MarkingSet.h"
#endif

#include <ostream>
#include <fstream>
#include <cmath>

const char* name = "Connected Components";
const char* desc = "Computes the connected components of a graph";
const char* url = 0;

enum Algo {
  async,
  asyncOc,
  asyncOSet,
  asyncHSet,
  blockedasync,
  graphchi,
  graphlab,
  labelProp,
  labelPropDivObim,
  labelPropDivObimMSet,
  labelPropDivObimOSet,
  labelPropDivObimHSet,
  labelPropLogObim,
  labelPropLogObimMSet,
  labelPropLogObimOSet,
  labelPropLogObimHSet,
  labelPropMSet,
  labelPropOSet,
  labelPropHSet,
  labelPropShrObim,
  labelPropShrObimMSet,
  labelPropShrObimOSet,
  labelPropShrObimHSet,
  gLp,
  gLpDivObim,
  gLpDivObimMSet,
  gLpDivObimOSet,
  gLpDivObimHSet,
  gLpLogObim,
  gLpLogObimMSet,
  gLpLogObimOSet,
  gLpLogObimHSet,
  gLpMSet,
  gLpOSet,
  gLpHSet,
  gLpShrObim,
  gLpShrObimMSet,
  gLpShrObimOSet,
  gLpShrObimHSet,
  gPullLp,
  ligra,
  ligraChi,
  pullLp,
  pullLpDivObim,
  pullLpDivObimMSet,
  pullLpDivObimOSet,
  pullLpDivObimHSet,
  pullLpLogObim,
  pullLpLogObimMSet,
  pullLpLogObimOSet,
  pullLpLogObimHSet,
  pullLpMSet,
  pullLpOSet,
  pullLpHSet,
  pullLpShrObim,
  pullLpShrObimMSet,
  pullLpShrObimOSet,
  pullLpShrObimHSet,
  serial,
  staleLp,
  staleLpDivObim,
  staleLpDivObimMSet,
  staleLpDivObimOSet,
  staleLpDivObimHSet,
  staleLpLogObim,
  staleLpLogObimMSet,
  staleLpLogObimOSet,
  staleLpLogObimHSet,
  staleLpMSet,
  staleLpOSet,
  staleLpHSet,
  staleLpShrObim,
  staleLpShrObimMSet,
  staleLpShrObimOSet,
  staleLpShrObimHSet,
  synchronous
};

enum OutputEdgeType {
  void_,
  int32_,
  int64_
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> largestComponentFilename("outputLargestComponent", cll::desc("[output graph file]"), cll::init(""));
static cll::opt<std::string> permutationFilename("outputNodePermutation", cll::desc("[output node permutation file]"), cll::init(""));
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"), cll::init(false));
cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<unsigned int> delta("delta", cll::desc("delta value for <algo>ShrObim"), cll::init(10U));
static cll::opt<OutputEdgeType> writeEdgeType("edgeType", cll::desc("Input/Output edge type:"),
    cll::values(
      clEnumValN(OutputEdgeType::void_, "void", "no edge values"),
      clEnumValN(OutputEdgeType::int32_, "int32", "32 bit edge values"),
      clEnumValN(OutputEdgeType::int64_, "int64", "64 bit edge values"),
      clEnumValEnd), cll::init(OutputEdgeType::void_));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::async, "async", "Asynchronous (default)"),
      clEnumValN(Algo::blockedasync, "blockedasync", "Blocked asynchronous"),
      clEnumValN(Algo::asyncOc, "asyncOc", "Asynchronous out-of-core memory"),
      clEnumValN(Algo::labelProp, "labelProp", "Using label propagation algorithm"),
      clEnumValN(Algo::staleLp, "staleLp", "Using label propagation algorithm w/ possibly stale lables"),
      clEnumValN(Algo::serial, "serial", "Serial"),
      clEnumValN(Algo::synchronous, "sync", "Synchronous"),
#ifdef GALOIS_USE_EXP
      clEnumValN(Algo::asyncOSet, "asyncOSet", "async with a two-level set uni-set scheduler"),
      clEnumValN(Algo::asyncHSet, "asyncHSet", "async with a two-level hash uni-set scheduler"),   
      clEnumValN(Algo::labelPropDivObim, "labelPropDivObim", "labelProp with DivObim scheduling"),
      clEnumValN(Algo::labelPropDivObimMSet, "labelPropDivObimMSet", "labelPropDivObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::labelPropDivObimOSet, "labelPropDivObimOSet", "labelPropDivObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::labelPropDivObimHSet, "labelPropDivObimHSet", "labelPropDivObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::labelPropLogObim, "labelPropLogObim", "labelProp with LogObim scheduling"),
      clEnumValN(Algo::labelPropLogObimMSet, "labelPropLogObimMSet", "labelPropLogObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::labelPropLogObimOSet, "labelPropLogObimOSet", "labelPropLogObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::labelPropLogObimHSet, "labelPropLogObimHSet", "labelPropLogObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::labelPropMSet, "labelPropMSet", "labelProp with an item-marking uni-set scheduler"), 
      clEnumValN(Algo::labelPropOSet, "labelPropOSet", "labelProp with a two-level set uni-set scheduler"), 
      clEnumValN(Algo::labelPropHSet, "labelPropHSet", "labelProp with a two-level hash uni-set scheduler"), 
      clEnumValN(Algo::labelPropShrObim, "labelPropShrObim", "labelProp with ShrObim scheduling"),
      clEnumValN(Algo::labelPropShrObimMSet, "labelPropShrObimMSet", "labelPropShrObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::labelPropShrObimOSet, "labelPropShrObimOSet", "labelPropShrObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::labelPropShrObimHSet, "labelPropShrObimHSet", "labelPropShrObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::gLp, "gLp", "labelProp using Galois sync mechanism"),
      clEnumValN(Algo::gLpDivObim, "gLpDivObim", "gLp with DivObim scheduling"),
      clEnumValN(Algo::gLpDivObimMSet, "gLpDivObimMSet", "gLpDivObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::gLpDivObimOSet, "gLpDivObimOSet", "gLpDivObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::gLpDivObimHSet, "gLpDivObimHSet", "gLpDivObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::gLpLogObim, "gLpLogObim", "gLp with LogObim scheduling"),
      clEnumValN(Algo::gLpLogObimMSet, "gLpLogObimMSet", "gLpLogObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::gLpLogObimOSet, "gLpLogObimOSet", "gLpLogObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::gLpLogObimHSet, "gLpLogObimHSet", "gLpLogObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::gLpMSet, "gLpMSet", "gLp with an item-marking uni-set scheduler"), 
      clEnumValN(Algo::gLpOSet, "gLpOSet", "gLp with a two-level set uni-set scheduler"), 
      clEnumValN(Algo::gLpHSet, "gLpHSet", "gLp with a two-level hash uni-set scheduler"), 
      clEnumValN(Algo::gLpShrObim, "gLpShrObim", "gLp with ShrObim scheduling"),
      clEnumValN(Algo::gLpShrObimMSet, "gLpShrObimMSet", "gLpShrObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::gLpShrObimOSet, "gLpShrObimOSet", "gLpShrObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::gLpShrObimHSet, "gLpShrObimHSet", "gLpShrObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::staleLpDivObim, "staleLpDivObim", "staleLp with DivObim scheduling"),
      clEnumValN(Algo::staleLpDivObimMSet, "staleLpDivObimMSet", "staleLpDivObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::staleLpDivObimOSet, "staleLpDivObimOSet", "staleLpDivObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::staleLpDivObimHSet, "staleLpDivObimHSet", "staleLpDivObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::staleLpLogObim, "staleLpLogObim", "staleLp with LogObim scheduling"),
      clEnumValN(Algo::staleLpLogObimMSet, "staleLpLogObimMSet", "staleLpLogObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::staleLpLogObimOSet, "staleLpLogObimOSet", "staleLpLogObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::staleLpLogObimHSet, "staleLpLogObimHSet", "staleLpLogObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::staleLpMSet, "staleLpMSet", "staleLp with an item-marking uni-set scheduler"), 
      clEnumValN(Algo::staleLpOSet, "staleLpOSet", "staleLp with a two-level set uni-set scheduler"), 
      clEnumValN(Algo::staleLpHSet, "staleLpHSet", "staleLp with a two-level hash uni-set scheduler"), 
      clEnumValN(Algo::staleLpShrObim, "staleLpShrObim", "staleLp with ShrObim scheduling"),
      clEnumValN(Algo::staleLpShrObimMSet, "staleLpShrObimMSet", "staleLpShrObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::staleLpShrObimOSet, "staleLpShrObimOSet", "staleLpShrObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::staleLpShrObimHSet, "staleLpShrObimHSet", "staleLpShrObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::gPullLp, "gPullLp", "Using pull-based label propagation algorithm"),
      clEnumValN(Algo::pullLp, "pullLp", "Using pull-based label propagation algorithm w/ CAS"),
      clEnumValN(Algo::pullLpDivObim, "pullLpDivObim", "pullLp with DivObim scheduling"),
      clEnumValN(Algo::pullLpDivObimMSet, "pullLpDivObimMSet", "pullLpDivObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::pullLpDivObimOSet, "pullLpDivObimOSet", "pullLpDivObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::pullLpDivObimHSet, "pullLpDivObimHSet", "pullLpDivObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::pullLpLogObim, "pullLpLogObim", "pullLp with LogObim scheduling"),
      clEnumValN(Algo::pullLpLogObimMSet, "pullLpLogObimMSet", "pullLpLogObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::pullLpLogObimOSet, "pullLpLogObimOSet", "pullLpLogObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::pullLpLogObimHSet, "pullLpLogObimHSet", "pullLpLogObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::pullLpMSet, "pullLpMSet", "pullLp with an item-marking uni-set scheduler"), 
      clEnumValN(Algo::pullLpOSet, "pullLpOSet", "pullLp with a two-level set uni-set scheduler"), 
      clEnumValN(Algo::pullLpHSet, "pullLpHSet", "pullLp with a two-level hash uni-set scheduler"), 
      clEnumValN(Algo::pullLpShrObim, "pullLpShrObim", "pullLp with ShrObim scheduling"),
      clEnumValN(Algo::pullLpShrObimMSet, "pullLpShrObimMSet", "pullLpShrObim with an item-marking uni-set scheduler"),
      clEnumValN(Algo::pullLpShrObimOSet, "pullLpShrObimOSet", "pullLpShrObim with a two-level set uni-set scheduler"),
      clEnumValN(Algo::pullLpShrObimHSet, "pullLpShrObimHSet", "pullLpShrObim with a two-level hash uni-set scheduler"),
      clEnumValN(Algo::graphchi, "graphchi", "Using GraphChi programming model"),
      clEnumValN(Algo::graphlab, "graphlab", "Using GraphLab programming model"),
      clEnumValN(Algo::ligraChi, "ligraChi", "Using Ligra and GraphChi programming model"),
      clEnumValN(Algo::ligra, "ligra", "Using Ligra programming model"),
#endif
      clEnumValEnd), cll::init(Algo::async));

static const bool traceWork = true;
static galois::Statistic* GoodWork;
static galois::Statistic* EmptyWork;

struct Node: public galois::UnionFindNode<Node> {
  typedef Node* component_type;
  unsigned int id;

  Node(): galois::UnionFindNode<Node>(const_cast<Node*>(this)) { }
  Node(const Node& o): galois::UnionFindNode<Node>(o.m_component), id(o.id) { }

  Node& operator=(const Node& o) {
    Node c(o);
    std::swap(c, *this);
    return *this;
  }

  component_type component() { return this->findAndCompress(); }
};

template<typename Graph>
void readInOutGraph(Graph& graph) {
  using namespace galois::Graph;
  if (symmetricGraph) {
    galois::Graph::readGraph(graph, inputFilename);
  } else if (transposeGraphName.size()) {
    galois::Graph::readGraph(graph, inputFilename, transposeGraphName);
  } else {
    GALOIS_DIE("Graph type not supported");
  }
}

/** 
 * Serial connected components algorithm. Just use union-find.
 */
struct SerialAlgo {
  typedef galois::Graph::LC_CSR_Graph<Node,void>
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  template<typename G>
  void readGraph(G& graph) { galois::Graph::readGraph(graph, inputFilename); }

  struct Merge {
    Graph& graph;
    Merge(Graph& g): graph(g) { }

    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      
      for (Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
          ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sdata.merge(&ddata);
      }
    }
  };

  void operator()(Graph& graph) {
    std::for_each(graph.begin(), graph.end(), Merge(graph));
  }
};

/**
 * Synchronous connected components algorithm.  Initially all nodes are in
 * their own component. Then, we merge endpoints of edges to form the spanning
 * tree. Merging is done in two phases to simplify concurrent updates: (1)
 * find components and (2) union components.  Since the merge phase does not
 * do any finds, we only process a fraction of edges at a time; otherwise,
 * the union phase may unnecessarily merge two endpoints in the same
 * component.
 */
struct SynchronousAlgo {
  typedef galois::Graph::LC_CSR_Graph<Node,void>
    ::with_no_lockable<true>::type
    ::with_numa_alloc<true>::type Graph;
  typedef Graph::GraphNode GNode;

  template<typename G>
  void readGraph(G& graph) { galois::Graph::readGraph(graph, inputFilename); }

  struct Edge {
    GNode src;
    Node* ddata;
    int count;
    Edge(GNode src, Node* ddata, int count): src(src), ddata(ddata), count(count) { }
  };

  galois::InsertBag<Edge> wls[2];
  galois::InsertBag<Edge>* next;
  galois::InsertBag<Edge>* cur;

  struct Initialize {
    Graph& graph;
    galois::InsertBag<Edge>& next;
    Initialize(Graph& g, galois::InsertBag<Edge>& next): graph(g), next(next) { }

    //! Add the first edge between components to the worklist
    void operator()(const GNode& src) const {
      for (Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
          ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        if (symmetricGraph && src >= dst)
          continue;
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        next.push(Edge(src, &ddata, 0));
        break;
      }
    }
  };

  struct Merge {
    Graph& graph;
    galois::Statistic& emptyMerges;
    Merge(Graph& g, galois::Statistic& e): graph(g), emptyMerges(e) { }

    void operator()(const Edge& edge) const {
      Node& sdata = graph.getData(edge.src, galois::MethodFlag::UNPROTECTED);
      if (!sdata.merge(edge.ddata))
        emptyMerges += 1;
    }
  };

  struct Find {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;
    typedef int tt_does_not_need_stats;

    Graph& graph;
    galois::InsertBag<Edge>& next;
    Find(Graph& g, galois::InsertBag<Edge>& next): graph(g), next(next) { }

    //! Add the next edge between components to the worklist
    void operator()(const Edge& edge, galois::UserContext<Edge>&) const {
      (*this)(edge);
    }

    void operator()(const Edge& edge) const {
      GNode src = edge.src;
      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      Node* scomponent = sdata.findAndCompress();
      Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
      Graph::edge_iterator ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
      int count = edge.count + 1;
      std::advance(ii, count);
      for (; ii != ei; ++ii, ++count) {
        GNode dst = graph.getEdgeDst(ii);
        if (symmetricGraph && src >= dst)
          continue;
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        Node* dcomponent = ddata.findAndCompress();
        if (scomponent != dcomponent) {
          next.push(Edge(src, dcomponent, count));
          break;
        }
      }
    }
  };

  void operator()(Graph& graph) {
    galois::Statistic rounds("Rounds");
    galois::Statistic emptyMerges("EmptyMerges");

    cur = &wls[0];
    next = &wls[1];
    galois::do_all_local(graph, Initialize(graph, *cur));

    while (!cur->empty()) {
      galois::do_all_local(*cur, Merge(graph, emptyMerges));
      galois::for_each_local(*cur, Find(graph, *next));
      cur->clear();
      std::swap(cur, next);
      rounds += 1;
    }
  }
};

#ifdef GALOIS_USE_EXP
template<typename Graph, typename GNode=typename Graph::GraphNode>
struct LabelDivIndexer: public std::unary_function<GNode, unsigned int> {
  Graph& graph;
  LabelDivIndexer(Graph& g): graph(g) {}

  unsigned int operator() (const GNode n) const {
    return graph.getData(n, galois::MethodFlag::UNPROTECTED).comp / 1000;
  }
};

template<typename Graph, typename GNode=typename Graph::GraphNode>
struct LabelShrIndexer: public std::unary_function<GNode, unsigned int> {
  Graph& graph;
  LabelShrIndexer(Graph& g): graph(g) {}

  unsigned int operator() (const GNode n) const {
    return graph.getData(n, galois::MethodFlag::UNPROTECTED).comp >> delta;
  }
};

template<typename Graph, typename GNode=typename Graph::GraphNode>
struct LabelLogIndexer: public std::unary_function<GNode, unsigned int> {
  Graph& graph;
  LabelLogIndexer(Graph& g): graph(g) {}

  unsigned int operator() (const GNode n) const {
    unsigned int value = graph.getData(n, galois::MethodFlag::UNPROTECTED).comp;
    return (unsigned int)floor(log((double)(value+1)));
  }
};

// a set marker returns the bool* to the set marking field, i.e. inSet, for a GNode
template<typename Graph, typename GNode=typename Graph::GraphNode>
struct LabelSetMarker: public std::unary_function<GNode, bool*> {
  Graph& graph;
  LabelSetMarker(Graph& g): graph(g) {}

  bool* operator()(const GNode n) const {
    return &(graph.getData(n, galois::MethodFlag::UNPROTECTED).inSet);
  }
};

struct LabelPropNoCasAlgo {
  struct LNode {
    typedef unsigned int component_type;
    unsigned int id;
    unsigned int comp;
    bool inSet;
    
    component_type component() { return comp; }
    bool isRep() { return id == comp; }
  };

  typedef typename galois::Graph::LC_CSR_Graph<LNode,void>
    ::template with_numa_alloc<true>::type InnerGraph;
  typedef galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;
  typedef typename LNode::component_type component_type;

  template<typename G>
  void readGraph(G& graph) {
    readInOutGraph(graph);
  }

  struct Initialize {
    Graph& graph;

    Initialize(Graph& g): graph(g) { }
    void operator()(GNode n) const {
      LNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.comp = data.id;
      data.inSet = false;
    }
  };

  template<bool Forward,bool Backward>
  struct Process {
    Graph& graph;
    Process(Graph& g): graph(g) { }

    template<typename Iterator,typename GetNeighbor>
    void update(LNode& sdata, Iterator ii, Iterator ei, GetNeighbor get, galois::UserContext<GNode>& ctx) {
      component_type newV = sdata.comp;
    
      for(; ii != ei; ++ii) {
        GNode dst = get(ii);
        LNode& ddata = graph.getData(dst);
        component_type old = ddata.comp;

        if (old <= newV) {
          if(traceWork) {
            *EmptyWork += 1;
          }
          continue;
        }

        ddata.comp = newV;
        if(traceWork) {
          *GoodWork += 1;
        }
        ctx.push(dst);
      }
    }

    struct BackwardUpdate {
      Graph& graph;
      BackwardUpdate(Graph& g): graph(g) { }
      GNode operator()(typename Graph::in_edge_iterator ii) { return graph.getInEdgeDst(ii); }
    };

    struct ForwardUpdate {
      Graph& graph;
      ForwardUpdate(Graph& g): graph(g) { }
      GNode operator()(typename Graph::edge_iterator ii) { return graph.getEdgeDst(ii); }
    };

    //! Add the next edge between components to the worklist
    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      LNode& sdata = graph.getData(src);

      if (Backward) {
        update(sdata, graph.in_edge_begin(src), graph.in_edge_end(src),
            BackwardUpdate(graph), ctx);
      } 
      if (Forward) {
        update(sdata, graph.edge_begin(src), graph.edge_end(src),
            ForwardUpdate(graph), ctx);
      }
    }
  };

  void operator()(Graph& graph) {
    using namespace galois::WorkList;
    typedef dChunkedFIFO<256> WL;
    typedef dChunkedMarkingSetFIFO<LabelSetMarker<Graph>,256> MSet;
    typedef dChunkedTwoLevelSetFIFO<256> OSet;
    typedef dChunkedTwoLevelHashFIFO<256> HSet;
    typedef OrderedByIntegerMetric<LabelDivIndexer<Graph> > DivObim;
    typedef OrderedByIntegerMetric<LabelShrIndexer<Graph> > ShrObim;
    typedef OrderedByIntegerMetric<LabelLogIndexer<Graph> > LogObim;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,DivObim> DivObimMSet;
    typedef detail::WorkSetMaster<GNode,DivObim,galois::ThreadSafeTwoLevelSet<GNode> > DivObimOSet;
    typedef detail::WorkSetMaster<GNode,DivObim,galois::ThreadSafeTwoLevelHash<GNode> > DivObimHSet;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,ShrObim> ShrObimMSet;
    typedef detail::WorkSetMaster<GNode,ShrObim,galois::ThreadSafeTwoLevelSet<GNode> > ShrObimOSet;
    typedef detail::WorkSetMaster<GNode,ShrObim,galois::ThreadSafeTwoLevelHash<GNode> > ShrObimHSet;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,LogObim> LogObimMSet;
    typedef detail::WorkSetMaster<GNode,LogObim,galois::ThreadSafeTwoLevelSet<GNode> > LogObimOSet;
    typedef detail::WorkSetMaster<GNode,LogObim,galois::ThreadSafeTwoLevelHash<GNode> > LogObimHSet;

    auto marker = LabelSetMarker<Graph>(graph);
    auto divIndexer = LabelDivIndexer<Graph>(graph);
    auto logIndexer = LabelLogIndexer<Graph>(graph);
    auto shrIndexer = LabelShrIndexer<Graph>(graph);

    galois::do_all_local(graph, Initialize(graph));
    switch(algo) {
    case Algo::gLpMSet:
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<MSet>(LabelSetMarker<Graph>(graph)));
      else
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<MSet>(LabelSetMarker<Graph>(graph)));
      break;
    case Algo::gLpOSet:
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<OSet>());
      else
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<OSet>());
      break;
    case Algo::gLpHSet:
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<HSet>());
      else
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<HSet>());
      break;
    case Algo::gLpDivObim:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObim>(divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObim>(divIndexer));
      break;
    case Algo::gLpDivObimMSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObimMSet>(marker,dummy,divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObimMSet>(marker,dummy,divIndexer));
      break;
    case Algo::gLpDivObimOSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObimOSet>(dummy,divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObimOSet>(dummy,divIndexer));
      break;
    case Algo::gLpDivObimHSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObimHSet>(dummy,divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObimHSet>(dummy,divIndexer));
      break;
    case Algo::gLpLogObim:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObim>(logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObim>(logIndexer));
      break;
    case Algo::gLpLogObimMSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObimMSet>(marker,dummy,logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObimMSet>(marker,dummy,logIndexer));
      break;
    case Algo::gLpLogObimOSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObimOSet>(dummy,logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObimOSet>(dummy,logIndexer));
      break;
    case Algo::gLpLogObimHSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObimHSet>(dummy,logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObimHSet>(dummy,logIndexer));
      break;
    case Algo::gLpShrObim:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObim>(shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObim>(shrIndexer));
      break;
    case Algo::gLpShrObimMSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObimMSet>(marker,dummy,shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObimMSet>(marker,dummy,shrIndexer));
      break;
    case Algo::gLpShrObimOSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObimOSet>(dummy,shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObimOSet>(dummy,shrIndexer));
      break;
    case Algo::gLpShrObimHSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObimHSet>(dummy,shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObimHSet>(dummy,shrIndexer));
      break;
    default:
      if (symmetricGraph) {
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<WL>());
      } else {
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<WL>());
      }
      break;
    } // end switch
  }
};
#endif

template<bool UseStaleValue>
struct LabelPropAlgo {
  struct LNode {
    typedef unsigned int component_type;
    unsigned int id;
    unsigned int comp;
    bool inSet;
    
    component_type component() { return comp; }
    bool isRep() { return id == comp; }
  };

  typedef typename galois::Graph::LC_CSR_Graph<LNode,void>
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type InnerGraph;
  typedef galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;
  typedef typename LNode::component_type component_type;

  template<typename G>
  void readGraph(G& graph) {
    readInOutGraph(graph);
  }

 
  struct Initialize {
    Graph& graph;

    Initialize(Graph& g): graph(g) { }
    void operator()(GNode n) const {
      LNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.comp = data.id;
      data.inSet = false;
    }
  };

  template<bool Forward,bool Backward>
  struct Process {
    typedef int tt_does_not_need_aborts;
    Graph& graph;
    Process(Graph& g): graph(g) { }

    template<typename Iterator,typename GetNeighbor>
    void update(LNode& sdata, Iterator ii, Iterator ei, GetNeighbor get, galois::UserContext<GNode>& ctx) {
      component_type newV;
      if(UseStaleValue) {
        newV = sdata.comp;
      }

      for (; ii != ei; ++ii) {
        GNode dst = get(ii);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        while (true) {
          component_type old = ddata.comp;
          if(!UseStaleValue) {
            newV = sdata.comp;
          }
          if (old <= newV) {
            if(traceWork) {
              *EmptyWork += 1;
            }
            break;
          }
          if(__sync_bool_compare_and_swap(&ddata.comp, old, newV)) {
            if(traceWork) {
              *GoodWork += 1;
            }
            ctx.push(dst);
            break;
          }
        }
      }
    }

    struct BackwardUpdate {
      Graph& graph;
      BackwardUpdate(Graph& g): graph(g) { }
      GNode operator()(typename Graph::in_edge_iterator ii) { return graph.getInEdgeDst(ii); }
    };

    struct ForwardUpdate {
      Graph& graph;
      ForwardUpdate(Graph& g): graph(g) { }
      GNode operator()(typename Graph::edge_iterator ii) { return graph.getEdgeDst(ii); }
    };

    //! Add the next edge between components to the worklist
    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      
      if (Backward) {
        update(sdata, graph.in_edge_begin(src, galois::MethodFlag::UNPROTECTED), graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED),
            BackwardUpdate(graph), ctx);
      } 
      if (Forward) {
        update(sdata, graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), graph.edge_end(src, galois::MethodFlag::UNPROTECTED),
            ForwardUpdate(graph), ctx);
      }
    }
  };

  void operator()(Graph& graph) {
    using namespace galois::WorkList;
    typedef dChunkedFIFO<256> WL;
    typedef dChunkedMarkingSetFIFO<LabelSetMarker<Graph>,256> MSet;
    typedef dChunkedTwoLevelSetFIFO<256> OSet;
    typedef dChunkedTwoLevelHashFIFO<256> HSet;
    typedef OrderedByIntegerMetric<LabelDivIndexer<Graph> > DivObim;
    typedef OrderedByIntegerMetric<LabelShrIndexer<Graph> > ShrObim;
    typedef OrderedByIntegerMetric<LabelLogIndexer<Graph> > LogObim;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,DivObim> DivObimMSet;
    typedef detail::WorkSetMaster<GNode,DivObim,galois::ThreadSafeTwoLevelSet<GNode> > DivObimOSet;
    typedef detail::WorkSetMaster<GNode,DivObim,galois::ThreadSafeTwoLevelHash<GNode> > DivObimHSet;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,ShrObim> ShrObimMSet;
    typedef detail::WorkSetMaster<GNode,ShrObim,galois::ThreadSafeTwoLevelSet<GNode> > ShrObimOSet;
    typedef detail::WorkSetMaster<GNode,ShrObim,galois::ThreadSafeTwoLevelHash<GNode> > ShrObimHSet;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,LogObim> LogObimMSet;
    typedef detail::WorkSetMaster<GNode,LogObim,galois::ThreadSafeTwoLevelSet<GNode> > LogObimOSet;
    typedef detail::WorkSetMaster<GNode,LogObim,galois::ThreadSafeTwoLevelHash<GNode> > LogObimHSet;

    auto marker = LabelSetMarker<Graph>(graph);
    auto divIndexer = LabelDivIndexer<Graph>(graph);
    auto logIndexer = LabelLogIndexer<Graph>(graph);
    auto shrIndexer = LabelShrIndexer<Graph>(graph);

    galois::do_all_local(graph, Initialize(graph));
#ifdef GALOIS_USE_EXP
    switch(algo) {
    case Algo::labelPropMSet:
    case Algo::staleLpMSet:
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<MSet>(LabelSetMarker<Graph>(graph)));
      else
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<MSet>(LabelSetMarker<Graph>(graph)));
      break;
    case Algo::labelPropOSet:
    case Algo::staleLpOSet:
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<OSet>());
      else
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<OSet>());
      break;
    case Algo::labelPropHSet:
    case Algo::staleLpHSet:
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<HSet>());
      else
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<HSet>());
      break;
    case Algo::labelPropDivObim:
    case Algo::staleLpDivObim:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObim>(divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObim>(divIndexer));
      break;
    case Algo::labelPropDivObimMSet:
    case Algo::staleLpDivObimMSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObimMSet>(marker,dummy,divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObimMSet>(marker,dummy,divIndexer));
      break;
    case Algo::labelPropDivObimOSet:
    case Algo::staleLpDivObimOSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObimOSet>(dummy,divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObimOSet>(dummy,divIndexer));
      break;
    case Algo::labelPropDivObimHSet:
    case Algo::staleLpDivObimHSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<DivObimHSet>(dummy,divIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<DivObimHSet>(dummy,divIndexer));
      break;
    case Algo::labelPropLogObim:
    case Algo::staleLpLogObim:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObim>(logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObim>(logIndexer));
      break;
    case Algo::labelPropLogObimMSet:
    case Algo::staleLpLogObimMSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObimMSet>(marker,dummy,logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObimMSet>(marker,dummy,logIndexer));
      break;
    case Algo::labelPropLogObimOSet:
    case Algo::staleLpLogObimOSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObimOSet>(dummy,logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObimOSet>(dummy,logIndexer));
      break;
    case Algo::labelPropLogObimHSet:
    case Algo::staleLpLogObimHSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<LogObimHSet>(dummy,logIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<LogObimHSet>(dummy,logIndexer));
      break;
    case Algo::labelPropShrObim:
    case Algo::staleLpShrObim:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObim>(shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObim>(shrIndexer));
      break;
    case Algo::labelPropShrObimMSet:
    case Algo::staleLpShrObimMSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObimMSet>(marker,dummy,shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObimMSet>(marker,dummy,shrIndexer));
      break;
    case Algo::labelPropShrObimOSet:
    case Algo::staleLpShrObimOSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObimOSet>(dummy,shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObimOSet>(dummy,shrIndexer));
      break;
    case Algo::labelPropShrObimHSet:
    case Algo::staleLpShrObimHSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      if(symmetricGraph)
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<ShrObimHSet>(dummy,shrIndexer));
      else
        galois::for_each_local(graph, Process<true, true>(graph), galois::wl<ShrObimHSet>(dummy,shrIndexer));
      break;
    default:
#endif
      if (symmetricGraph) {
        galois::for_each_local(graph, Process<true,false>(graph), galois::wl<WL>());
      } else {
        galois::for_each_local(graph, Process<true,true>(graph), galois::wl<WL>());
      }
#ifdef GALOIS_USE_EXP
      break;
    } // end switch
#endif
  }
};

#ifdef GALOIS_USE_EXP
//! Assumes symmetric graph
struct PullLPAlgo {
  struct LNode {
    typedef unsigned int component_type;
    unsigned int id;
    unsigned int comp;
    
    component_type component() { return comp; }
    bool isRep() { return id == comp; }
  };

  typedef galois::Graph::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;
  typedef LNode::component_type component_type;

  template<typename G>
  void readGraph(G& graph) {
    using namespace galois::Graph;
    if (symmetricGraph) {
      galois::Graph::readGraph(graph, inputFilename);
    } else {
      GALOIS_DIE("Graph type not supported");
    }
  }

  struct Initialize {
    Graph& graph;

    Initialize(Graph& g): graph(g) { }
    void operator()(GNode n) const {
      LNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.comp = data.id;
    }
  };

  struct Process {
    Graph& graph;
    Process(Graph& g): graph(g) { }

    //! Add the next edge between components to the worklist
    void operator()(const GNode& n, galois::UserContext<GNode>& ctx) {
      LNode& data = graph.getData(n);
      component_type old = data.comp, newV = data.comp;
      unsigned int good = 0, empty = 0;
     
      //! Pull from incoming neighbors to update my own label 
      auto inIt = graph.in_edge_begin(n), inIe = graph.in_edge_end(n);
      for( ; inIt != inIe; ++inIt) {
        GNode src = graph.getInEdgeDst(inIt);
        LNode& sData = graph.getData(src);

        if(sData.comp < newV) {
          newV = sData.comp;
          if(traceWork) {
            good += 1;
          }
        } else {
          if(traceWork) {
            empty += 1;
          }
        }
      }

      if(old != newV) {
        data.comp = newV;
        if(traceWork) {
          *GoodWork += good;
        }

        //! Ask my outgoing neighbors to update themselves
        auto outIt = graph.edge_begin(n), outIe = graph.edge_end(n);
        for( ; outIt != outIe; ++outIt) {
          GNode dst = graph.getEdgeDst(outIt);
          ctx.push(dst);
        }
      }

      if(traceWork) {
        *EmptyWork += empty;
      }
    }
  };

  void operator()(Graph& graph) {
    typedef galois::WorkList::dChunkedFIFO<256> WL;

    galois::do_all_local(graph, Initialize(graph));
    galois::for_each_local(graph, Process(graph), galois::wl<WL>());
  }
};

//! Assumes symmetric graph
struct PullLPCASAlgo {
  struct LNode {
    typedef unsigned int component_type;
    unsigned int id;
    unsigned int comp;
    bool inSet;
    
    component_type component() { return comp; }
    bool isRep() { return id == comp; }
  };

  typedef galois::Graph::LC_CSR_Graph<LNode,void>
    ::with_no_lockable<true>::type
    ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;
  typedef LNode::component_type component_type;

  template<typename G>
  void readGraph(G& graph) {
    using namespace galois::Graph;
    if (symmetricGraph) {
      galois::Graph::readGraph(graph, inputFilename);
    } else {
      GALOIS_DIE("Graph type not supported");
    }
  }

  struct Initialize {
    Graph& graph;

    Initialize(Graph& g): graph(g) { }
    void operator()(GNode n) const {
      LNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.comp = data.id;
      data.inSet = false;
    }
  };

  struct Process {
    typedef int tt_does_not_need_aborts;
    Graph& graph;
    Process(Graph& g): graph(g) { }

    //! Add the next edge between components to the worklist
    void operator()(const GNode& n, galois::UserContext<GNode>& ctx) {
      LNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      component_type old = data.comp, newV = data.comp;
     
      //! Pull from incoming neighbors to update my own label 
      auto inIt = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED), inIe = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED);
      for( ; inIt != inIe; ++inIt) {
        GNode src = graph.getInEdgeDst(inIt);
        LNode& sData = graph.getData(src, galois::MethodFlag::UNPROTECTED);

        if(sData.comp < newV) {
          newV = sData.comp;
          if(traceWork) {
            *GoodWork += 1;
          }
        } else {
          if(traceWork) {
            *EmptyWork += 1;
          }
        }
      }

      if(old != newV) {
        __sync_bool_compare_and_swap(&(data.comp), old, newV);

        //! Ask my outgoing neighbors to update themselves
        auto outIt = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED), outIe = graph.edge_end(n, galois::MethodFlag::UNPROTECTED);
        for( ; outIt != outIe; ++outIt) {
          GNode dst = graph.getEdgeDst(outIt);
          ctx.push(dst);
        }
      }
    }
  };

  void operator()(Graph& graph) {
    using namespace galois::WorkList;
    typedef dChunkedFIFO<256> WL;
    typedef dChunkedMarkingSetFIFO<LabelSetMarker<Graph>,256> MSet;
    typedef dChunkedTwoLevelSetFIFO<256> OSet;
    typedef dChunkedTwoLevelHashFIFO<256> HSet;
    typedef OrderedByIntegerMetric<LabelDivIndexer<Graph> > DivObim;
    typedef OrderedByIntegerMetric<LabelShrIndexer<Graph> > ShrObim;
    typedef OrderedByIntegerMetric<LabelLogIndexer<Graph> > LogObim;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,DivObim> DivObimMSet;
    typedef detail::WorkSetMaster<GNode,DivObim,galois::ThreadSafeTwoLevelSet<GNode> > DivObimOSet;
    typedef detail::WorkSetMaster<GNode,DivObim,galois::ThreadSafeTwoLevelHash<GNode> > DivObimHSet;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,ShrObim> ShrObimMSet;
    typedef detail::WorkSetMaster<GNode,ShrObim,galois::ThreadSafeTwoLevelSet<GNode> > ShrObimOSet;
    typedef detail::WorkSetMaster<GNode,ShrObim,galois::ThreadSafeTwoLevelHash<GNode> > ShrObimHSet;
    typedef detail::MarkingWorkSetMaster<GNode,LabelSetMarker<Graph>,LogObim> LogObimMSet;
    typedef detail::WorkSetMaster<GNode,LogObim,galois::ThreadSafeTwoLevelSet<GNode> > LogObimOSet;
    typedef detail::WorkSetMaster<GNode,LogObim,galois::ThreadSafeTwoLevelHash<GNode> > LogObimHSet;

    auto marker = LabelSetMarker<Graph>(graph);
    auto divIndexer = LabelDivIndexer<Graph>(graph);
    auto logIndexer = LabelLogIndexer<Graph>(graph);
    auto shrIndexer = LabelShrIndexer<Graph>(graph);

    galois::do_all_local(graph, Initialize(graph));
    switch(algo) {
    case Algo::pullLpMSet:
      galois::for_each_local(graph, Process(graph), galois::wl<MSet>(LabelSetMarker<Graph>(graph)));
      break;
    case Algo::pullLpOSet:
      galois::for_each_local(graph, Process(graph), galois::wl<OSet>());
      break;
    case Algo::pullLpHSet:
      galois::for_each_local(graph, Process(graph), galois::wl<HSet>());
      break;
    case Algo::pullLpDivObim:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<DivObim>(divIndexer));
      break;
    case Algo::pullLpDivObimMSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<DivObimMSet>(marker,dummy,divIndexer));
      break;
    case Algo::pullLpDivObimOSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<DivObimOSet>(dummy,divIndexer));
      break;
    case Algo::pullLpDivObimHSet:
      std::cout << "using priority scheduling based on (label / 1000).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<DivObimHSet>(dummy,divIndexer));
      break;
    case Algo::pullLpLogObim:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<LogObim>(logIndexer));
      break;
    case Algo::pullLpLogObimMSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<LogObimMSet>(marker,dummy,logIndexer));
      break;
    case Algo::pullLpLogObimOSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<LogObimOSet>(dummy,logIndexer));
      break;
    case Algo::pullLpLogObimHSet:
      std::cout << "using priority scheduling based on floor(log(label+1)).\n";
      galois::for_each_local(graph, Process(graph), galois::wl<LogObimHSet>(dummy,logIndexer));
      break;
    case Algo::pullLpShrObim:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      galois::for_each_local(graph, Process(graph), galois::wl<ShrObim>(shrIndexer));
      break;
    case Algo::pullLpShrObimMSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      galois::for_each_local(graph, Process(graph), galois::wl<ShrObimMSet>(marker,dummy,shrIndexer));
      break;
    case Algo::pullLpShrObimOSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      galois::for_each_local(graph, Process(graph), galois::wl<ShrObimOSet>(dummy,shrIndexer));
      break;
    case Algo::pullLpShrObimHSet:
      std::cout << "using priority scheduling based on (label >> " << delta << ").\n";
      std::cout << "default delta may not be the best for performance.\n";
      galois::for_each_local(graph, Process(graph), galois::wl<ShrObimHSet>(dummy,shrIndexer));
      break;
    default:
      galois::for_each_local(graph, Process(graph), galois::wl<WL>());
      break;
    } // end switch
  }
};
#endif

struct AsyncOCAlgo {
  typedef galois::Graph::OCImmutableEdgeGraph<Node,void> Graph;
  typedef Graph::GraphNode GNode;

  template<typename G>
  void readGraph(G& graph) {
    readInOutGraph(graph);
  }

  struct Merge {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    galois::Statistic& emptyMerges;
    Merge(galois::Statistic& e): emptyMerges(e) { }

    //! Add the next edge between components to the worklist
    template<typename GTy>
    void operator()(GTy& graph, const GNode& src) const {
      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      for (typename GTy::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
          ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        if (symmetricGraph && src >= dst)
          continue;

        if (!sdata.merge(&ddata))
          emptyMerges += 1;
      }
    }
  };

  void operator()(Graph& graph) {
    galois::Statistic emptyMerges("EmptyMerges");
    
    galois::GraphChi::vertexMap(graph, Merge(emptyMerges), memoryLimit);
  }
};

/**
 * Like synchronous algorithm, but if we restrict path compression (as done is
 * @link{UnionFindNode}), we can perform unions and finds concurrently.
 */
struct AsyncAlgo {
  typedef galois::Graph::LC_CSR_Graph<Node,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  template<typename G>
  void readGraph(G& graph) { galois::Graph::readGraph(graph, inputFilename); }

  struct Merge {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    Graph& graph;
    galois::Statistic& emptyMerges;
    Merge(Graph& g, galois::Statistic& e): graph(g), emptyMerges(e) { }

    //! Add the next edge between components to the worklist
    void operator()(const GNode& src, galois::UserContext<GNode>&) const {
      (*this)(src);
    }

    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      for (Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
          ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        if (symmetricGraph && src >= dst)
          continue;

        if (!sdata.merge(&ddata)) {
          emptyMerges += 1;
        } else if (traceWork) {
          *GoodWork += 1;
        }
      }
    }
  };

  void operator()(Graph& graph) {
    galois::Statistic emptyMerges("EmptyMerges");
#ifdef GALOIS_USE_EXP
    if(algo == Algo::asyncOSet) {
      galois::for_each_local(graph, Merge(graph, emptyMerges), galois::wl<galois::WorkList::dChunkedTwoLevelSetFIFO<32> >());
    } else if(algo == Algo::asyncHSet) {
      galois::for_each_local(graph, Merge(graph, emptyMerges), galois::wl<galois::WorkList::dChunkedTwoLevelHashFIFO<32> >());
    } else {
#endif
      galois::for_each_local(graph, Merge(graph, emptyMerges));
#ifdef GALOIS_USE_EXP
    }
#endif
  }
};

/**
 * Improve performance of async algorithm by following machine topology.
 */
struct BlockedAsyncAlgo {
  typedef galois::Graph::LC_CSR_Graph<Node,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  struct WorkItem {
    GNode src;
    Graph::edge_iterator start;
  };

  template<typename G>
  void readGraph(G& graph) { galois::Graph::readGraph(graph, inputFilename); }

  struct Merge {
    typedef int tt_does_not_need_aborts;

    Graph& graph;
    galois::InsertBag<WorkItem>& items;

    //! Add the next edge between components to the worklist
    template<bool MakeContinuation, int Limit, typename Pusher>
    void process(const GNode& src, const Graph::edge_iterator& start, Pusher& pusher) const {
      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      int count = 1;
      for (Graph::edge_iterator ii = start, ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
          ii != ei; 
          ++ii, ++count) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        if (symmetricGraph && src >= dst)
          continue;

        if (sdata.merge(&ddata)) {
          if (Limit == 0 || count != Limit)
            continue;
        }

        if (MakeContinuation || (Limit != 0 && count == Limit)) {
          WorkItem item = { src, ii + 1 };
          pusher.push(item);
          break;
        }
      }
    }

    void operator()(const GNode& src) const {
      Graph::edge_iterator start = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
      if (galois::Substrate::ThreadPool::getPackage() == 0) {
        process<true, 0>(src, start, items);
      } else {
        process<true, 1>(src, start, items);
      }
    }

    void operator()(const WorkItem& item, galois::UserContext<WorkItem>& ctx) const {
      process<true, 0>(item.src, item.start, ctx);
    }
  };

  void operator()(Graph& graph) {
    galois::InsertBag<WorkItem> items;
    Merge merge = { graph, items };
    galois::do_all_local(graph, merge, galois::loopname("Initialize"));
    galois::for_each_local(items, merge,
        galois::loopname("Merge"), galois::wl<galois::WorkList::dChunkedFIFO<128> >());
  }
};

template<typename Graph>
struct is_bad {
  typedef typename Graph::GraphNode GNode;
  Graph& graph;

  is_bad(Graph& g): graph(g) { }

  bool operator()(const GNode& n) const {
    typedef typename Graph::node_data_reference node_data_reference;

    node_data_reference me = graph.getData(n);
    for (typename Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      node_data_reference data = graph.getData(dst);
      if (data.component() != me.component()) {
        std::cerr << "not in same component: "
          << me.id << " (" << me.component() << ")"
          << " and "
          << data.id << " (" << data.component() << ")"
          << "\n";
        return true;
      }
    }
    return false;
  }
};

template<typename Graph>
bool verify(Graph& graph,
    typename std::enable_if<galois::Graph::is_segmented<Graph>::value>::type* = 0) {
  return true;
}

template<typename Graph>
bool verify(Graph& graph,
    typename std::enable_if<!galois::Graph::is_segmented<Graph>::value>::type* = 0) {
  return galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad<Graph>(graph)) == graph.end();
}

template<typename EdgeTy, typename Algo, typename CGraph>
void writeComponent(Algo& algo, CGraph& cgraph, typename CGraph::node_data_type::component_type component) {
  typedef typename CGraph::template with_edge_data<EdgeTy>::type Graph;

  if (std::is_same<Graph,CGraph>::value) {
    doWriteComponent(cgraph, component);
  } else {
    // copy node data from cgraph
    Graph graph;
    algo.readGraph(graph);
    typename Graph::iterator cc = graph.begin();
    for (typename CGraph::iterator ii = cgraph.begin(), ei = cgraph.end(); ii != ei; ++ii, ++cc) {
      graph.getData(*cc) = cgraph.getData(*ii);
    }
    doWriteComponent(graph, component);
  }
}

template<typename Graph>
void doWriteComponent(Graph& graph, typename Graph::node_data_type::component_type component,
    typename std::enable_if<galois::Graph::is_segmented<Graph>::value>::type* = 0) {
  GALOIS_DIE("Writing component not supported for this graph");
}

template<typename Graph>
void doWriteComponent(Graph& graph, typename Graph::node_data_type::component_type component,
    typename std::enable_if<!galois::Graph::is_segmented<Graph>::value>::type* = 0) {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_reference node_data_reference;

  // set id to 1 if node is in component
  size_t numEdges = 0;
  size_t numNodes = 0;
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    node_data_reference data = graph.getData(*ii);
    data.id = data.component() == component ? 1 : 0;
    if (data.id) {
      size_t degree = std::distance(graph.edge_begin(*ii), graph.edge_end(*ii));
      numEdges += degree;
      numNodes += 1;
    }
  }

  typedef galois::Graph::FileGraphWriter Writer;
  typedef galois::LargeArray<typename Graph::edge_data_type> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Writer p;
  EdgeData edgeData;
  p.setNumNodes(numNodes);
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0); 
  edgeData.create(numEdges);

  p.phase1();
  // partial sums of ids: id == new_index + 1
  typename Graph::node_data_type* prev = 0;
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    node_data_reference data = graph.getData(*ii);
    if (prev)
      data.id = prev->id + data.id;
    if (data.component() == component) {
      size_t degree = std::distance(graph.edge_begin(*ii), graph.edge_end(*ii));
      size_t sid = data.id - 1;
      assert(sid < numNodes);
      p.incrementDegree(sid, degree);
    }
    
    prev = &data;
  }

  assert(!prev || prev->id == numNodes);

  if (largestComponentFilename != "") {
    p.phase2();
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      node_data_reference data = graph.getData(*ii);
      if (data.component() != component)
        continue;

      size_t sid = data.id - 1;

      for (typename Graph::edge_iterator jj = graph.edge_begin(*ii),
          ej = graph.edge_end(*ii); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        node_data_reference ddata = graph.getData(dst);
        size_t did = ddata.id - 1;

        //assert(ddata.component == component);
        assert(sid < numNodes && did < numNodes);
        if (EdgeData::has_value)
          edgeData.set(p.addNeighbor(sid, did), graph.getEdgeData(jj));
        else
          p.addNeighbor(sid, did);
      }
    }

    edge_value_type* rawEdgeData = p.finish<edge_value_type>();
    if (EdgeData::has_value)
      std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), rawEdgeData);

    std::cout
      << "Writing largest component to " << largestComponentFilename
      << " (nodes: " << numNodes << " edges: " << numEdges << ")\n";

    p.toFile(largestComponentFilename);
  }

  if (permutationFilename != "") {
    std::ofstream out(permutationFilename);
    size_t oid = 0;
    std::cout << "Writing permutation to " << permutationFilename << "\n";
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++oid) {
      node_data_reference data = graph.getData(*ii);
      out << oid << ",";
      if (data.component() != component) {
        ;
      } else {
        out << data.id - 1;
      }
      out << "\n";
    }
  }
}

template<typename Graph>
struct CountLargest {
  typedef typename Graph::node_data_type::component_type component_type;
  typedef std::map<component_type,int> Map;
  typedef typename Graph::GraphNode GNode;

  struct Accums {
    galois::GMapElementAccumulator<Map> map;
    galois::GAccumulator<size_t> reps;
  };

  Graph& graph;
  Accums& accums;
  
  CountLargest(Graph& g, Accums& accums): graph(g), accums(accums) { }
  
  void operator()(const GNode& x) const {
    typename Graph::node_data_reference n = graph.getData(x, galois::MethodFlag::UNPROTECTED);
    if (n.isRep()) {
      accums.reps += 1;
      return;
    }

    // Don't add reps to table to avoid adding components of size 1
    accums.map.update(n.component(), 1);
  }
};

template<typename Graph>
struct ComponentSizePair {
  typedef typename Graph::node_data_type::component_type component_type;

  component_type component;
  int size;

  struct Max {
    ComponentSizePair operator()(const ComponentSizePair& a, const ComponentSizePair& b) const {
      if (a.size > b.size)
        return a;
      return b;
    }
  };

  ComponentSizePair(): component(0), size(0) { }
  ComponentSizePair(component_type c, int s): component(c), size(s) { }
};

template<typename Graph>
struct ReduceMax {
  typedef typename Graph::node_data_type::component_type component_type;
  typedef galois::GSimpleReducible<ComponentSizePair<Graph>,typename ComponentSizePair<Graph>::Max> Accum;

  Accum& accum;

  ReduceMax(Accum& accum): accum(accum) { }

  void operator()(const std::pair<component_type,int>& x) const {
    accum.update(ComponentSizePair<Graph>(x.first, x.second));
  }
};

template<typename Graph>
typename Graph::node_data_type::component_type findLargest(Graph& graph) {
  typedef CountLargest<Graph> CL;
  typedef ReduceMax<Graph> RM;

  typename CL::Accums accums;
  galois::do_all_local(graph, CL(graph, accums));
  typename CL::Map& map = accums.map.reduce();
  size_t reps = accums.reps.reduce();

  typename RM::Accum accumMax;
  galois::do_all(map.begin(), map.end(), RM(accumMax));
  ComponentSizePair<Graph>& largest = accumMax.reduce();

  // Compensate for dropping representative node of components
  double ratio = graph.size() - reps + map.size();
  size_t largestSize = largest.size + 1;
  if (ratio)
    ratio = largestSize / ratio;

  std::cout << "Total components: " << reps << "\n";
  std::cout << "Number of non-trivial components: " << map.size()
    << " (largest size: " << largestSize << " [" << ratio << "])\n";

  return largest.component;
}

template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;

  Algo algo;
  Graph graph;

  algo.readGraph(graph);
  std::cout << "Read " << graph.size() << " nodes\n"; 

  unsigned int id = 0;
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++id) {
    graph.getData(*ii).id = id;
  }
  
  galois::preAlloc(numThreads + (2 * graph.size() * sizeof(typename Graph::node_data_type)) / galois::Runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  T.start();
  algo(graph);
  T.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify || largestComponentFilename != "" || permutationFilename != "") {
    auto component = findLargest(graph);
    if (!verify(graph)) {
      GALOIS_DIE("verification failed");
    }
    if (component && (largestComponentFilename != "" || permutationFilename != "")) {
      switch (writeEdgeType) {
        case OutputEdgeType::void_: writeComponent<void>(algo, graph, component); break;
        case OutputEdgeType::int32_: writeComponent<uint32_t>(algo, graph, component); break;
        case OutputEdgeType::int64_: writeComponent<uint64_t>(algo, graph, component); break;
        default: abort();
      }
    }
  }
}

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if(traceWork) {
    GoodWork = new galois::Statistic("GoodWork");
    EmptyWork = new galois::Statistic("EmptyWork");
  }

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::asyncOc: run<AsyncOCAlgo>(); break;
    case Algo::async: run<AsyncAlgo>(); break;
    case Algo::blockedasync: run<BlockedAsyncAlgo>(); break;
    case Algo::labelProp: run<LabelPropAlgo<false> >(); break;
    case Algo::staleLp: run<LabelPropAlgo<true> >(); break;
    case Algo::serial: run<SerialAlgo>(); break;
    case Algo::synchronous: run<SynchronousAlgo>(); break;
#ifdef GALOIS_USE_EXP
    case Algo::asyncOSet: run<AsyncAlgo>(); break;
    case Algo::asyncHSet: run<AsyncAlgo>(); break;
    case Algo::labelPropMSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropOSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropHSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropDivObim: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropDivObimMSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropDivObimOSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropDivObimHSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropLogObim: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropLogObimMSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropLogObimOSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropLogObimHSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropShrObim: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropShrObimMSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropShrObimOSet: run<LabelPropAlgo<false> >(); break;
    case Algo::labelPropShrObimHSet: run<LabelPropAlgo<false> >(); break;
    case Algo::gLp: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpMSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpOSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpHSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpDivObim: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpDivObimMSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpDivObimOSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpDivObimHSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpLogObim: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpLogObimMSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpLogObimOSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpLogObimHSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpShrObim: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpShrObimMSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpShrObimOSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::gLpShrObimHSet: run<LabelPropNoCasAlgo>(); break;
    case Algo::staleLpMSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpOSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpHSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpDivObim: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpDivObimMSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpDivObimOSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpDivObimHSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpLogObim: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpLogObimMSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpLogObimOSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpLogObimHSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpShrObim: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpShrObimMSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpShrObimOSet: run<LabelPropAlgo<true> >(); break;
    case Algo::staleLpShrObimHSet: run<LabelPropAlgo<true> >(); break;
    case Algo::gPullLp: run<PullLPAlgo>(); break;
    case Algo::pullLp: run<PullLPCASAlgo>(); break;
    case Algo::pullLpMSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpOSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpHSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpDivObim: run<PullLPCASAlgo>(); break;
    case Algo::pullLpDivObimMSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpDivObimOSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpDivObimHSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpLogObim: run<PullLPCASAlgo>(); break;
    case Algo::pullLpLogObimMSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpLogObimOSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpLogObimHSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpShrObim: run<PullLPCASAlgo>(); break;
    case Algo::pullLpShrObimMSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpShrObimOSet: run<PullLPCASAlgo>(); break;
    case Algo::pullLpShrObimHSet: run<PullLPCASAlgo>(); break;
    case Algo::graphchi: run<GraphChiAlgo>(); break;
    case Algo::graphlab: run<GraphLabAlgo>(); break;
    case Algo::ligraChi: run<LigraAlgo<true> >(); break;
    case Algo::ligra: run<LigraAlgo<false> >(); break;
#endif
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  if(traceWork) {
    delete GoodWork;
    delete EmptyWork;
  }

  return 0;
}

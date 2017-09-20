/** BFS -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * BFS.
 *
 * @author <ahassaan@ices.utexas.edu>
 */


#ifndef _BFS_H_
#define _BFS_H_

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>
#include "galois/Graphs/LC_InlineEdge_Graph.h"

// #define GALOIS_USE_MIC_CSR_IMPL
#undef GALOIS_USE_MIC_CSR_IMPL

#ifdef GALOIS_USE_MIC_CSR_IMPL
  #include "galois/Graphs/LC_CSR_MIC_Graph.h"
#else
  #include "galois/Graphs/LC_CSR_Graph.h"
#endif

#include "galois/Accumulator.h"
#include "galois/DoAllWrap.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/Graphs/Util.h"
#include "galois/Graphs/FileGraph.h"
#include "llvm/Support/CommandLine.h"

#include "galois/Substrate/CacheLineStorage.h"
#include "galois/Runtime/Sampling.h"
#include "galois/Substrate/CompilerSpecific.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* const name = "Breadth First Search";
static const char* const desc = "Computes BFS levels in a graph";
static const char* const url = "bfs";

static cll::opt<unsigned int> startId("startnode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

static const unsigned BFS_LEVEL_INFINITY = (1 << 20) - 1;

static const unsigned DEFAULT_CHUNK_SIZE = 16; 

#if 0
struct NodeData {
  unsigned data;

  NodeData (): data (0) {}
  NodeData (unsigned _data): data (_data) {}

  inline operator unsigned& () { return data; }
  inline operator const unsigned& () const { return data; }

};

struct NodeDataCacheLine: public galois::runtime::LL::CacheLineStorage<unsigned> {
  typedef galois::runtime::LL::CacheLineStorage<unsigned> Super_ty;

  NodeDataCacheLine (): Super_ty (0) {}
  NodeDataCacheLine (unsigned _data): Super_ty (_data) {}

  inline operator unsigned& () { return Super_ty::data; }
  inline operator const unsigned& () const { return Super_ty::data; }

};
#endif

typedef galois::GAccumulator<unsigned> ParCounter;


template <typename G, bool IS_BFS=true> 
class BFS_SSSP_Base {

public:

  using Graph = G;
  using GNode = typename Graph::GraphNode;
  using ND = typename Graph::node_data_type;

  // TODO: change to a generic name
  virtual size_t runBFS (Graph& graph, GNode& startNode) = 0;

  virtual const std::string getVersion () const = 0;

  virtual void initGraph (const std::string& filename, Graph& graph) const {
    std::cout << "Reading graph from file: " << filename << std::endl;

    galois::StatTimer t_read ("Time for reading from file: ");

    t_read.start ();
    galois::graphs::readGraph(graph, filename);
    t_read.stop ();


    galois::StatTimer t_init ("Time for initializing node data and counting edges: ");


    t_init.start ();
    unsigned numNodes = graph.size ();
    ParCounter numEdges;

    galois::do_all_choice (galois::runtime::makeLocalRange(graph),
        [&numEdges,&graph] (GNode n) {
          graph.getData (n, galois::MethodFlag::UNPROTECTED) = ND (BFS_LEVEL_INFINITY);
          numEdges += std::distance (graph.edge_begin (n, galois::MethodFlag::UNPROTECTED),  graph.edge_end (n, galois::MethodFlag::UNPROTECTED));
          // std::cout << "Degree: " << graph.edge_end (n, galois::MethodFlag::UNPROTECTED) - graph.edge_begin (n, galois::MethodFlag::UNPROTECTED) << std::endl;
        },
        std::make_tuple (
          galois::loopname ("node-data-init"),
          galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    t_init.stop();
    std::cout << "Graph read with nodes=" << numNodes << ", edges=" << numEdges.reduce () << std::endl;
  }

  virtual GNode getStartNode (Graph& graph, unsigned startId) const {
    assert (startId < graph.size ());

    unsigned id = 0;
    GNode startNode = 0;
    for (auto i = graph.begin (), ei = graph.end ();
        i != ei; ++i) {

      if (id == startId) {
        startNode = *i;
        break;
      }
      ++id;
    }

    return startNode;
  }

  template<bool useOneL, typename I>
  static ND getEdgeWeight(Graph& graph, I ii, typename std::enable_if<useOneL>::type* = nullptr) {
    return 1;
  }

  template<bool useOneL, typename I>
  static ND getEdgeWeight(Graph& graph, I ii, typename std::enable_if<!useOneL>::type* = nullptr) {
    return graph.getEdgeData(ii);
  }

  bool verify (Graph& graph, GNode& startNode) const {

    galois::StatTimer t_verify ("Verification time: ");

    t_verify.start ();
    galois::substrate::PerThreadStorage<bool> result;
    for (unsigned i = 0; i < result.size (); ++i) {
      *result.getRemote(i) = true;
    }

    ParCounter numUnreachable;

    galois::do_all_choice (
        galois::runtime::makeLocalRange(graph),
        [&graph, &numUnreachable, &result, &startNode] (GNode n) {
          const ND srcLevel = graph.getData (n, galois::MethodFlag::UNPROTECTED);
          if (srcLevel >= BFS_LEVEL_INFINITY) { 
            numUnreachable += 1;
            // std::cerr << "BAD Level value >= INFINITY at node " << (*i) << std::endl;
            // result = false;
          }


          for (auto e = graph.edge_begin (n, galois::MethodFlag::UNPROTECTED)
            , ende = graph.edge_end (n, galois::MethodFlag::UNPROTECTED); e != ende; ++e) {
            
            GNode dst = graph.getEdgeDst (e);
            const ND dstLevel = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
            const ND edgeWeight = getEdgeWeight<IS_BFS> (graph, e);

            if (dstLevel > (srcLevel + edgeWeight) || (dst != startNode && dstLevel == 0)) {
              *result.getLocal () = false;
              std::cerr << "BAD Level value=" << dstLevel << " at neighbor " << dst << " of node " 
                << n << std::endl; 
            }
          }
        },
        std::make_tuple (
          galois::loopname ("bfs-verify"),
          galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    if (numUnreachable.reduce () > 0) {
      std::cerr << "WARNING: " << numUnreachable.reduce () << " nodes were unreachable. "
        << "This is an error if the input is strongly connected" << std::endl;
    }

    bool conj = true;
    for (unsigned i = 0; i < result.size (); ++i) {
      conj = conj && *result.getRemote (i);
    }

    t_verify.stop ();

    return conj;
  }

public:

  virtual void run (int argc, char* argv[]) {
    LonestarStart (argc, argv, name, desc, url);
    galois::StatManager sm;

    Graph graph;

    std::cout << "Size of node data= " << sizeof (ND) << std::endl;

    initGraph (filename, graph);

    GNode startNode = getStartNode (graph, startId);

    std::cout << "Running <" << getVersion () << "> BFS" << std::endl;
    std::cout << "start node = " << startNode << std::endl;

    galois::StatTimer timer;


    // for node based versions
    // galois::preAlloc (galois::getActiveThreads () + 8*graph.size ()/galois::runtime::MM::hugePageSize);
    // // for edge based versions
    galois::preAlloc ((galois::getActiveThreads () * 10 * graph.sizeEdges ())/galois::runtime::pagePoolSize());
    galois::reportPageAlloc("MeminfoPre");

    timer.start ();
    galois::runtime::beginSampling ();
    size_t numIter = runBFS (graph, startNode);
    galois::runtime::endSampling ();
    timer.stop ();
    galois::reportPageAlloc("MeminfoPost");

    std::cout << "BFS " << getVersion () << " iterations=" << numIter << std::endl;

    if (!skipVerify) {
      if (verify (graph, startNode)) {
        std::cout << "OK. Result verified" << std::endl;

      } else {
        std::cerr << "ERROR. Result incorrect" << std::endl;
        abort ();
      }
    }

  }

  struct Update {
    GNode node;
    ND level;

    Update (const GNode& node, const ND& level)
      : node (node), level (level) 
    {}

    friend std::ostream& operator << (std::ostream& out, const Update& up) {
      out << "(node:" << up.node << ",level:" << up.level << ")";
      return out;
    }
  };

  struct GetLevel {

    ND operator () (const Update& up) const {
      return up.level;
    }
  };

  struct Comparator {
    bool operator () (const Update& left, const Update& right) const {
      int d = left.level - right.level;

      if (d == 0) {
        // FIXME: assuming nodes are actually integer like
        d = left.node - right.node;
      }

      return (d < 0);
    }
  };

  struct VisitNhood {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    Graph& graph;

    explicit VisitNhood (Graph& graph): graph (graph) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) {

      // just like DES, we only lock the node being updated, but not its
      // outgoing neighbors
      graph.getData (up.node, galois::MethodFlag::WRITE);
    }
  };

  struct OpFunc {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    typedef int tt_does_not_need_aborts; // used by LevelExecBFS

    Graph& graph;
    ParCounter& numIter;

    OpFunc (Graph& graph, ParCounter& numIter): graph (graph), numIter (numIter) {}

    template <typename C>
      void operator () (const Update& up, C& ctx) const {

        if (graph.getData (up.node, galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {

          graph.getData (up.node, galois::MethodFlag::UNPROTECTED) = up.level;


          for (auto ni = graph.edge_begin (up.node, galois::MethodFlag::UNPROTECTED)
              , eni = graph.edge_end (up.node, galois::MethodFlag::UNPROTECTED); ni != eni; ++ni) {

            GNode dst = graph.getEdgeDst (ni);

            if (graph.getData (dst, galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {
              ctx.push (Update (dst, up.level + 1));
            }
          }

        }

        numIter += 1;
      }

  };

  struct OpFuncSpec: public OpFunc {

    OpFuncSpec (Graph& graph, ParCounter& numIter): OpFunc (graph, numIter) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) const {
      auto& graph = OpFunc::graph;
      auto& ndata = graph.getData (up.node, galois::MethodFlag::UNPROTECTED);


      if (ndata == BFS_LEVEL_INFINITY) {

        ndata = up.level;

        auto undo = [this, &graph, up] (void) {
          graph.getData (up.node, galois::MethodFlag::UNPROTECTED) = BFS_LEVEL_INFINITY;
        };

        ctx.addUndoAction (undo);

        for (auto ni = graph.edge_begin (up.node, galois::MethodFlag::UNPROTECTED)
            , eni = graph.edge_end (up.node, galois::MethodFlag::UNPROTECTED); ni != eni; ++ni) {

          GNode dst = graph.getEdgeDst (ni);
          auto w = getEdgeWeight<IS_BFS> (graph, ni);

          ctx.push (Update (dst, up.level + w));
        }

      }

      auto inc = [this] (void) {
        OpFunc::numIter += 1;
      };

      ctx.addCommitAction (inc);
    }
  };

  // relies on round based execution of IKDG executor
  struct OpFuncLocalMin: public OpFunc {

    OpFuncLocalMin (Graph& graph, ParCounter& numIter): OpFunc (graph, numIter) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) {
      auto& graph = OpFunc::graph;
      auto& ndata = graph.getData (up.node, galois::MethodFlag::UNPROTECTED);
      if (ndata > up.level) {

        ndata = up.level;


        for (auto ni = graph.edge_begin (up.node, galois::MethodFlag::UNPROTECTED)
            , eni = graph.edge_end (up.node, galois::MethodFlag::UNPROTECTED); ni != eni; ++ni) {

          GNode dst = graph.getEdgeDst (ni);
          auto w = getEdgeWeight<IS_BFS> (graph, ni);

          if (graph.getData (dst, galois::MethodFlag::UNPROTECTED) > (up.level + w)) {
            ctx.push (Update (dst, up.level + w));
          }
        }

      }

      OpFunc::numIter += 1;
    }

  };

};

#ifdef GALOIS_USE_MIC_CSR_IMPL
  typedef typename galois::graphs::LC_CSR_MIC_Graph<unsigned, void>
#else
  typedef typename galois::graphs::LC_CSR_Graph<unsigned, void>
#endif
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<false>::type BFSgraph; // TODO: make lockable a template parameter for different BFS implementations

class BFS: public BFS_SSSP_Base<BFSgraph> {

public:
  typedef typename Graph::GraphNode GNode;

  template <typename WL, typename T>
  GALOIS_ATTRIBUTE_PROF_NOINLINE static void addToWL (
      WL& workList, 
      const T& val) {
    workList.push_back (val);
  }


  //! @return number of adds
  template <bool doLock, typename WL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE static unsigned bfsOperator (
      Graph& graph, GNode& src, 
      WL& workList) {


    unsigned numAdds = 0;

    const unsigned srcLevel = graph.getData (src, (doLock ? galois::MethodFlag::WRITE : galois::MethodFlag::UNPROTECTED));

    // putting a loop to acquire locks. For now, edge_begin does not acquire locks on neighbors,
    // which it should
    if (doLock) {
      for (auto ni = graph.edge_begin (src, galois::MethodFlag::WRITE)
          , eni = graph.edge_end (src, galois::MethodFlag::WRITE); ni != eni; ++ni) {

        GNode dst = graph.getEdgeDst (ni);
        graph.getData (dst, galois::MethodFlag::WRITE);
      }
    }


    for (auto ni = graph.edge_begin (src, galois::MethodFlag::UNPROTECTED), eni = graph.edge_end (src, galois::MethodFlag::UNPROTECTED);
        ni != eni; ++ni) {

      GNode dst = graph.getEdgeDst (ni);

      ND& dstData = graph.getData (dst, galois::MethodFlag::UNPROTECTED); // iterator should already have acquired locks on neighbors
      if (dstData == BFS_LEVEL_INFINITY) {
        dstData = srcLevel + 1;

        // workList.push_back (dst);
        addToWL (workList, dst);
        ++numAdds;
      }

    }

    return numAdds;
  }


};

using SsspGraph =  typename galois::graphs::LC_InlineEdge_Graph<unsigned int, uint32_t>
    ::with_no_lockable<false>::type
    ::with_numa_alloc<true>::type;

struct SSSP: public BFS_SSSP_Base<SsspGraph, false> {

};

#endif // _BFS_H_

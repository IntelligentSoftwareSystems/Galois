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

#define GALOIS_USE_MIC_CSR_IMPL
// #undef GALOIS_USE_MIC_CSR_IMPL

#ifdef GALOIS_USE_MIC_CSR_IMPL
  #include "Galois/Graph/LC_CSR_Graph_MIC.h"
#else
  #include "Galois/Graph/LC_CSR_Graph.h"
#endif

#include "Galois/Accumulator.h"
#include "Galois/DoAllWrap.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/Graph/Util.h"
#include "Galois/Graph/FileGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Galois/Runtime/ll/CacheLineStorage.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* const name = "Breadth First Search";
static const char* const desc = "Computes BFS levels in a graph";
static const char* const url = "bfs";

static cll::opt<unsigned int> startId("startnode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

static const unsigned BFS_LEVEL_INFINITY = (1 << 20) - 1;

static const unsigned CHUNK_SIZE = 128; 

#if 0
struct NodeData {
  unsigned data;

  NodeData (): data (0) {}
  NodeData (unsigned _data): data (_data) {}

  inline operator unsigned& () { return data; }
  inline operator const unsigned& () const { return data; }

};

struct NodeDataCacheLine: public Galois::Runtime::LL::CacheLineStorage<unsigned> {
  typedef Galois::Runtime::LL::CacheLineStorage<unsigned> Super_ty;

  NodeDataCacheLine (): Super_ty (0) {}
  NodeDataCacheLine (unsigned _data): Super_ty (_data) {}

  inline operator unsigned& () { return Super_ty::data; }
  inline operator const unsigned& () const { return Super_ty::data; }

};
#endif

typedef Galois::GAccumulator<unsigned> ParCounter;

template <typename ND>
class BFS {

public:
  typedef typename Galois::Graph::LC_CSR_Graph<ND, void>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef typename Graph::GraphNode GNode;
  typedef ND NodeData_ty;

protected:
  virtual size_t runBFS (Graph& graph, GNode& startNode) = 0;

  virtual const std::string getVersion () const = 0;

  virtual void initGraph (const std::string& filename, Graph& graph) const {
    std::cout << "Reading graph from file: " << filename << std::endl;

    Galois::StatTimer t_read ("Time for reading from file: ");

    t_read.start ();
    Galois::Graph::readGraph(graph, filename);
    t_read.stop ();


    Galois::StatTimer t_init ("Time for initializing node data and counting edges: ");


    t_init.start ();
    unsigned numNodes = graph.size ();
    ParCounter numEdges;

    Galois::do_all_choice (Galois::Runtime::makeLocalRange(graph),
        [&numEdges,&graph] (GNode n) {
          graph.getData (n, Galois::NONE) = ND (BFS_LEVEL_INFINITY);
          numEdges += graph.edge_end (n, Galois::NONE) - graph.edge_begin (n, Galois::NONE);
          // std::cout << "Degree: " << graph.edge_end (n, Galois::NONE) - graph.edge_begin (n, Galois::NONE) << std::endl;
        },
        "node-data-init",
        Galois::doall_chunk_size<CHUNK_SIZE> ());

    t_init.stop();
    std::cout << "Graph read with nodes=" << numNodes << ", edges=" << numEdges.reduce () << std::endl;
  }

  virtual GNode getStartNode (const Graph& graph, unsigned startId) const {
    assert (startId < graph.size ());

    unsigned id = 0;
    GNode startNode = 0;
    for (typename Graph::iterator i = graph.begin (), ei = graph.end ();
        i != ei; ++i) {

      if (id == startId) {
        startNode = *i;
        break;
      }
      ++id;
    }

    return startNode;
  }

  virtual bool verify (Graph& graph, GNode& startNode) const {

    Galois::StatTimer t_verify ("Verification time: ");

    t_verify.start ();
    Galois::Runtime::PerThreadStorage<bool> result;
    for (unsigned i = 0; i < result.size (); ++i) {
      *result.getRemote(i) = true;
    }

    ParCounter numUnreachable;

    Galois::do_all_choice (Galois::Runtime::makeLocalRange(graph),
        [&graph, &numUnreachable, &result, &startNode] (GNode n) {
          const unsigned srcLevel = graph.getData (n, Galois::NONE);
          if (srcLevel >= BFS_LEVEL_INFINITY) { 
            numUnreachable += 1;
            // std::cerr << "BAD Level value >= INFINITY at node " << (*i) << std::endl;
            // result = false;
          }


          for (typename Graph::edge_iterator e = graph.edge_begin (n, Galois::NONE)
            , ende = graph.edge_end (n, Galois::NONE); e != ende; ++e) {
            
            GNode dst = graph.getEdgeDst (e);
            const unsigned dstLevel = graph.getData (dst, Galois::NONE);
            if (dstLevel > (srcLevel + 1) || (dst != startNode && dstLevel == 0)) {
              *result.getLocal () = false;
              std::cerr << "BAD Level value=" << dstLevel << " at neighbor " << dst << " of node " 
                << n << std::endl; 
            }
          }
        },
        "node-data-init",
        Galois::doall_chunk_size<CHUNK_SIZE> ());

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
    Galois::StatManager sm;

    Graph graph;

    std::cout << "Size of node data= " << sizeof (ND) << std::endl;

    initGraph (filename, graph);

    GNode startNode = getStartNode (graph, startId);

    std::cout << "Running <" << getVersion () << "> BFS" << std::endl;
    std::cout << "start node = " << startNode << std::endl;

    Galois::StatTimer timer;


    // for node based versions
    // Galois::preAlloc (Galois::getActiveThreads () + 8*graph.size ()/Galois::Runtime::MM::hugePageSize);
    // // for edge based versions
    unsigned p = Galois::getActiveThreads () + 8*graph.sizeEdges () / Galois::Runtime::MM::hugePageSize;
    std::printf ("going to pre-alloc %u pages, hugePageSize=%d,\n", p, (unsigned)Galois::Runtime::MM::hugePageSize);
    Galois::preAlloc (Galois::getActiveThreads () + 8*graph.sizeEdges ()/Galois::Runtime::MM::hugePageSize);
    Galois::reportPageAlloc("MeminfoPre");

    timer.start ();
    Galois::Runtime::beginSampling ();
    size_t numIter = runBFS (graph, startNode);
    Galois::Runtime::endSampling ();
    timer.stop ();
    Galois::reportPageAlloc("MeminfoPost");

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

    const unsigned srcLevel = graph.getData (src, (doLock ? Galois::CHECK_CONFLICT : Galois::NONE));

    // putting a loop to acquire locks. For now, edge_begin does not acquire locks on neighbors,
    // which it should
    if (doLock) {
      for (typename Graph::edge_iterator ni = graph.edge_begin (src, Galois::CHECK_CONFLICT)
          , eni = graph.edge_end (src, Galois::CHECK_CONFLICT); ni != eni; ++ni) {

        GNode dst = graph.getEdgeDst (ni);
        graph.getData (dst, Galois::CHECK_CONFLICT);
      }
    }


    for (typename Graph::edge_iterator ni = graph.edge_begin (src, Galois::NONE), eni = graph.edge_end (src, Galois::NONE);
        ni != eni; ++ni) {

      GNode dst = graph.getEdgeDst (ni);

      ND& dstData = graph.getData (dst, Galois::NONE); // iterator should already have acquired locks on neighbors
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


#endif // _BFS_H_

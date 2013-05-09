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

#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/Graph/LCGraph.h"
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

static const unsigned LEVEL_INFINITY = 1 << 20;


struct NodeData {
  unsigned data;

  NodeData (): data (0) {}
  explicit NodeData (unsigned _data): data (_data) {}

  inline unsigned& level () { return data; }
  inline const unsigned& level () const { return data; }

};

struct NodeDataCacheLine: public Galois::Runtime::LL::CacheLineStorage<unsigned> {
  typedef Galois::Runtime::LL::CacheLineStorage<unsigned> Super_ty;

  NodeDataCacheLine (): Super_ty (0) {}
  explicit NodeDataCacheLine (unsigned _data): Super_ty (_data) {}

  inline unsigned& level () { return Super_ty::data; }
  inline const unsigned& level () const { return Super_ty::data; }

};


template <typename ND>
class BFS {

public:
  typedef typename Galois::Graph::LC_CSR_Graph<ND, void> Graph;
  typedef typename Graph::GraphNode GNode;
  typedef ND NodeData_ty;

protected:
  virtual size_t runBFS (Graph& graph, GNode& startNode) = 0;

  virtual const std::string getVersion () const = 0;

  virtual void initGraph (const std::string& filename, Graph& graph) const {
    std::cout << "Reading graph from file: " << filename << std::endl;

    Galois::Graph::readGraph(graph, filename);

    unsigned numNodes = graph.size ();
    unsigned numEdges = 0;

    for (typename Graph::iterator i = graph.begin (), ei = graph.end ();
        i != ei; ++i) {

      graph.getData (*i, Galois::NONE) = ND (LEVEL_INFINITY);
      numEdges += graph.edge_end (*i, Galois::NONE) - graph.edge_begin (*i, Galois::NONE);
    }

    std::cout << "Graph read with nodes=" << numNodes << ", edges=" << numEdges << std::endl;
  }

  virtual GNode getStartNode (const Graph& graph, unsigned startId) const {
    assert (startId < graph.size ());

    unsigned id = 0;
    GNode startNode;
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
    bool result = true;

    for (typename Graph::iterator i = graph.begin (), ei = graph.end ();
        i != ei; ++i) {

      const unsigned srcLevel = graph.getData (*i, Galois::NONE).level ();
      if (srcLevel >= LEVEL_INFINITY) { 
        std::cerr << "BAD Level value >= INFINITY at node " << (*i) << std::endl;
        result = false;
      }


      for (typename Graph::edge_iterator ni = graph.edge_begin (*i, Galois::NONE), eni = graph.edge_end (*i, Galois::NONE);
          ni != eni; ++ni) {
        
        GNode dst = graph.getEdgeDst (ni);
        const unsigned dstLevel = graph.getData (dst, Galois::NONE).level ();
        if (dstLevel > (srcLevel + 1)) {
          result = false;
          std::cerr << "BAD Level value=" << dstLevel << " at neighbor " << dst << " of node " 
            << (*i) << std::endl; 
        }
      }
    }

    return result;
  }

public:

  virtual void run (int argc, char* argv[]) {
    LonestarStart (argc, argv, name, desc, url);

    Graph graph;

    std::cout << "Size of node data= " << sizeof (ND) << std::endl;

    initGraph (filename, graph);

    GNode startNode = getStartNode (graph, startId);

    std::cout << "Running <" << getVersion () << "> BFS" << std::endl;
    std::cout << "start node = " << startNode << std::endl;

    Galois::StatTimer timer ("BFS time: ");

    timer.start ();
    Galois::Runtime::beginSampling ();
    unsigned numIter = runBFS (graph, startNode);
    Galois::Runtime::endSampling ();
    timer.stop ();

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
      void (WL::*pushFn) (const typename WL::value_type&),
      const T& val) {
    (workList.*pushFn) (val);
  }


  //! @return number of adds
  template <bool doLock, typename WL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE static unsigned bfsOperator (
      Graph& graph, GNode& src, 
      WL& workList, void (WL::*pushFn) (const typename WL::value_type&)) {


    unsigned numAdds = 0;

    const unsigned srcLevel = graph.getData (src, (doLock ? Galois::CHECK_CONFLICT : Galois::NONE)).level ();

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
      if (dstData.level () == LEVEL_INFINITY) {
        dstData.level () = srcLevel + 1;

        // (workList.*pushFn) (dst);
        addToWL (workList, pushFn, dst);
        ++numAdds;
      }

    }

    return numAdds;
  }


};


#endif // _BFS_H_

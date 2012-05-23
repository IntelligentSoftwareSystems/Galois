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

// TODO: write a LCGraph without edge data


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
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/FileGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"


namespace cll = llvm::cl;

static const char* const name = "Breadth First Search";
static const char* const desc = "Level of a node is shortest number of edges from root to node";
static const char* const url = "bfs";

static cll::opt<unsigned int> startId("startnode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);


template <typename GraphTy, typename GNodeTy>

class BFS {
public:
  static const unsigned LEVEL_INFINITY = 1 << 20;

protected:
  virtual size_t runBFS (GraphTy& graph, GNodeTy& startNode) = 0;

  virtual const std::string getVersion () const = 0;

  virtual void initGraph (const std::string& filename, GraphTy& graph) const {
    std::cout << "Reading graph from file: " << filename << std::endl;


    graph.structureFromFile (filename);

    unsigned numNodes = graph.size ();
    unsigned numEdges = 0;

    for (typename GraphTy::iterator i = graph.begin (), ei = graph.end ();
        i != ei; ++i) {

      graph.getData (*i, Galois::NONE) = LEVEL_INFINITY;
      numEdges += graph.edge_end (*i, Galois::NONE) - graph.edge_begin (*i, Galois::NONE);
    }

    std::cout << "Graph read with nodes=" << numNodes << ", edges=" << numEdges << std::endl;
  }

  virtual GNodeTy getStartNode (const GraphTy& graph, unsigned startId) const {
    assert (startId < graph.size ());

    unsigned id = 0;
    GNodeTy startNode;
    for (typename GraphTy::iterator i = graph.begin (), ei = graph.end ();
        i != ei; ++i) {

      if (id == startId) {
        startNode = *i;
        break;
      }
      ++id;
    }

    return startNode;
  }

  virtual bool verify (GraphTy& graph, GNodeTy& startNode) const {
    bool result = true;

    for (typename GraphTy::iterator i = graph.begin (), ei = graph.end ();
        i != ei; ++i) {

      unsigned srcLevel = graph.getData (*i, Galois::NONE);
      if (srcLevel >= LEVEL_INFINITY) { 
        std::cerr << "BAD Level value >= INFINITY at node " << (*i) << std::endl;
        result = false;
      }


      for (typename GraphTy::edge_iterator ni = graph.edge_begin (*i, Galois::NONE), eni = graph.edge_end (*i, Galois::NONE);
          ni != eni; ++ni) {
        
        GNodeTy dst = graph.getEdgeDst (ni);
        unsigned dstLevel = graph.getData (dst, Galois::NONE);
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
    LonestarStart (argc, argv, std::cout, name, desc, url);

    GraphTy graph;

    initGraph (filename, graph);

    GNodeTy startNode = getStartNode (graph, startId);

    std::cout << "Running <" << getVersion () << "> BFS" << std::endl;
    std::cout << "start node = " << startNode << std::endl;

    Galois::StatTimer timer;

    timer.start ();
    unsigned numIter = runBFS (graph, startNode);
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

protected:
  //! @return number of adds
  template <bool doLock, typename WL>
  inline static unsigned bfsOperator (
      GraphTy& graph, GNodeTy& src, 
      WL& workList, void (WL::*pushFn) (const typename WL::value_type&)) {

    unsigned numAdds = 0;

    unsigned srcLevel = graph.getData (src, (doLock ? Galois::CHECK_CONFLICT : Galois::NONE));

    // putting a loop to acquire locks. For now, edge_begin does not acquire locks on neighbors,
    // which it should
    if (doLock) {
      for (typename GraphTy::edge_iterator ni = graph.edge_begin (src, Galois::CHECK_CONFLICT)
          , eni = graph.edge_end (src, Galois::CHECK_CONFLICT); ni != eni; ++ni) {

        GNodeTy dst = graph.getEdgeDst (ni);
        graph.getData (dst, Galois::CHECK_CONFLICT);
      }
    }


    for (typename GraphTy::edge_iterator ni = graph.edge_begin (src, Galois::NONE), eni = graph.edge_end (src, Galois::NONE);
        ni != eni; ++ni) {

      GNodeTy dst = graph.getEdgeDst (ni);

      unsigned dstLevel = graph.getData (dst, Galois::NONE); // iterator should already have acquired locks on neighbors
      if (dstLevel == LEVEL_INFINITY) {
        graph.getData (dst, Galois::NONE) = srcLevel + 1;

        (workList.*pushFn) (dst);
        ++numAdds;
      }

    }

    return numAdds;
  }
};



#endif // _BFS_H_

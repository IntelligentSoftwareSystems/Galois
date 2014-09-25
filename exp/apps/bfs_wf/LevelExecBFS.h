/** BFS with level-by-level executor -*- C++ -*-
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
 * BFS with level-by-level executor.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef LEVEL_EXEC_BFS_H
#define LEVEL_EXEC_BFS_H

#include <vector>
#include <functional>

#include "Galois/Runtime/LevelExecutor.h"
#include "Galois/WorkList/WorkList.h"

#include "bfs.h"

using Level_ty = unsigned;
class LevelExecBFS: public BFS<Level_ty> {

  struct Update {
    GNode node;
    Level_ty level;

    Update (const GNode& node, const Level_ty& level)
      : node (node), level (level) 
    {}

    friend std::ostream& operator << (std::ostream& out, const Update& up) {
      out << "(node:" << up.node << ",level:" << up.level << ")";
      return out;
    }
  };
  
  struct GetLevel {

    Level_ty operator () (const Update& up) const {
      return up.level;
    }
  };

  struct VisitNhood {

    Graph& graph;

    explicit VisitNhood (Graph& graph): graph (graph) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) const {

      // just like DES, we only lock the node being updated, but not its
      // outgoing neighbors
      // graph.getData (up.node, Galois::MethodFlag::WRITE_INTENT);
    }
  };

  struct OpFunc {

    static const unsigned CHUNK_SIZE = 128;

    typedef int tt_does_not_need_aborts;

    Graph& graph;
    ParCounter& numIter;

    OpFunc (Graph& graph, ParCounter& numIter): graph (graph), numIter (numIter) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) const {

      if (graph.getData (up.node, Galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {

        graph.getData (up.node, Galois::MethodFlag::UNPROTECTED) = up.level;


        for (auto ni = graph.edge_begin (up.node, Galois::MethodFlag::UNPROTECTED)
            , eni = graph.edge_end (up.node, Galois::MethodFlag::UNPROTECTED); ni != eni; ++ni) {

          GNode dst = graph.getEdgeDst (ni);

          if (graph.getData (dst, Galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {
            ctx.push (Update (dst, up.level + 1));
          }
        }

      }

      numIter += 1;
    }

  };

public:

  virtual const std::string getVersion () const { return "using Level-by-Level executor"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    ParCounter numIter;


    // update request for root
    Update first (startNode, 0);

    std::vector<Update> wl;
    wl.push_back (first);

    typedef Galois::WorkList::dChunkedFIFO<OpFunc::CHUNK_SIZE, Update> C;
    typedef Galois::WorkList::OrderedByIntegerMetric<GetLevel, C>::with_barrier<true>::type WL_ty;

    Galois::Runtime::for_each_ordered_level (
        Galois::Runtime::makeStandardRange (wl.begin (), wl.end ()), 
        GetLevel (), 
        std::less<Level_ty> (),
        VisitNhood (graph),
        OpFunc (graph, numIter));

    // Galois::for_each (first,
        // OpFunc (graph, numIter),
        // Galois::loopname ("bfs-level-exec"),
        // Galois::wl<WL_ty> ());


    std::cout << "number of iterations: " << numIter.reduce () << std::endl;


    return numIter.reduce ();
  }


};


#endif //  LEVEL_EXEC_BFS_H

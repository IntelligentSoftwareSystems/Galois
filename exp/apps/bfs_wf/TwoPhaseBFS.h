/** Two Phase BFS -*- C++ -*-
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
 * Two Phase BFS.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef SPEC_ORDERED_BFS_H
#define SPEC_ORDERED_BFS_H

#include <vector>
#include <functional>

#include "Galois/Runtime/KDGtwoPhase.h"

#include "bfs.h"

using Level_ty = unsigned;
class TwoPhaseBFS: public BFS<Level_ty> {

  struct Update {
    GNode src;
    GNode dst;
    Level_ty level;

    Update (const GNode& src, const GNode& dst, const Level_ty& level)
      : src (src), dst (dst), level (level) 
    {}

    friend std::ostream& operator << (std::ostream& out, const Update& up) {
      out << "(dst:" << up.dst << ",src:" << up.src <<  ",level:" << up.level << ")";
      return out;
    }
  };
  
  struct Comparator {
    bool operator () (const Update& left, const Update& right) const {
      return left.level < right.level;
      // int d = left.level - right.level;
// 
      // // if (d == 0) {
        // // // FIXME: assuming nodes are actually integer like
        // // d = left.dst - right.dst;
// // 
        // // if (d == 0) {
          // // d = left.src - right.src;
        // // }
      // // }
// 
      // return (d < 0);
    }
  };

  struct VisitNhood {

    Graph& graph;

    explicit VisitNhood (Graph& graph): graph (graph) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) {

      // just like DES, we only lock the node being updated, but not its
      // outgoing neighbors
      graph.getData (up.dst, Galois::MethodFlag::WRITE);
    }
  };

  struct OpFunc {

    static const unsigned CHUNK_SIZE = 1024;

    Graph& graph;
    ParCounter& numIter;

    OpFunc (Graph& graph, ParCounter& numIter): graph (graph), numIter (numIter) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) {

      if (graph.getData (up.dst, Galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {

        graph.getData (up.dst, Galois::MethodFlag::UNPROTECTED) = up.level;


        for (typename Graph::edge_iterator ni = graph.edge_begin (up.dst, Galois::MethodFlag::UNPROTECTED)
            , eni = graph.edge_end (up.dst, Galois::MethodFlag::UNPROTECTED); ni != eni; ++ni) {

          GNode src = up.dst;
          GNode dst = graph.getEdgeDst (ni);

          if (graph.getData (dst, Galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {
            ctx.push (Update (src, dst, up.level + 1));
          }
        }

      }

      numIter += 1;

    }

  };

public:

  virtual const std::string getVersion () const { return "Two Phase ordered"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    ParCounter numIter;


    // update request for root
    Update first (startNode, startNode, 0);

    std::vector<Update> wl;
    wl.push_back (first);

    Galois::Runtime::for_each_ordered_2p_win (
        Galois::Runtime::makeStandardRange(wl.begin (), wl.end ()),
        Comparator (), 
        VisitNhood (graph),
        OpFunc (graph, numIter));


    std::cout << "number of iterations: " << numIter.reduce () << std::endl;


    return numIter.reduce ();
  }


};


#endif // SPEC_ORDERED_BFS_H

/** Wavefront BFS -*- C++ -*-
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
 * Wavefront BFS.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef WAVEFRONT_BULK_SYNC_H
#define WAVEFRONT_BULK_SYNC_H

#include "bfs.h"

#include "Galois/PerThreadContainer.h"

#include "Galois/Runtime/ParallelWorkInline_Exp.h"

typedef uint32_t ND_ty;

class WavefrontBulkSync: public BFS<ND_ty> {

  using Base = BFS<ND_ty>;

protected:
  virtual const std::string getVersion () const { return "Wavefront using Bulk Synchronous Executor"; }

  struct Update {
    GNode src;
    ND_ty srcLevel;

    Update (const GNode& src, const ND_ty& srcLevel): src (src), srcLevel (srcLevel) {}

    friend std::ostream& operator << (std::ostream& out, const Update& up) {
      return (out << "(" << up.src << "," << up.srcLevel << ")");
    }
  };


  struct PftchFunc {
    typedef int tt_does_not_need_aborts;
    typedef char tt_does_not_need_push;

    Graph& graph;

    explicit PftchFunc (Graph& graph): graph (graph) {}

    template <typename K>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Update& up, K pftch_kind) {
      graph.prefetchOutNeighbors(up.src, pftch_kind);
    }
  };

  struct OpFunc {
    typedef int tt_does_not_need_aborts;
    // typedef char tt_does_not_need_push;
    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;


    Graph& graph;

    explicit OpFunc (Graph& graph) : graph (graph) {}

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Update& up, C& wl) {

      graph.mapOutNeighbors (up.src, 
          [&up,this, &wl] (GNode dst) {
            ND_ty dstLevel = graph.getData (dst, galois::MethodFlag::UNPROTECTED);

            if (dstLevel == BFS_LEVEL_INFINITY) {
              graph.getData (dst, galois::MethodFlag::UNPROTECTED) = up.srcLevel + 1;
              wl.push (Update (dst, up.srcLevel + 1));
              // wl.push_back (dst);
            }
          },
          galois::MethodFlag::UNPROTECTED);

    }
  };


  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    // typedef galois::WorkList::BulkSynchronousInline<> WL_ty;

    ParCounter numAdds;
    graph.getData (startNode, galois::MethodFlag::UNPROTECTED) = 0;

    Update init[] = { Update (startNode, 0) };

    // galois::for_each (startNode, OpFunc (graph), galois::wl<WL_ty> ());
    galois::Runtime::for_each_bs (galois::Runtime::makeStandardRange (&init[0], &init[1]), OpFunc (graph), PftchFunc (graph));

    return numAdds.reduce ();
  }

};



#endif // WAVEFRONT_BULK_SYNC_H

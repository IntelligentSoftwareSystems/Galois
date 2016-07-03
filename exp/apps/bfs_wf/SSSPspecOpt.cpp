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

#include <vector>
#include <functional>

#include "Galois/Runtime/KDGspecLocalMin.h"

#include "bfs.h"

class SSSPspecOpt: public SSSP {
  


  // relies on round based execution of IKDG executor
  struct OpFuncLocalMin: public OpFunc {

    OpFuncLocalMin (Graph& graph, ParCounter& numIter): OpFunc (graph, numIter) {}

    template <typename C>
    void operator () (const Update& up, C& ctx) {
      auto& ndata = graph.getData (up.node, Galois::MethodFlag::UNPROTECTED);
      if (ndata > up.level) {

        ndata = up.level;


        for (auto ni = graph.edge_begin (up.node, Galois::MethodFlag::UNPROTECTED)
            , eni = graph.edge_end (up.node, Galois::MethodFlag::UNPROTECTED); ni != eni; ++ni) {

          GNode dst = graph.getEdgeDst (ni);
          auto w = graph.getEdgeData (ni);

          if (graph.getData (dst, Galois::MethodFlag::UNPROTECTED) > (up.level + w)) {
            ctx.push (Update (dst, up.level + w));
          }
        }

      }

      numIter += 1;
    }

  };



public:

  virtual const std::string getVersion () const { return "Speculative SSSP with optimizations"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    ParCounter numIter;

    // update request for root
    Update first (startNode, 0);

    std::vector<Update> wl;
    wl.push_back (first);

    Galois::Runtime::for_each_ordered_kdg_spec_local_min (
        Galois::Runtime::makeStandardRange(wl.begin (), wl.end ()),
        Comparator (), 
        VisitNhood (graph),
        OpFuncLocalMin (graph, numIter),
        std::make_tuple (
          Galois::loopname ("bfs_spec_local_min")));


    std::cout << "number of iterations: " << numIter.reduce () << std::endl;


    return numIter.reduce ();
  }


};

int main (int argc, char* argv[]) {
  SSSPspecOpt wf;
  wf.run (argc, argv);
  return 0;
}

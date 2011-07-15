/** AVI unordered algorithm with no abstract locks -*- C++ -*-
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
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef AVI_UNORDERED_NO_LOCK_H_
#define AVI_UNORDERED_NO_LOCK_H_


#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"


#include "Galois/Graphs/Serialize.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/util/Atomic.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <set>

#include <cassert>

#include "AuxDefs.h"
#include "AVI.h"
#include "Element.h"

#include "AVIabstractMain.h"
#include "AVIunordered.h"

/**
 * AVI unordered algorithm that uses atomic integers 
 * and no abstract locks
 */
class AVIunorderedNoLock: public AVIunordered {

protected:

  virtual const std::string getVersion () const {
    return "Parallel Unordered, no abstract locks";
  }
  
  /**
   * Functor for loop body
   */
  struct process {

    Graph& graph;
    std::vector<AtomicInteger>& inDegVec;
    MeshInit& meshInit;
    GlobalVec& g;
    GaloisRuntime::PerCPU<LocalVec>& perIterLocalVec;
    GaloisRuntime::PerCPU< std::vector<GNode> >& perIterAddList;
    const AVIComparator& aviCmp;
    bool createSyncFiles;
    IterCounter& iter;

    process (
        Graph& graph,
        std::vector<AtomicInteger>& inDegVec,
        MeshInit& meshInit,
        GlobalVec& g,
        GaloisRuntime::PerCPU<LocalVec>& perIterLocalVec,
        GaloisRuntime::PerCPU< std::vector<GNode> >& perIterAddList,
        const AVIComparator& aviCmp,
        bool createSyncFiles,
        IterCounter& iter):

        graph (graph),
        inDegVec (inDegVec),
        meshInit (meshInit),
        g (g),
        perIterLocalVec (perIterLocalVec),
        perIterAddList (perIterAddList),
        aviCmp (aviCmp),
        createSyncFiles (createSyncFiles),
        iter (iter) {}
    

    process (const process& that):
        graph (that.graph),
        inDegVec (that.inDegVec),
        meshInit (that.meshInit),
        g (that.g),
        perIterLocalVec (that.perIterLocalVec),
        perIterAddList (that.perIterAddList),
        aviCmp (that.aviCmp),
        createSyncFiles (that.createSyncFiles),
        iter (that.iter) {}


    /**
     * Loop body
     *
     * The key condition is that a node and its neighbor cannot be active at the same time,
     * and it must be impossible for two of them to be processed in parallel
     * Therefore a node can add its newly active neighbors to the workset as the last step only when
     * it has finished performing all other updates
     * For the same reason, active node src must update its own in degree before updating the
     * indegree of any of the neighbors. Imagine the alternative, where active node updates its in
     * degree and that of it's neighbor in the same loop. For example A is current active node and
     * has a neighbor B. A > B, therefore A increments its own in degree and decrements that of B to
     * 1. Another active node C is neighbor of B but not of A, and C decreases in degree of B to 0
     * and adds B to the workset while A is not finished yet. This violates our key condition
     * mentioned above
     *
     * @param src is active elemtn
     * @param lwl is the worklist handle
     */
    template <typename ContextTy> 
      void operator () (const GNode& src, ContextTy& lwl) {
        AVI* srcAVI = graph.getData (src, Galois::NONE);

        int inDeg = (int)inDegVec[srcAVI->getGlobalIndex ()];
        // assert  inDeg == 0 : String.format ("active node %s with inDeg = %d\n", srcAVI, inDeg);

//        // TODO: DEBUG
//        std::cout << "Processing element: " << srcAVI->toString() << std::endl;

        assert (inDeg == 0);

        LocalVec& l = perIterLocalVec.get();

        AVIabstractMain::simulate(srcAVI, meshInit, g, l, createSyncFiles);


        // update the inEdges count and determine
        // which neighbor is at local minimum and needs to be added to the worklist
        
        int addAmt = 0;

        for (Graph::neighbor_iterator j = graph.neighbor_begin (src, Galois::NONE), ej = graph.neighbor_end (src, Galois::NONE);
            j != ej; ++j) {

          AVI* dstAVI = graph.getData (*j, Galois::NONE);

          if (aviCmp.compare (srcAVI, dstAVI) > 0) {
            ++addAmt;
          }

        }


        // may be the active node is still at the local minimum
        // and no updates to neighbors are necessary
        if (addAmt == 0) {
          if (srcAVI->getNextTimeStamp () < meshInit.getSimEndTime ()) {
            lwl.push (src);
          }
        }
        else {
          inDegVec[srcAVI->getGlobalIndex ()] += addAmt;

          std::vector<GNode> toAdd = perIterAddList.get ();
          toAdd.clear ();

          for (Graph::neighbor_iterator j = graph.neighbor_begin (src, Galois::NONE), ej = graph.neighbor_end(src, Galois::NONE);
              j != ej; ++j) {

            const GNode& dst = *j;
            AVI* dstAVI = graph.getData (dst, Galois::NONE);

            if (aviCmp.compare (srcAVI, dstAVI) > 0) {
              int din = --inDegVec[dstAVI->getGlobalIndex ()];

              assert (din >= 0);

              if (din == 0) {
                if (dstAVI->getNextTimeStamp () < meshInit.getSimEndTime ()) {
                  toAdd.push_back (dst);

//                  // TODO: DEBUG
//                  std::cout << "Adding: " << dstAVI->toString () << std::endl;
                }
              }
            }

          } // end for


          for (std::vector<GNode>::const_iterator i = toAdd.begin (), ei = toAdd.end (); i != ei; ++i) {
            const GNode& gn = (*i);
            lwl.push (gn);
          }

        } // end else


        // for debugging, remove later
        ++(iter.get ());


      }
  };

public:

  /**
   * For the in degree vector, we use a vector of atomic integers
   * This along with other changes in the loop body allow us to 
   * no use abstract locks. @see process
   */
  virtual  void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    /////////////////////////////////////////////////////////////////
    // populate an initial  worklist
    /////////////////////////////////////////////////////////////////
    std::vector<AtomicInteger> inDegVec(meshInit.getNumElements (), AtomicInteger (0));
    std::vector<GNode> initWl;

    AVIComparator aviCmp;

    for (Graph::active_iterator i = graph.active_begin (), e = graph.active_end (); i != e; ++i) {
      const GNode& src = *i;
      AVI* srcAVI = graph.getData (src, Galois::NONE);

      // calculate the in degree of src by comparing it against its neighbors
      for (Graph::neighbor_iterator n = graph.neighbor_begin (src, Galois::NONE), 
          en= graph.neighbor_end (src, Galois::NONE); n != en; ++n) {
        
        AVI* dstAVI = graph.getData (*n, Galois::NONE);
        if (aviCmp.compare (srcAVI, dstAVI) > 0) {
          ++inDegVec[srcAVI->getGlobalIndex ()];
        }
      }

      // if src is less than all its neighbors then add to initWl
      if ((int)inDegVec[srcAVI->getGlobalIndex ()] == 0) {
        initWl.push_back (src);
      }
    }


 
    printf ("Initial worklist contains %zd elements\n", initWl.size ());

//    // TODO: DEBUG
//    std::cout << "Initial Worklist = " << std::endl;
//    for (size_t i = 0; i < initWl.size (); ++i) {
//      std::cout << graph.getData (initWl[i], Galois::NONE)->toString () << ", ";
//    }
//    std::cout << std::endl;

    /////////////////////////////////////////////////////////////////
    // perform the simulation
    /////////////////////////////////////////////////////////////////

    // temporary matrices
    size_t nrows = meshInit.getSpatialDim ();
    size_t ncols = meshInit.getNodesPerElem();

    LocalVec l(nrows, ncols);

    GaloisRuntime::PerCPU<LocalVec> perIterLocalVec (l);


    GaloisRuntime::PerCPU< std::vector<GNode> > perIterAddList;




    IterCounter iter(0);


    process p( graph, inDegVec, meshInit, g, perIterLocalVec, perIterAddList, aviCmp, createSyncFiles, iter);


    Galois::for_each< AVIWorkList >(initWl.begin (), initWl.end (), p);


    printf ("iterations = %d\n", iter.get ());

  }



};

#endif


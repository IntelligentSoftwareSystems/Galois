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

class AVIunorderedNoLock: public AVIunordered {

protected:

  virtual const std::string getVersion () const {
    return "Parallel Unordered, no abstract locks";
  }
  
  struct process {

    Graph& graph;
    std::vector<AtomicInteger>& inDegVec;
    MeshInit& meshInit;
    GlobalVec& g;
    GaloisRuntime::PerCPU<LocalVec>& perIterLocalVec;
    GaloisRuntime::PerCPU< std::vector<GNode> >& perIterAddList;
    const AVIComparator& aviCmp;
    bool createSyncFiles;
    AtomicInteger& iter;

    process (
        Graph& graph,
        std::vector<AtomicInteger>& inDegVec,
        MeshInit& meshInit,
        GlobalVec& g,
        GaloisRuntime::PerCPU<LocalVec>& perIterLocalVec,
        GaloisRuntime::PerCPU< std::vector<GNode> >& perIterAddList,
        const AVIComparator& aviCmp,
        bool createSyncFiles,
        AtomicInteger& iter):

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


    template <typename ContextTy> 
      void operator () (const GNode& src, ContextTy& lwl) {
        AVI* srcAVI = graph.getData (src, Galois::Graph::NONE);

        int inDeg = inDegVec[srcAVI->getGlobalIndex ()].get ();
        // assert  inDeg == 0 : String.format ("active node %s with inDeg = %d\n", srcAVI, inDeg);

//        // TODO: DEBUG
//        std::cout << "Processing element: " << srcAVI->toString() << std::endl;

        assert (inDeg == 0);

        LocalVec& l = perIterLocalVec.get();

        AVIabstractMain::simulate(srcAVI, meshInit, g, l, createSyncFiles);


        // update the inEdges count and determine
        // which neighbor is at local minimum and needs to be added to the worklist
        
        int addAmt = 0;

        for (Graph::neighbor_iterator j = graph.neighbor_begin (src, Galois::Graph::NONE), ej = graph.neighbor_end (src, Galois::Graph::NONE);
            j != ej; ++j) {

          AVI* dstAVI = graph.getData (*j, Galois::Graph::NONE);

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
          inDegVec[srcAVI->getGlobalIndex ()].addAndGet (addAmt);

          std::vector<GNode> toAdd = perIterAddList.get ();
          toAdd.clear ();

          for (Graph::neighbor_iterator j = graph.neighbor_begin (src, Galois::Graph::NONE), ej = graph.neighbor_end(src, Galois::Graph::NONE);
              j != ej; ++j) {

            GNode dst = *j;
            AVI* dstAVI = graph.getData (dst, Galois::Graph::NONE);

            if (aviCmp.compare (srcAVI, dstAVI) > 0) {
              int din = inDegVec[dstAVI->getGlobalIndex ()].decrementAndGet ();

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


          for (std::vector<GNode>::const_iterator i = toAdd.begin (), e = toAdd.end (); i != e; ++i) {
            const GNode& gn = (*i);
            lwl.push (gn);
          }

        } // end else


        // for debugging, remove later
        iter.incrementAndGet ();


      }
  };

public:

  virtual  void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    /////////////////////////////////////////////////////////////////
    // populate an initial  worklist
    /////////////////////////////////////////////////////////////////
    std::vector<AtomicInteger> inDegVec(meshInit.getNumElements (), AtomicInteger (0));
    std::vector<GNode> initWl;

    AVIComparator aviCmp;

    for (Graph::active_iterator i = graph.active_begin (), e = graph.active_end (); i != e; ++i) {
      const GNode& src = *i;
      AVI* srcAVI = graph.getData (src, Galois::Graph::NONE);

      // calculate the in degree of src by comparing it against its neighbors
      for (Graph::neighbor_iterator n = graph.neighbor_begin (src, Galois::Graph::NONE), 
          en= graph.neighbor_end (src, Galois::Graph::NONE); n != en; ++n) {
        
        AVI* dstAVI = graph.getData (*n, Galois::Graph::NONE);
        if (aviCmp.compare (srcAVI, dstAVI) > 0) {
          inDegVec[srcAVI->getGlobalIndex ()].incrementAndGet ();
        }
      }

      // if src is less than all its neighbors then add to initWl
      if (inDegVec[srcAVI->getGlobalIndex ()].get () == 0) {
        initWl.push_back (src);
      }
    }


 
    printf ("Initial worklist contains %zd elements\n", initWl.size ());

//    // TODO: DEBUG
//    std::cout << "Initial Worklist = " << std::endl;
//    for (size_t i = 0; i < initWl.size (); ++i) {
//      std::cout << graph.getData (initWl[i], Galois::Graph::NONE)->toString () << ", ";
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




    AtomicInteger iter(0);


    process p( graph, inDegVec, meshInit, g, perIterLocalVec, perIterAddList, aviCmp, createSyncFiles, iter);


    Galois::for_each< GaloisRuntime::WorkList::FIFO<GNode> >(initWl.begin (), initWl.end (), p);


    printf ("iterations = %d\n", iter.get ());

  }



};

#endif

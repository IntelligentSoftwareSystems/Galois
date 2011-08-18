/** AVI unordered algorithm with abstract locks -*- C++ -*-
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

#ifndef AVI_UNORDERED_H_
#define AVI_UNORDERED_H_


#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"


#include "Galois/Graphs/Serialize.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Atomic.h"
#include "Galois/Accumulator.h"

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

/**
 *
 * Unordered AVI algorithm uses two key data structures
 *
 * 1) Element Adjacency Graph
 * 2) in degree vector
 *
 * This graph has a node for each mesh element and 
 * keeps track of node-adjacency between AVI elements. Two elements
 * are adjacent if they share a node in the mesh between them.
 * We create a graph by connecting adjacent elements with an edge.
 * Conceptually the edge is directed from the avi element with smaller
 * time stamp to the greater one. But in implementation this direction information
 * is not kept in the graph but in an array 'inDegVec', which has an entry corresponding
 * to each AVI element. 
 * An avi element with 0 in edges has the minimum time stamp among its neighbors
 * and is therefore eligible for an update
 * It is assumed that AVI elements have unique integer id's 0..numElements-1, and 
 * the id is used to index into inDegVec
 *
 */

class AVIunordered: public AVIabstractMain {

protected:
  typedef Galois::GAccumulator<int> IterCounter;

  static const bool DEBUG = false;

  static const int CHUNK_SIZE = 32;

// TODO: use LCGraph 
  typedef Galois::Graph::FirstGraph<AVI*, void, false> Graph;
  typedef Graph::GraphNode GNode;

  typedef GaloisRuntime::WorkList::ChunkedFIFO<CHUNK_SIZE, GNode> AVIWorkList;


  Graph graph;

  virtual const std::string getVersion () const {
    return "Parallel Unordered";
  }

  /**
   * Generate element adjacency graph, where nodes are elements
   * in the mesh, and there is an edge between the nodes if their
   * corresponding elements share a vertex in the mesh
   *
   * @param meshInit
   * @param g
   */
  void genElemAdjGraph (const MeshInit& meshInit, const GlobalVec& g) {
    std::vector<GNode> aviAdjNodes;

    const std::vector<AVI*>& aviList = meshInit.getAVIVec ();

    for (std::vector<AVI*>::const_iterator i = aviList.begin (), e = aviList.end (); i != e; ++i) {
      AVI* avi = *i;
      GNode gn = graph.createNode (avi);
      graph.addNode (gn);

      aviAdjNodes.push_back (gn);
    }



    // map where
    // key is node id
    // value is a list of avi elements that share this node
    std::vector< std::vector<GNode> > nodeSharers(meshInit.getNumNodes ());

    // for (int i = 0; i < nodeSharers.size (); ++i) {
      // nodeSharers[i] = new ArrayList<GNode<AVIAdjNode>> ();
    // }

    for (std::vector<GNode>::const_iterator i = aviAdjNodes.begin (), ei = aviAdjNodes.end (); i != ei; ++i) {
      GNode aviAdjN = *i;
      AVI* avi = graph.getData (aviAdjN, Galois::NONE);
      const std::vector<GlobalNodalIndex>& conn = avi->getGeometry ().getConnectivity ();

      for (std::vector<GlobalNodalIndex>::const_iterator j = conn.begin (), ej = conn.end (); j != ej; ++j) {
        GlobalNodalIndex n = *j;
        nodeSharers[n].push_back (aviAdjN);
      }
      
    }

    int numEdges = 0;

    for (std::vector< std::vector<GNode> >::const_iterator it = nodeSharers.begin (), ei = nodeSharers.end ();
        it != ei; ++it) {

      const std::vector<GNode>& adjElms = *it;

      // adjElms is the list of elements who share the node with id == current index pos in the array
      // and therefore form a clique among themselves
      for (size_t i = 0; i < adjElms.size (); ++i) {
        // populate the upper triangle of the adj matrix
        for (size_t j = i + 1; j < adjElms.size (); ++j) {
          if (!adjElms[i].hasNeighbor (adjElms[j])) {
            ++numEdges;
          }
          graph.addEdge (adjElms[i], adjElms[j]);
        }
      }

    }

    printf ("Graph created with %d nodes and %d edges\n", graph.size (), numEdges);

  }
  
  virtual void initRemaining (const MeshInit& meshInit, const GlobalVec& g) {
    genElemAdjGraph (meshInit, g);
  }

  //! Functor for loop body
  struct process {

    Graph& graph;
    std::vector<int>& inDegVec;
    MeshInit& meshInit;
    GlobalVec& g;
    GaloisRuntime::PerCPU<LocalVec>& perIterLocalVec;
    const AVIComparator& aviCmp;
    bool createSyncFiles;
    IterCounter& iter;

    process (
        Graph& graph,
        std::vector<int>& inDegVec,
        MeshInit& meshInit,
        GlobalVec& g,
        GaloisRuntime::PerCPU<LocalVec>& perIterLocalVec,
        const AVIComparator& aviCmp,
        bool createSyncFiles,
        IterCounter& iter):

        graph (graph),
        inDegVec (inDegVec),
        meshInit (meshInit),
        g (g),
        perIterLocalVec (perIterLocalVec),
        aviCmp (aviCmp),
        createSyncFiles (createSyncFiles),
        iter (iter) {}
    

    /**
     * Loop body
     *
     * The loop body uses one-shot optimization, where we grab abstract locks on the node
     * and its neighbors before performing the udpates. This removes the need for saving 
     * and performing undo operations.
     *
     *
     * @param src is active elemtn
     * @param lwl is the worklist handle
     */
    template <typename ContextTy> 
      void operator () (const GNode& src, ContextTy& lwl) {
        // one-shot optimization: acquire abstract locks on active node and
        // neighbors (all its neighbors, in this case) before performing any modifications

        AVI* srcAVI = graph.getData (src, Galois::CHECK_CONFLICT);

        for (Graph::neighbor_iterator j = graph.neighbor_begin (src, Galois::CHECK_CONFLICT)
            , ej = graph.neighbor_end (src, Galois::CHECK_CONFLICT); j != ej; ++j) {
        }


        // past the fail-safe point now


        int inDeg = inDegVec[srcAVI->getGlobalIndex ()];
        // assert  inDeg == 0 : String.format ("active node %s with inDeg = %d\n", srcAVI, inDeg);

        //        // TODO: DEBUG
        //        std::cout << "Processing element: " << srcAVI->toString() << std::endl;

        assert (inDeg == 0);


        LocalVec& l = perIterLocalVec.get();

        AVIabstractMain::simulate(srcAVI, meshInit, g, l, createSyncFiles);


        // update the inEdges count and determine
        // which neighbor is at local minimum and needs to be added to the worklist

        for (Graph::neighbor_iterator j = graph.neighbor_begin (src, Galois::NONE), ej = graph.neighbor_end (src, Galois::NONE);
            j != ej; ++j) {

          const GNode& dst = *j;
          AVI* dstAVI = graph.getData (*j, Galois::NONE);

          if (aviCmp.compare (srcAVI, dstAVI) > 0) {
            // if srcAVI has a higher time stamp that dstAVI

            ++inDegVec[srcAVI->getGlobalIndex ()];

            int din = (--inDegVec[dstAVI->getGlobalIndex ()] );

            if (din == 0) {
              // dstAVI has become minimum among its neighbors
              if (dstAVI->getNextTimeStamp () < meshInit.getSimEndTime ()) {
                lwl.push (dst);
              }
            }
          }

        } // end for

        if (inDegVec[srcAVI->getGlobalIndex ()] == 0) {
          // srcAVI is still the minimum among its neighbors
          if (srcAVI->getNextTimeStamp () < meshInit.getSimEndTime ()) {
            lwl.push (src);
          }
        }


        ++ (iter.get ());

        // if (iter.get () == 5000) {
           // meshInit.plotMesh ();
           // meshInit.plotMeshCenters ();
        // }

      }
  };

  struct Indexer : std::binary_function<GNode, unsigned, unsigned> {
    unsigned operator()(const GNode& val) const {
      double ts = val.getData(Galois::NONE)->getNextTimeStamp();
      unsigned r = static_cast<unsigned>(ts * 1000000);
      return r;
    }
  };
public:

  virtual  void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    /////////////////////////////////////////////////////////////////
    // populate an initial  worklist
    /////////////////////////////////////////////////////////////////
    std::vector<int> inDegVec(meshInit.getNumElements (), 0);

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
      if (inDegVec[srcAVI->getGlobalIndex ()] == 0) {
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

    // uncomment to plot the mesh
    meshInit.plotMesh ();
    // meshInit.plotMeshCenters ();

    // temporary matrices
    size_t nrows = meshInit.getSpatialDim ();
    size_t ncols = meshInit.getNodesPerElem();

    LocalVec l(nrows, ncols);

    GaloisRuntime::PerCPU<LocalVec> perIterLocalVec (l);

    IterCounter iter(0);

    process p( graph, inDegVec, meshInit, g, perIterLocalVec, aviCmp, createSyncFiles, iter);

    using namespace GaloisRuntime::WorkList;

    typedef ChunkedLIFO<64> LChunk;
    typedef ChunkedFIFO<32> Chunk;
    typedef dChunkedFIFO<16> dChunk;
    typedef OrderedByIntegerMetric<Indexer, dChunk> OBIM;

    if (wltype == "obim") {
      std::cout << wltype << "\n";
      Galois::for_each<OBIM>(initWl.begin (), initWl.end (), p);
    } else if (wltype == "lifo") {
      std::cout << wltype << "\n";
      Galois::for_each<LIFO<> >(initWl.begin (), initWl.end (), p);
    } else if (wltype == "clifo") {
      std::cout << wltype << "\n";
      Galois::for_each<LChunk>(initWl.begin (), initWl.end (), p);
    } else {
      Galois::for_each<Chunk>(initWl.begin (), initWl.end (), p);
    }

    printf ("iterations = %d\n", iter.get ());

  }



};

#endif

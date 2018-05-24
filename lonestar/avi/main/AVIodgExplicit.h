#ifndef AVI_ODG_EXPLICIT_H
#define AVI_ODG_EXPLICIT_H

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <set>
#include <utility>

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

//#define USE_LC_GRAPH

class AVIodgExplicit: public AVIabstractMain {

protected:
  static const bool DEBUG = false;

#ifdef USE_LC_GRAPH
  typedef galois::graphs::LC_CSR_Graph<AVI*, void> Graph;
  typedef Graph::GraphNode GNode;
#else
  typedef galois::graphs::FirstGraph<AVI*, void, false> Graph;
  typedef Graph::GraphNode GNode;
#endif


  Graph graph;

  virtual const std::string getVersion () const {
    return "ODG explicit, abstract locks on ODG nodes";
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

#ifdef USE_LC_GRAPH
    typedef galois::graphs::FirstGraph<AVI*, void, false> MGraph;
    typedef MGraph::GraphNode MNode;

    MGraph mgraph;
#else 
    Graph& mgraph = graph;
    typedef GNode MNode;
#endif


    std::vector<MNode> aviAdjNodes;

    const std::vector<AVI*>& aviList = meshInit.getAVIVec ();

    for (std::vector<AVI*>::const_iterator i = aviList.begin (), e = aviList.end (); i != e; ++i) {
      AVI* avi = *i;
      MNode gn = mgraph.createNode (avi);
      mgraph.addNode (gn);

      aviAdjNodes.push_back (gn);
    }



    // map where
    // key is node id
    // value is a list of avi elements that share this node
    std::vector< std::vector<MNode> > nodeSharers(meshInit.getNumNodes ());

    // for (int i = 0; i < nodeSharers.size (); ++i) {
      // nodeSharers[i] = new ArrayList<GNode<AVIAdjNode>> ();
    // }

    for (std::vector<MNode>::const_iterator i = aviAdjNodes.begin (), ei = aviAdjNodes.end (); i != ei; ++i) {
      MNode aviAdjN = *i;
      AVI* avi = mgraph.getData (aviAdjN, galois::MethodFlag::UNPROTECTED);
      const VecSize_t& conn = avi->getGeometry ().getConnectivity ();

      for (VecSize_t::const_iterator j = conn.begin (), ej = conn.end (); j != ej; ++j) {
        GlobalNodalIndex n = *j;
        nodeSharers[n].push_back (aviAdjN);
      }
      
    }

    int numEdges = 0;

    for (std::vector< std::vector<MNode> >::const_iterator it = nodeSharers.begin (), ei = nodeSharers.end ();
        it != ei; ++it) {

      const std::vector<MNode>& adjElms = *it;

      // adjElms is the list of elements who share the node with id == current index pos in the array
      // and therefore form a clique among themselves
      for (size_t i = 0; i < adjElms.size (); ++i) {
        // populate the upper triangle of the adj matrix
        for (size_t j = i + 1; j < adjElms.size (); ++j) {
//          if (!adjElms[i].hasNeighbor (adjElms[j])) {
          if (mgraph.findEdge(adjElms[i], adjElms[j]) == mgraph.edge_end(adjElms[i])) {
            ++numEdges;
          }
          mgraph.addEdge (adjElms[i], adjElms[j]);
        }
      }

    }

#ifdef USE_LC_GRAPH
    graph.copyFromGraph (mgraph);
#endif

    printf ("Graph created with %u nodes and %d edges\n", graph.size (), numEdges);
  }
  
  virtual void initRemaining (const MeshInit& meshInit, const GlobalVec& g) {

    galois::StatTimer t_graph ("Time spent in creating the graph: ");

    t_graph.start ();
    genElemAdjGraph (meshInit, g);
    t_graph.stop ();
  }

  //! Functor for loop body
  struct Process {
    Graph& graph;
    std::vector<int>& inDegVec;
    MeshInit& meshInit;
    GlobalVec& g;
    PerThrdLocalVec& perIterLocalVec;
    bool createSyncFiles;
    IterCounter& iter;

    Process (
        Graph& graph,
        std::vector<int>& inDegVec,
        MeshInit& meshInit,
        GlobalVec& g,
        PerThrdLocalVec& perIterLocalVec,
        bool createSyncFiles,
        IterCounter& iter):

        graph (graph),
        inDegVec (inDegVec),
        meshInit (meshInit),
        g (g),
        perIterLocalVec (perIterLocalVec),
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
      void operator () (GNode& src, ContextTy& lwl) {
        // one-shot optimization: acquire abstract locks on active node and
        // neighbors (all its neighbors, in this case) before performing any modifications

        AVI* srcAVI = graph.getData (src, galois::MethodFlag::WRITE);

        for (Graph::edge_iterator e = graph.edge_begin (src, galois::MethodFlag::WRITE)
             , ende = graph.edge_end (src, galois::MethodFlag::WRITE); e != ende; ++e) {
        }


        // past the fail-safe point now


        int inDeg = inDegVec[srcAVI->getGlobalIndex ()];
        // assert  inDeg == 0 : String.format ("active node %s with inDeg = %d\n", srcAVI, inDeg);

        //        // TODO: DEBUG
        //        std::cout << "Processing element: " << srcAVI->toString() << std::endl;

        assert (inDeg == 0);


        LocalVec& l = *perIterLocalVec.getLocal();

        AVIabstractMain::simulate(srcAVI, meshInit, g, l, createSyncFiles);


        // update the inEdges count and determine
        // which neighbor is at local minimum and needs to be added to the worklist

        for (Graph::edge_iterator e = graph.edge_begin (src, galois::MethodFlag::UNPROTECTED)
            , ende = graph.edge_end (src, galois::MethodFlag::UNPROTECTED); e != ende; ++e) {

          const GNode& dst = graph.getEdgeDst (e);
          AVI* dstAVI = graph.getData (dst, galois::MethodFlag::UNPROTECTED);

          if (AVIComparator::compare (srcAVI, dstAVI) > 0) {
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


        iter += 1;

        // if (iter.get () == 5000) {
           // meshInit.writeMesh ();
           // meshInit.plotMeshCenters ();
        // }

      }
  };

  template <typename T>
  void initWorkList (std::vector<GNode>& initWL, std::vector<T>& inDegVec) {

    galois::StatTimer t_wl ("Time to populate the worklist");
    t_wl.start ();

    for (Graph::iterator i = graph.begin (), e = graph.end (); i != e; ++i) {
      const GNode& src = *i;
      AVI* srcAVI = graph.getData (src, galois::MethodFlag::UNPROTECTED);

      // calculate the in degree of src by comparing it against its neighbors
      for (Graph::edge_iterator e = graph.edge_begin  (src, galois::MethodFlag::UNPROTECTED), 
          ende = graph.edge_end (src, galois::MethodFlag::UNPROTECTED); e != ende; ++e) {
        
        GNode dst = graph.getEdgeDst (e);
        AVI* dstAVI = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
        if (AVIComparator::compare (srcAVI, dstAVI) > 0) {
          ++inDegVec[srcAVI->getGlobalIndex ()];
        }
      }

      // if src is less than all its neighbors then add to initWL
      if (inDegVec[srcAVI->getGlobalIndex ()] == 0) {
        initWL.push_back (src);
      }
    }

    t_wl.stop ();
    printf ("Initial worklist contains %zd elements\n", initWL.size ());
    
  }
public:

  virtual  void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    /////////////////////////////////////////////////////////////////
    // populate an initial  worklist
    /////////////////////////////////////////////////////////////////
    std::vector<int> inDegVec(meshInit.getNumElements (), 0);

    std::vector<GNode> initWL;

    initWorkList (initWL, inDegVec);


//    // TODO: DEBUG
//    std::cout << "Initial Worklist = " << std::endl;
//    for (size_t i = 0; i < initWL.size (); ++i) {
//      std::cout << graph.getData (initWL[i], galois::MethodFlag::UNPROTECTED)->toString () << ", ";
//    }
//    std::cout << std::endl;

    /////////////////////////////////////////////////////////////////
    // perform the simulation
    /////////////////////////////////////////////////////////////////

    // uncomment to plot the mesh
    meshInit.writeMesh ();
    // meshInit.plotMeshCenters ();
    writeAdjacencyGraph (meshInit, graph); 

    // temporary matrices
    size_t nrows = meshInit.getSpatialDim ();
    size_t ncols = meshInit.getNodesPerElem();

    PerThrdLocalVec perIterLocalVec;
    for (unsigned int i = 0; i < perIterLocalVec.size(); ++i)
      *perIterLocalVec.getRemote(i) = LocalVec(nrows, ncols);

    IterCounter iter;

    Process p(graph, inDegVec, meshInit, g, perIterLocalVec, createSyncFiles, iter);

    galois::for_each(galois::iterate(initWL), p, galois::wl<AVIWorkList>());

    printf ("iterations = %zd\n", iter.reduce ());

  }


  static void writeAdjacencyGraph (const MeshInit& meshInit, Graph& graph,
      const char* nodesFileName="mesh-nodes.csv", const char* edgesFileName="mesh-edges.csv") {

    if (meshInit.getSpatialDim () != 2) {
        std::cerr << "implemented for 2D elements only" << std::endl;
        abort ();
    }


    FILE* nodesFile = fopen (nodesFileName, "w");
    if (nodesFile == NULL) { abort (); }

    fprintf (nodesFile, "nodeId, inDeg, outDeg, centerX, centerY, timeStamp\n");

    // a set of edges computed by picking outgoing edges for each node
    std::vector<std::pair<GNode, GNode> > outEdges;

    VecDouble center (meshInit.getSpatialDim(), 0.0);

    for (Graph::iterator i = graph.begin (), e = graph.end (); i != e; ++i) {
      const GNode& src = *i;
      AVI* srcAVI = graph.getData (src, galois::MethodFlag::UNPROTECTED);

      size_t inDeg = 0;
      // calculate the in degree of src by comparing it against its neighbors
      for (Graph::edge_iterator e = graph.edge_begin (src, galois::MethodFlag::UNPROTECTED)
          , ende = graph.edge_end (src, galois::MethodFlag::UNPROTECTED); e != ende; ++e) {


        GNode dst = graph.getEdgeDst (e);
        AVI* dstAVI = graph.getData (dst, galois::MethodFlag::UNPROTECTED);
        if (AVIComparator::compare (srcAVI, dstAVI) > 0) {
          ++inDeg;

        } else { // is an out-going edge
          outEdges.push_back (std::make_pair(src, dst));
        }
      }

      // size_t outDeg = graph.neighborsSize(src, galois::MethodFlag::UNPROTECTED) - inDeg;
      size_t outDeg = std::distance (graph.edge_begin (src, galois::MethodFlag::UNPROTECTED), graph.edge_end (src, galois::MethodFlag::UNPROTECTED));

      std::fill (center.begin (), center.end (), 0.0);
      srcAVI->getElement ().getGeometry ().computeCenter (center);

      fprintf (nodesFile, "%zd, %zd, %zd, %g, %g, %g\n",
          srcAVI->getGlobalIndex(), inDeg, outDeg, center[0], center[1], srcAVI->getNextTimeStamp());

    }

    fclose (nodesFile);

    FILE* edgesFile = fopen (edgesFileName, "w");
    if (edgesFile == NULL) { abort (); }


    fprintf (edgesFile, "srcId, dstId\n");
    for (std::vector<std::pair<GNode, GNode> >::const_iterator i = outEdges.begin(), ei = outEdges.end();
        i != ei; ++i) {
       size_t srcId = graph.getData (i->first, galois::MethodFlag::UNPROTECTED)->getGlobalIndex ();
       size_t dstId = graph.getData (i->second, galois::MethodFlag::UNPROTECTED)->getGlobalIndex ();

       fprintf (edgesFile, "%zd, %zd\n", srcId, dstId);

    }

    fclose (edgesFile);

  }

};

#endif

/** AVI ODG graph readonly -*- C++ -*-
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

#ifndef AVI_ODG_READONLY_H
#define AVI_ODG_READONLY_H


#include "Galois/Graph/Graph.h"
#include "Galois/Graph/FileGraph.h"

#include "Galois/Galois.h"
#include "Galois/Atomic.h"
#include "Galois/Timer.h"

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/PerThreadWorkList.h"

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
#include "AVIodgExplicit.h"

/**
 * AVI unordered algorithm that uses atomic integers 
 * and no abstract locks
 */
class AVIodgReadonly: public AVIodgExplicit {
  // typedef Galois::GAtomicPadded<bool> AtomicBool;
  // typedef std::vector<AtomicBool> VecAtomicBool;
  //

  typedef Galois::Runtime::PerThreadVector<GNode> WL_ty;

protected:

  virtual const std::string getVersion () const {
    return "ODG readonly (two-phases), no edges reversed, "
      "just sources computed by reading the graph";
  }
  
 
  // has highest priority among its neighbors and 
  // thus is a source in ODG
  inline static bool isSrcInODG (Graph& graph, const GNode& src) {
    AVI* srcAVI = graph.getData (src, Galois::NONE);
    return isSrcInODG (graph, src, srcAVI);
  }

  inline static bool isSrcInODG (Graph& graph, const GNode& src, AVI* srcAVI) {

    assert (graph.getData (src, Galois::NONE) == srcAVI);

    bool ret = true;
    for (Graph::edge_iterator e = graph.edge_begin (src, Galois::NONE)
        , ende = graph.edge_end (src, Galois::NONE); e != ende; ++e) {
      AVI* dstAVI = graph.getData (graph.getEdgeDst (e), Galois::NONE);

      if (AVIComparator::compare (srcAVI, dstAVI) > 0) {
        ret = false;
        break;
      }
    }

    return ret;
  }

  struct FindSources {
    Graph& graph;
    MeshInit& meshInit;
    WL_ty& sources;
    IterCounter& findIter;


    FindSources (
        Graph& graph, 
        MeshInit& meshInit,
        WL_ty& sources,
        IterCounter& findIter)
      : 
        graph (graph), 
        meshInit (meshInit),
        sources (sources),
        findIter (findIter) 
    {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const GNode& src) {
      AVI* srcAVI = graph.getData (src, Galois::NONE);

      if (srcAVI->getNextTimeStamp () < meshInit.getSimEndTime ()) {
        if (isSrcInODG (graph, src, srcAVI)) {
          sources.get ().push_back (src);
        }
      }

      findIter += 1;
    }
  };


  struct ApplyOperator {
    Graph& graph;
    MeshInit& meshInit;
    GlobalVec& g;
    Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec;
    bool createSyncFiles;
    IterCounter& opIter;

    ApplyOperator (
        Graph& graph,
        MeshInit& meshInit,
        GlobalVec& g,
        Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec,
        bool createSyncFiles,
        IterCounter& opIter)
      :

        graph (graph),
        meshInit (meshInit),
        g (g),
        perIterLocalVec (perIterLocalVec),
        createSyncFiles (createSyncFiles),
        opIter (opIter) 
    {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const GNode& src) {

      AVI* srcAVI = graph.getData (src, Galois::NONE);
      assert (isSrcInODG (graph, src, srcAVI));

      LocalVec& l = *perIterLocalVec.getLocal();

      AVIabstractMain::simulate(srcAVI, meshInit, g, l, createSyncFiles);

      // for debugging, remove later
      opIter += 1;
    }
  };

  
public:

  virtual  void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {

    // temporary matrices
    const size_t nrows = meshInit.getSpatialDim ();
    const size_t ncols = meshInit.getNodesPerElem();

    Galois::Runtime::PerThreadStorage<LocalVec> perIterLocalVec;
    for (unsigned int i = 0; i < perIterLocalVec.size(); ++i)
      *perIterLocalVec.getRemote(i) = LocalVec(nrows, ncols);

    IterCounter findIter;
    IterCounter opIter;


    Galois::TimeAccumulator findTime;
    Galois::TimeAccumulator opTime;

    WL_ty sources;

    size_t rounds = 0;
    while (true) {
      ++rounds;
      sources.clear_all ();

      findTime.start ();
      Galois::do_all (graph.begin (), graph.end (),
          FindSources (graph, meshInit, sources, findIter),
          "find_sources_loop");
      findTime.stop ();

      // std::cout << "Num. Sources found: " << sources.size_all () << std::endl;
      if (sources.empty_all ()) {
        break;
      }

      opTime.start ();
      Galois::do_all (sources.begin_all (), sources.end_all (),
          ApplyOperator (graph, meshInit, g, perIterLocalVec, createSyncFiles, opIter),
          "apply_operator_loop");
      opTime.stop ();


    }

    std::cout << "Number of rounds executed: " << rounds << std::endl;


    std::cout << "iterations in finding sources = " << findIter.reduce () << std::endl;
    std::cout << "Time spent finding sources = " << findTime.get () << std::endl;

    std::cout << "iterations in applying operator = " << opIter.reduce () << std::endl;
    std::cout << "Time spent in applying operator = " << opTime.get () << std::endl;


  }



};

#endif // AVI_ODG_READONLY_H

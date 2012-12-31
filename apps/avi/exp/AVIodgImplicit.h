/** AVI a version without explicit ODG, ODG info computed from neighborhood -*- C++ -*-
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

#ifndef AVI_ODG_IMPLICIT_H
#define AVI_ODG_IMPLICIT_H

#include "Galois/Galois.h"
#include "Galois/Atomic.h"
#include "Galois/Timer.h"

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/PerThreadWorkList.h"

#include "Galois/Runtime/WorkList.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <set>
#include <utility>

#include <cassert>

#include <boost/iterator/counting_iterator.hpp>

#include "AuxDefs.h"
#include "AVI.h"
#include "Element.h"

#include "AVIabstractMain.h"

class AVIodgImplicit: public AVIabstractMain {

protected:

  static const bool DEBUG = false;

  typedef std::vector<AVI*> VecAVI;
  typedef std::vector<VecAVI> VecVecAVI;

  typedef Galois::Runtime::PerThreadVector<AVI*> WL_ty;

  VecVecAVI vertexSharers;

  virtual const std::string getVersion () const {
    return "Parallel version, where ODG is implicit, i.e. sources (in ODG) are computed over neighborhood";
  }



  void computeVertexAdjacency (const MeshInit& meshInit) {

    vertexSharers.clear ();
    vertexSharers.resize (meshInit.getNumNodes (), VecAVI ());

    for (std::vector<AVI*>::const_iterator i = meshInit.getAVIVec ().begin ()
        , endi = meshInit.getAVIVec ().end (); i != endi; ++i) {

      AVI* avi = *i;

      const std::vector<GlobalNodalIndex>& conn = avi->getGeometry ().getConnectivity ();

      for (std::vector<GlobalNodalIndex>::const_iterator n = conn.begin ()
          , endn = conn.end (); n != endn; ++n) {
        assert ((*n) < vertexSharers.size ());
        vertexSharers[*n].push_back (avi);
      }
    }
  }

  
  virtual void initRemaining (const MeshInit& meshInit, const GlobalVec& g) {

    Galois::StatTimer t_adj ("Time spent in creating vertex adjacency: ");

    t_adj.start ();
    computeVertexAdjacency (meshInit);
    t_adj.stop ();
  }

  static inline AVI* getMin (const VecAVI& vec) {
    assert (vec.size () != 0);

    if (vec.size () == 0) {
      return NULL;
    } else {

      AVI* min = vec[0];
      for (size_t i = 1; i < vec.size (); ++i) {
        if (AVIComparator::compare (vec[i], min) < 0) { // vec[i] < min
          min = vec[i];
        }
      }

      return min;
    }
  }

  static inline bool isMin (const VecAVI& vec, AVI* saidMin) {
    assert (vec.size () != 0);

    bool ret = true;

    for (VecAVI::const_iterator i = vec.begin (), endi = vec.end ();
        i != endi; ++i) {

      if (AVIComparator::compare (*i, saidMin) < 0) { // vec[i] < saidMin
        ret = false;
        break;
      }
    }

    return ret;
  }


  // avi element is highest priority among elements that share a vertex with it
  static inline bool isSrcInODG (const VecVecAVI& vertexSharers, AVI* avi) {

    assert (avi != NULL);

    const std::vector<GlobalNodalIndex>& conn = avi->getGeometry ().getConnectivity ();
    
    bool ret = true;
    for (std::vector<GlobalNodalIndex>::const_iterator n = conn.begin ()
        , endn = conn.end (); n != endn; ++n) {

      if (!isMin (vertexSharers[*n], avi)) {
        ret = false;
        break;
      }
    }

    return ret;
  }

  // used to avoid duplicate adds to workList.
  // Only the leading vertex (with min id) addes a src avi element to workList
  static inline bool isLeader (const AVI* avi, const GlobalNodalIndex& n) {
      const std::vector<GlobalNodalIndex>& conn = avi->getGeometry ().getConnectivity ();

      assert (std::find (conn.begin (), conn.end (), n) != conn.end ());

      return (n == *(std::min_element (conn.begin (), conn.end ())));
  }

  struct FindSources {
    VecVecAVI& vertexSharers;
    MeshInit& meshInit;
    WL_ty& sources;
    IterCounter& findIter;

    FindSources (
        VecVecAVI& vertexSharers,
        MeshInit& meshInit,
        WL_ty& sources,
        IterCounter& findIter)
      :
        vertexSharers (vertexSharers),
        meshInit (meshInit),
        sources (sources),
        findIter (findIter) 
    {}


    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const GlobalNodalIndex& n) {
      assert (n < vertexSharers.size ());
      findIter += 1;

      AVI* avi = getMin (vertexSharers[n]);

      assert (avi != NULL);

      if ((avi->getNextTimeStamp () < meshInit.getSimEndTime ()) 
          && isLeader (avi, n) && isSrcInODG (vertexSharers, avi)) {

        sources.get ().push_back (avi);
      }
      
    }

  };



  /**
   * Functor for loop body
   */
  struct ApplyOperator {

    VecVecAVI& vertexSharers;
    MeshInit& meshInit;
    GlobalVec& g;
    Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec;
    bool createSyncFiles;
    IterCounter& opIter;

    ApplyOperator (
        VecVecAVI& vertexSharers,
        MeshInit& meshInit,
        GlobalVec& g,
        Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec,
        bool createSyncFiles,
        IterCounter& opIter):

        vertexSharers (vertexSharers),
        meshInit (meshInit),
        g (g),
        perIterLocalVec (perIterLocalVec),
        createSyncFiles (createSyncFiles),
        opIter (opIter) {}
    

    void operator () (AVI* srcAVI) {

      assert (isSrcInODG (vertexSharers, srcAVI));

      LocalVec& l = *perIterLocalVec.getLocal();

      AVIabstractMain::simulate(srcAVI, meshInit, g, l, createSyncFiles);

      opIter += 1;


    }
  };

public:

  /**
   * For the in degree vector, we use a vector of atomic integers
   * This along with other changes in the loop body allow us to 
   * no use abstract locks. @see process
   */
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
      Galois::do_all (
          boost::counting_iterator<GlobalNodalIndex> (0),
          boost::counting_iterator<GlobalNodalIndex> (meshInit.getNumNodes ()),
          FindSources (vertexSharers, meshInit, sources, findIter),
          "find_sources_loop");
      findTime.stop ();

      // std::cout << "Num. Sources found: " << sources.size_all () << std::endl;
      if (sources.empty_all ()) {
        break;
      }

      opTime.start ();
      Galois::do_all (sources.begin_all (), sources.end_all (),
          ApplyOperator (vertexSharers, meshInit, g, perIterLocalVec, createSyncFiles, opIter),
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




#endif // AVI_ODG_IMPLICIT_H 

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

#ifndef AVI_ODG_AUTO_H
#define AVI_ODG_AUTO_H

#include "Galois/Galois.h"
#include "Galois/Atomic.h"

#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ODGtwoPhase.h"

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

class AVIodgAuto: public AVIabstractMain {

protected:

  typedef std::vector<AVI*> VecAVI;
  typedef std::vector<VecAVI> VecVertexAdj;

#ifdef USE_SHARE_LIST
  typedef Galois::Runtime::NhoodItemShareList<AVI*> NItem;
#else 
  typedef Galois::Runtime::NhoodItemPriorityLock<AVI*> NItem;
#endif 

  typedef std::vector<NItem> VecNItem;
  typedef std::vector<NItem*> VecNItemPtr;

  VecNItem vertices;
  VecNItemPtr vtxPtrs;


  virtual const std::string getVersion () const {
    return "Parallel version, ODG automatically managed";
  }

  
  virtual void initRemaining (const MeshInit& meshInit, const GlobalVec& g) {
    vertices.clear ();
    vtxPtrs.clear ();

    for (int i = 0; i < meshInit.getNumNodes (); ++i) {
      vertices.push_back (NItem (i));
    }

    for (VecNItem::iterator i = vertices.begin ()
        , endi = vertices.end (); i != endi; ++i) {
      vtxPtrs.push_back (&(*i));
    }
  }


  struct NhoodVisitor {
    VecNItem& vertices;
    AVIComparator& cmp;

    NhoodVisitor (VecNItem& vertices, AVIComparator& cmp): vertices (vertices), cmp (cmp) {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (AVI* avi) {
      assert (avi != NULL);

      const std::vector<GlobalNodalIndex>& conn = avi->getGeometry ().getConnectivity ();

      for (std::vector<GlobalNodalIndex>::const_iterator n = conn.begin ()
          , endn = conn.end (); n != endn; ++n) {

        vertices[*n].visit (cmp);
      }
    }
  };

  /**
   * Functor for loop body
   */
  struct OperFunc {

    MeshInit& meshInit;
    GlobalVec& g;
    Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec;
    bool createSyncFiles;
    IterCounter& iter;

    OperFunc (
        MeshInit& meshInit,
        GlobalVec& g,
        Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec,
        bool createSyncFiles,
        IterCounter& iter):

        meshInit (meshInit),
        g (g),
        perIterLocalVec (perIterLocalVec),
        createSyncFiles (createSyncFiles),
        iter (iter) {}

    // TODO: add note on add to WL semantics i.e. adds should happen on commit and
    // not earlier
    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void addToWL (C& lwl, AVI* avi) {

      if (avi->getNextTimeStamp () < meshInit.getSimEndTime ()) {
        lwl.push_back (avi);
      }

    }


    template <typename C> 
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (AVI* srcAVI, C& lwl) {
      // for debugging, remove later
      iter += 1;

      LocalVec& l = *perIterLocalVec.getLocal();

      AVIabstractMain::simulate(srcAVI, meshInit, g, l, createSyncFiles);

      addToWL (lwl, srcAVI);

    }
  };

public:

  /**
   * For the in degree vector, we use a vector of atomic integers
   * This along with other changes in the loop body allow us to 
   * no use abstract locks. @see process
   */
  virtual void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {

    const VecAVI& elems = meshInit.getAVIVec ();


    // temporary matrices
    const size_t nrows = meshInit.getSpatialDim ();
    const size_t ncols = meshInit.getNodesPerElem();

    Galois::Runtime::PerThreadStorage<LocalVec> perIterLocalVec;
    for (unsigned int i = 0; i < perIterLocalVec.size(); ++i)
      *perIterLocalVec.getRemote(i) = LocalVec(nrows, ncols);

    IterCounter iter;

    AVIComparator cmp;

    NhoodVisitor nv (vertices, cmp);
    OperFunc op (meshInit, g, perIterLocalVec, createSyncFiles, iter);


#ifdef USE_SHARE_LIST
    // Galois::for_each<AVIWorkList> (initWL.begin (), initWL.end (), p);
    Galois::Runtime::for_each_ordered (
        elems.begin (), elems.end (),
        vtxPtrs.begin (), vtxPtrs.end (),
        op, nv, cmp);
#else 
    Galois::Runtime::for_each_ordered (
        elems.begin (), elems.end (),
        op, nv, cmp);
#endif

    printf ("iterations = %d\n", iter.reduce ());

  }



};




#endif // AVI_ODG_IMPLICIT_H 

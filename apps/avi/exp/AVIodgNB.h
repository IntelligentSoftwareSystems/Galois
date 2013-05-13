/** AVI a version without explicit ODG using deterministic infrastructure -*- C++ -*-
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

#ifndef AVI_ODG_NB_H
#define AVI_ODG_NB_H

#include "Galois/Galois.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/LCordered.h"

// #include <boost/iterator/transform_iterator.hpp>

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

class AVIodgNB: public AVIabstractMain {
protected:

  typedef Galois::Graph::FirstGraph<void*,void,true> Graph;
  typedef Graph::GraphNode Lockable;
  typedef std::vector<Lockable> Locks;

  Graph graph;
  Locks locks;

  virtual const std::string getVersion() const {
    return "ODG automatically managed, single phase, no barrier ";
  }
  
  virtual void initRemaining(const MeshInit& meshInit, const GlobalVec& g) {
    assert(locks.empty());
    locks.reserve(meshInit.getNumNodes());
    for (int i = 0; i < meshInit.getNumNodes(); ++i) {
      locks.push_back(graph.createNode(0));
    }
  }

  struct Update {
    double timestamp;
    AVI* avi;
    Update(double t, AVI* a): timestamp(t), avi(a) { }

    friend std::ostream& operator << (std::ostream& out, const Update& up) {
      return (out << "(id:" << up.avi->getGlobalIndex() << ", ts:" << up.timestamp << ")");
    }

    struct Comparator {
      bool operator () (const Update& left, const Update& right) const {
        int result = DoubleComparator::compare (left.timestamp, right.timestamp);

        if (result == 0) { 
          result = left.avi->getGlobalIndex () - right.avi->getGlobalIndex ();
        }

        return (result < 0);
      };
    };
  };



  struct MakeItem: public std::unary_function<AVI*,Update> {
    Update operator()(AVI* avi) const { return Update(avi->getNextTimeStamp(), avi); }
  };

  struct Prefix {
    Graph& graph;
    Locks& locks;

    Prefix(Graph& g, Locks& l): graph(g), locks(l) { }

    void operator () (const Update& item) {
      typedef std::vector<GlobalNodalIndex> V;

      const V& conn = item.avi->getGeometry().getConnectivity();

      for (V::const_iterator ii = conn.begin(), ei = conn.end(); ii != ei; ++ii) {
        graph.getData(locks[*ii]);
      }
    }
  };

  struct Process {
    MeshInit& meshInit;
    GlobalVec& g;
    Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec;
    bool createSyncFiles;
    IterCounter& niter;

    Process(
        MeshInit& meshInit,
        GlobalVec& g,
        Galois::Runtime::PerThreadStorage<LocalVec>& perIterLocalVec,
        bool createSyncFiles,
        IterCounter& niter):
      meshInit(meshInit),
      g(g),
      perIterLocalVec(perIterLocalVec),
      createSyncFiles(createSyncFiles),
      niter(niter) { }

    template <typename WL>
    void operator()(const Update& item, WL& wl) {
      // for debugging, remove later
      niter += 1;

      LocalVec& l = *perIterLocalVec.getLocal();

      AVIabstractMain::simulate(item.avi, meshInit, g, l, createSyncFiles);

      if (item.avi->getNextTimeStamp() < meshInit.getSimEndTime()) {
        wl.push_back (Update(item.avi->getNextTimeStamp(), item.avi));
      }
    }
  };


public:
  virtual void runLoop(MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    const size_t nrows = meshInit.getSpatialDim();
    const size_t ncols = meshInit.getNodesPerElem();

    Galois::Runtime::PerThreadStorage<LocalVec> perIterLocalVec;
    for (unsigned int i = 0; i < perIterLocalVec.size(); ++i)
      *perIterLocalVec.getRemote(i) = LocalVec(nrows, ncols);

    IterCounter niter;

    Prefix prefix(graph, locks);
    Process op(meshInit, g, perIterLocalVec, createSyncFiles, niter);

    const std::vector<AVI*>& elems = meshInit.getAVIVec();

    Galois::Runtime::for_each_ordered_lc<CHUNK_SIZE> (
        boost::make_transform_iterator(elems.begin(), MakeItem()),
        boost::make_transform_iterator(elems.end(), MakeItem()),
        Update::Comparator (),
        op, prefix);
        

    printf("iterations = %zd\n", niter.reduce());
  }
};

#endif

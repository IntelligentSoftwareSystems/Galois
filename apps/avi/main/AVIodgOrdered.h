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

#ifndef AVI_ODG_ORDERED_H
#define AVI_ODG_ORDERED_H

#include "Galois/Galois.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/WorkList.h"

#include <boost/iterator/transform_iterator.hpp>

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


enum ExecType {
  useAddRem,
  useTwoPhase,
};

static cll::opt<ExecType> execType (
    cll::desc ("Ordered Executor Type:"),
    cll::values (
      clEnumVal(useAddRem, "Use Add-Remove executor"),
      clEnumVal(useTwoPhase, "Use Two-Phase executor"),
      clEnumValEnd),
    cll::init (useAddRem));

class AVIodgOrdered: public AVIabstractMain {
protected:
  typedef Galois::Graph::FirstGraph<void*,void,true> Graph;
  typedef Graph::GraphNode Lockable;
  typedef std::vector<Lockable> Locks;

  Graph graph;
  Locks locks;

  virtual const std::string getVersion() const {
    return "Parallel version, ODG automatically managed";
  }
  
  virtual void initRemaining(const MeshInit& meshInit, const GlobalVec& g) {
    assert(locks.empty());
    locks.reserve(meshInit.getNumNodes());
    for (int i = 0; i < meshInit.getNumNodes(); ++i) {
      locks.push_back(graph.createNode(0));
    }
  }

  struct Update {
    AVI* avi;
    double ts;
    explicit Update(AVI* a, double t): avi(a), ts(t) { }

    Update updatedCopy () const {
      return Update (avi, avi->getNextTimeStamp ());
    }

    friend std::ostream& operator << (std::ostream& out, const Update& up) {
      return (out << "(id:" << up.avi->getGlobalIndex() << ", ts:" << up.ts << ")");
    }
  };
  struct Comparator {

    bool operator() (const Update& a, const Update& b) const {
      int c = DoubleComparator::compare (a.ts, b.ts);
      if (c == 0) { 
        c = a.avi->getGlobalIndex () - b.avi->getGlobalIndex ();
      }
      return (c < 0);
    }
  };


  struct MakeUpdate: public std::unary_function<AVI*,Update> {
    Update operator()(AVI* avi) const { return Update(avi, avi->getNextTimeStamp ()); }
  };

  struct NhoodVisit {
    Graph& graph;
    Locks& locks;

    NhoodVisit(Graph& g, Locks& l): graph(g), locks(l) { }

    template <typename C>
    void operator()(const Update& item, C&) {
      typedef std::vector<GlobalNodalIndex> V;

      const V& conn = item.avi->getGeometry().getConnectivity();

      for (V::const_iterator ii = conn.begin(), ei = conn.end(); ii != ei; ++ii) {
        graph.getData(locks[*ii]);
      }
    }
  };

  struct NhoodVisitAddRem: public NhoodVisit {
    typedef int tt_has_fixed_neighborhood;
    NhoodVisitAddRem (Graph& g, Locks& l): NhoodVisit (g, l) {}
  };

  struct Process {
    MeshInit& meshInit;
    GlobalVec& g;
    GaloisRuntime::PerThreadStorage<LocalVec>& perIterLocalVec;
    bool createSyncFiles;
    IterCounter& niter;

    Process(
        MeshInit& meshInit,
        GlobalVec& g,
        GaloisRuntime::PerThreadStorage<LocalVec>& perIterLocalVec,
        bool createSyncFiles,
        IterCounter& niter):
      meshInit(meshInit),
      g(g),
      perIterLocalVec(perIterLocalVec),
      createSyncFiles(createSyncFiles),
      niter(niter) { }

    void operator()(const Update& item, Galois::UserContext<Update>& ctx) {
      // for debugging, remove later
      niter += 1;

      LocalVec& l = *perIterLocalVec.getLocal();

      AVIabstractMain::simulate(item.avi, meshInit, g, l, createSyncFiles);

      if (item.avi->getNextTimeStamp() < meshInit.getSimEndTime()) {
        ctx.push(item.updatedCopy ());
      }
    }
  };


public:
  virtual void runLoop(MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    const size_t nrows = meshInit.getSpatialDim();
    const size_t ncols = meshInit.getNodesPerElem();

    GaloisRuntime::PerThreadStorage<LocalVec> perIterLocalVec;
    for (unsigned int i = 0; i < perIterLocalVec.size(); ++i)
      *perIterLocalVec.getRemote(i) = LocalVec(nrows, ncols);

    IterCounter niter;

    NhoodVisit nhVisitor(graph, locks);
    NhoodVisitAddRem nhVisitorAddRem (graph, locks);
    Process p(meshInit, g, perIterLocalVec, createSyncFiles, niter);

    const std::vector<AVI*>& elems = meshInit.getAVIVec();

    switch (execType) {
      case useAddRem:
        Galois::for_each_ordered (
            boost::make_transform_iterator(elems.begin(), MakeUpdate()),
            boost::make_transform_iterator(elems.end(), MakeUpdate()), 
            Comparator(), nhVisitorAddRem, p);
        break;
      case useTwoPhase:
        Galois::for_each_ordered (
            boost::make_transform_iterator(elems.begin(), MakeUpdate()),
            boost::make_transform_iterator(elems.end(), MakeUpdate()), 
            Comparator(), nhVisitor, p);
        break;
      default:
        GALOIS_ERROR(true, "Unknown executor type");
        break;
    }


    printf("iterations = %lu\n", niter.reduce());
  }
};

#endif

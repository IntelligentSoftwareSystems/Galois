/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef AVI_ORDERED_SPEC_H
#define AVI_ORDERED_SPEC_H

#include "galois/Galois.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/worklists/WorkList.h"
#include "galois/runtime/LevelExecutor.h"

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

class AVIlevelExec : public AVIabstractMain {
protected:
  typedef galois::graphs::MorphGraph<void*, void, true> Graph;
  typedef Graph::GraphNode Lockable;
  typedef std::vector<Lockable> Locks;

  Graph graph;
  Locks locks;

  virtual const std::string getVersion() const {
    return "Parallel version, Level-by-Level executor ";
  }

  virtual void initRemaining(const MeshInit& meshInit, const GlobalVec& g) {
    assert(locks.empty());
    locks.reserve(meshInit.getNumNodes());
    for (int i = 0; i < meshInit.getNumNodes(); ++i) {
      locks.push_back(graph.createNode(nullptr));
    }
  }

  struct Update {
    AVI* avi;
    double ts;
    explicit Update(AVI* a, double t) : avi(a), ts(t) {}

    Update updatedCopy() const { return Update(avi, avi->getNextTimeStamp()); }

    friend std::ostream& operator<<(std::ostream& out, const Update& up) {
      return (out << "(id:" << up.avi->getGlobalIndex() << ", ts:" << up.ts
                  << ")");
    }
  };

  struct GetNextTS {
    double operator()(const Update& up) const { return up.ts; }
  };

  struct MakeUpdate : public std::unary_function<AVI*, Update> {
    Update operator()(AVI* avi) const {
      return Update(avi, avi->getNextTimeStamp());
    }
  };

  struct NhoodVisit {
    Graph& graph;
    Locks& locks;

    NhoodVisit(Graph& g, Locks& l) : graph(g), locks(l) {}

    template <typename C>
    void operator()(const Update& item, C&) const {
      typedef VecSize_t V;

      const V& conn = item.avi->getGeometry().getConnectivity();

      for (V::const_iterator ii = conn.begin(), ei = conn.end(); ii != ei;
           ++ii) {
        graph.getData(locks[*ii]);
      }
    }
  };

  struct NhoodVisitAddRem : public NhoodVisit {
    typedef int tt_has_fixed_neighborhood;
    NhoodVisitAddRem(Graph& g, Locks& l) : NhoodVisit(g, l) {}
  };

  struct Process {

    static const unsigned CHUNK_SIZE = 16;

    MeshInit& meshInit;
    GlobalVec& g;
    galois::substrate::PerThreadStorage<LocalVec>& perIterLocalVec;
    bool createSyncFiles;
    IterCounter& niter;

    Process(MeshInit& meshInit, GlobalVec& g,
            galois::substrate::PerThreadStorage<LocalVec>& perIterLocalVec,
            bool createSyncFiles, IterCounter& niter)
        : meshInit(meshInit), g(g), perIterLocalVec(perIterLocalVec),
          createSyncFiles(createSyncFiles), niter(niter) {}

    void operator()(const Update& item,
                    galois::UserContext<Update>& ctx) const {
      // for debugging, remove later
      niter += 1;
      LocalVec& l = *perIterLocalVec.getLocal();

      AVI* avi = item.avi;

      if (createSyncFiles) {
        meshInit.writeSync(*avi, g.vecQ, g.vecV_b, g.vecT);
      }

      AVIabstractMain::simulate(item.avi, meshInit, g, l, createSyncFiles);

      if (avi->getNextTimeStamp() < meshInit.getSimEndTime()) {
        ctx.push(item.updatedCopy());
      }
    }
  };

public:
  virtual void runLoop(MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    const size_t nrows = meshInit.getSpatialDim();
    const size_t ncols = meshInit.getNodesPerElem();

    galois::substrate::PerThreadStorage<LocalVec> perIterLocalVec;
    for (unsigned int i = 0; i < perIterLocalVec.size(); ++i)
      *perIterLocalVec.getRemote(i) = LocalVec(nrows, ncols);

    IterCounter niter;

    NhoodVisit nhVisitor(graph, locks);
    Process p(meshInit, g, perIterLocalVec, createSyncFiles, niter);

    const std::vector<AVI*>& elems = meshInit.getAVIVec();

    // galois::for_each_ordered (
    galois::runtime::for_each_ordered_level(
        galois::runtime::makeStandardRange(
            boost::make_transform_iterator(elems.begin(), MakeUpdate()),
            boost::make_transform_iterator(elems.end(), MakeUpdate())),
        GetNextTS(), std::less<double>(), nhVisitor, p, "level-by-level-avi");

    printf("iterations = %lu\n", niter.reduce());
  }
};

#endif

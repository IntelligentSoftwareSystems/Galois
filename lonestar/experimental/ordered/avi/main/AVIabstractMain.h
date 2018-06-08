/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef AVI_ABSTRACT_MAIN_H_
#define AVI_ABSTRACT_MAIN_H_

#include <vector>
#include <string>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>

#include <cassert>
#include <cstdlib>
#include <cstdio>


#include "AVI.h"
#include "MeshInit.h"
#include "GlobalVec.h"
#include "LocalVec.h"



#include "galois/Reduction.h"
#include "galois/Atomic.h"
#include "galois/Galois.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include "galois/runtime/Profile.h"
#include "galois/substrate/PerThreadStorage.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static cll::opt<std::string> fileNameOpt("f", cll::desc("<input mesh file>"), cll::Required);
static cll::opt<int> spDimOpt("d", cll::desc("spatial dimensionality of the problem i.e. 2 for 2D, 3 for 3D"), cll::init(2));
static cll::opt<int> ndivOpt("n", cll::desc("number of times the mesh should be subdivided"), cll::init(0));
static cll::opt<double> simEndTimeOpt("e", cll::desc("simulation end time"), cll::init(1.0));


static const char* name = "Asynchronous Variational Integrators";
static const char* desc = "Performs elasto-dynamic simulation of a mesh with minimal number of simulation updates";
static const char* url = "asynchronous_variational_integrators";

/**
 * Common functionality for different versions and algorithms
 */
class AVIabstractMain {
private:
  // TODO: add support for verifying from a file
  struct InputConfig {

    std::string fileName;
    int spDim;
    int ndiv;
    double simEndTime;
    std::string verifile;
    std::string wltype;

    InputConfig (const std::string& fileName, int spDim, int ndiv, double simEndTime, const std::string& verifile, std::string w)
      :fileName (fileName), spDim (spDim), ndiv (ndiv), simEndTime (simEndTime), verifile (verifile), wltype(w) {
    }
  };

private:
  static const std::string getUsage ();

  static InputConfig readCmdLine ();

  static MeshInit* initMesh (const InputConfig& input);

  static void initGlobalVec (const MeshInit& meshInit, GlobalVec& g);



protected:
  using PerThrdLocalVec = galois::substrate::PerThreadStorage<LocalVec>;
  using AtomicInteger =  galois::GAtomic<int>;
  using IterCounter = galois::GAccumulator<size_t>;

  static const int DEFAULT_CHUNK_SIZE = 16;
  using AVIWorkList =  galois::worklists::PerSocketChunkFIFO<DEFAULT_CHUNK_SIZE>;


  std::string wltype;

  /** version name */
  virtual const std::string getVersion () const = 0;

  /**
   * To be implemented by derived classes for some type specific initialization
   * e.g. unordered needs element adjacency graph
   * while ordered needs a lock per node of the original mesh.
   * @param meshInit
   * @param g
   */
  virtual void initRemaining (const MeshInit& meshInit, const GlobalVec& g) = 0;


public:

  /**
   *
   * @param meshInit
   * @param g
   * @param createSyncFiles
   */
  virtual void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) = 0;

  /**
   * The main method to call 
   * @param argc
   * @param argv
   */
  void run (int argc, char* argv[]);

  void verify (const InputConfig& input, const MeshInit& meshInit, const GlobalVec& g) const;

  /**
   * Code common to loop body of different versions
   * Performs the updates to avi parameter
   *
   * @param avi
   * @param meshInit
   * @param g
   * @param l
   * @param createSyncFiles
   */
  GALOIS_ATTRIBUTE_PROF_NOINLINE static void simulate (AVI* avi, MeshInit& meshInit,
        GlobalVec& g, LocalVec& l, bool createSyncFiles);

  virtual ~AVIabstractMain() { }
};

/**
 * Serial ordered AVI algorithm
 */
class AVIorderedSerial: public AVIabstractMain {

protected:
  virtual const std::string getVersion () const {
    return std::string ("Serial");
  }

  virtual void initRemaining (const MeshInit& meshInit, const GlobalVec& g) {
    // Nothing to do, so far
  }

public:
  virtual void runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles);
};

AVIabstractMain::InputConfig AVIabstractMain::readCmdLine () {
  const char* fileName = fileNameOpt.c_str();
  int spDim = spDimOpt;
  int ndiv = ndivOpt;
  double simEndTime = simEndTimeOpt;
  std::string wltype;

  return InputConfig (fileName, spDim, ndiv, simEndTime, "", wltype);
}

MeshInit* AVIabstractMain::initMesh (const AVIabstractMain::InputConfig& input) {
  MeshInit* meshInit = NULL;

  if (input.spDim == 2) {
    meshInit = new TriMeshInit (input.simEndTime);
  }
  else if (input.spDim == 3) {
    meshInit = new TetMeshInit (input.simEndTime);
  }
  else {
    std::cerr << "ERROR: Wrong spatical dimensionality, run with -help" << std::endl;
    std::cerr << spDimOpt.HelpStr << std::endl;
    std::abort ();
  }

  // read in the mesh from file and setup the mesh, bc etc
  meshInit->initializeMesh (input.fileName, input.ndiv);

  return meshInit;
}

void AVIabstractMain::initGlobalVec (const MeshInit& meshInit, GlobalVec& g) {
  if (meshInit.isWave ()) {
    meshInit.setupVelocities (g.vecV);
    meshInit.setupVelocities (g.vecV_b);
  }
  else {
    meshInit.setupDisplacements (g.vecQ);
  }
}

void AVIabstractMain::run (int argc, char* argv[]) {
  galois::StatManager sm;
  LonestarStart(argc, argv, name, desc, url);

  // print messages e.g. version, input etc.
  InputConfig input = readCmdLine ();
  wltype = input.wltype;

  MeshInit* meshInit = initMesh (input);

  GlobalVec g(meshInit->getTotalNumDof ());

  const std::vector<AVI*>& aviList = meshInit->getAVIVec ();
  for (size_t i = 0; i < aviList.size (); ++i) {
    assert (aviList[i]->getOperation ().getFields ().size () == meshInit->getSpatialDim());
  }


  initGlobalVec (*meshInit, g);


  // derived classes may have some data to initialize before running the loop
  initRemaining (*meshInit, g);



  printf ("AVI %s version\n", getVersion ().c_str ());
  printf ("input mesh: %d elements, %d nodes\n", meshInit->getNumElements (), meshInit->getNumNodes ());

  galois::preAlloc (64*galois::getActiveThreads ());

  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer t;
  t.start ();

  // don't write to files when measuring time
  runLoop (*meshInit, g, false);

  t.stop ();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    verify (input, *meshInit, g);
  }

  delete meshInit;

}


void AVIabstractMain::verify (const InputConfig& input, const MeshInit& meshInit, const GlobalVec& g) const {

  if (input.verifile == ("")) {
    AVIorderedSerial* serial = new AVIorderedSerial ();

    MeshInit* serialMesh = initMesh (input);

    GlobalVec sg(serialMesh->getTotalNumDof ());

    initGlobalVec (*serialMesh, sg);

    //  do write to sync files when verifying
    serial->runLoop (*serialMesh, sg, true);

    // compare the global vectors for equality (within some tolerance)
    bool gvecCmp = g.cmpState (sg);

    // compare the final state of avi elements in the mesh
    bool aviCmp = meshInit.cmpState (*serialMesh);

    if (!gvecCmp || !aviCmp) {
      g.printDiff (sg);

      meshInit.printDiff (*serialMesh);

      std::cerr << "BAD: results don't match against Serial" << std::endl;
      abort ();
    }

    std::cout << ">>> OK: result verified against serial" << std::endl;

    delete serialMesh;
    delete serial;

  }
  else {
    std::cerr << "TODO: cmp against file data needs implementation" << std::endl;
    abort ();
  }
}


GALOIS_ATTRIBUTE_PROF_NOINLINE void AVIabstractMain::simulate (AVI* avi, MeshInit& meshInit,
    GlobalVec& g, LocalVec& l, bool createSyncFiles) {

  if (createSyncFiles) {
    meshInit.writeSync (*avi, g.vecQ, g.vecV_b, g.vecT);
  }

  const LocalToGlobalMap& l2gMap = meshInit.getLocalToGlobalMap();

  avi->gather (l2gMap, g.vecQ, g.vecV, g.vecV_b, g.vecT,
      l.q, l.v, l.vb, l.ti);

  avi->computeLocalTvec (l.tnew);

  if (avi->getTimeStamp () == 0.0) {
    avi->vbInit (l.q, l.v, l.vb, l.ti, l.tnew,
        l.qnew, l.vbinit,
        l.forcefield, l.funcval, l.deltaV);
    avi->update (l.q, l.v, l.vbinit, l.ti, l.tnew,
        l.qnew, l.vnew, l.vbnew,
        l.forcefield, l.funcval, l.deltaV);
  }
  else {
    avi->update (l.q, l.v, l.vb, l.ti, l.tnew,
        l.qnew, l.vnew, l.vbnew,
        l.forcefield, l.funcval, l.deltaV);
  }

  avi->incTimeStamp ();

  avi->assemble (l2gMap, l.qnew, l.vnew, l.vbnew, l.tnew, g.vecQ, g.vecV, g.vecV_b, g.vecT, g.vecLUpdate);
}

void AVIorderedSerial::runLoop (MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {

  typedef std::priority_queue<AVI*, std::vector<AVI*>, AVIReverseComparator> PQ;
  // typedef std::set<AVI*, AVIComparator> PQ;

  // temporary matrices
  int nrows = meshInit.getSpatialDim ();
  int ncols = meshInit.getNodesPerElem ();

  LocalVec l(nrows, ncols);

  const std::vector<AVI*>& aviList = meshInit.getAVIVec ();

  for (size_t i = 0; i < aviList.size (); ++i) {
    assert (aviList[i]->getOperation ().getFields ().size () == meshInit.getSpatialDim());
  }


  PQ pq;
  for (std::vector<AVI*>::const_iterator i = aviList.begin (), e = aviList.end (); i != e; ++i) {
    pq.push (*i);
    // pq.insert (*i);
  }


  int iter = 0;
  while (!pq.empty ()) {

    AVI* avi = pq.top (); pq.pop ();
    // AVI* avi = *pq.begin (); pq.erase (pq.begin ());

    assert (avi != NULL);

    AVIabstractMain::simulate (avi, meshInit, g, l, createSyncFiles);


    if (avi->getNextTimeStamp () < meshInit.getSimEndTime ()) {
      pq.push (avi);
      // pq.insert (avi);
    }

    ++iter;
  }


  // printf ("iterations = %d, time taken (in ms) = %d, average time per iter = %g\n", iter, time, ((double)time)/iter);
  printf ("iterations = %d\n", iter);

}
#endif // AVI_ABSTRACT_MAIN_H_ 

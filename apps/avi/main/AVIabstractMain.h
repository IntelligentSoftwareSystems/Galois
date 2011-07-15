/** Common code for different AVI algorithms -*- C++ -*-
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



#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* fileNameOpt = "-f";
static const char* spDimOpt = "-d";
static const char* ndivOpt = "-n";
static const char* simEndTimeOpt = "-e";


static const char* name = "Asynchronous Variational Integrators";
static const char* description = "Elasto-dynamic simulation of a mesh with minimal number of simulation updates";
static const char* url = "http://iss.ices.utexas.edu/lonestar/asynchronous_variational_integrators";

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

    InputConfig (const std::string& fileName, int spDim, int ndiv, double simEndTime, const std::string& verifile)
      :fileName (fileName), spDim (spDim), ndiv (ndiv), simEndTime (simEndTime), verifile (verifile) {
    }
  };

private:
  static const std::string getUsage ();

  static void printUsage ();

  static InputConfig readCmdLine (std::vector<const char*> args);

  static MeshInit* initMesh (const InputConfig& input);

  static void initGlobalVec (const MeshInit& meshInit, GlobalVec& g);

protected:


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
  void run (int argc, const char* argv[]);

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
  inline static void simulate (AVI* avi, MeshInit& meshInit,
        GlobalVec& g, LocalVec& l, bool createSyncFiles);
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

const std::string AVIabstractMain::getUsage () {
  std::stringstream ss;
  ss << fileNameOpt << " fileName.neu " << spDimOpt << " spDim " << ndivOpt << " ndiv " 
    << simEndTimeOpt << " simEndTime" << std::endl;
  return ss.str ();

}

void AVIabstractMain::printUsage () {
  fprintf (stderr, "%s\n", getUsage ().c_str ());
  abort ();
}

AVIabstractMain::InputConfig AVIabstractMain::readCmdLine (std::vector<const char*> args) {
  const char* fileName = NULL;
  int spDim = 2;
  int ndiv = 0;
  double simEndTime = 1.0;

  if (args.size() == 0) {
    printUsage ();
  }

  for (std::vector<const char*>::const_iterator i = args.begin (), e = args.end (); i != e; ++i) {

    if (std::string (*i) == fileNameOpt) {
      ++i;
      fileName = *i;

    } else if (std::string (*i) == spDimOpt) {
      ++i;
      spDim = atoi (*i);

    } else if (std::string (*i) == ndivOpt) {
      ++i;
      ndiv = atoi (*i);

    } else if (std::string (*i) == simEndTimeOpt) {
      ++i;
      simEndTime = atof (*i);
    } else {
      fprintf (stderr, "Unkown option: %s\n Exiting ...\n", *i);
      printUsage ();
    }
  }



  return InputConfig (fileName, spDim, ndiv, simEndTime, "");
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
    printUsage ();
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

void AVIabstractMain::run (int argc, const char* argv[]) {

  std::vector<const char*> args = parse_command_line (argc, argv, getUsage ().c_str ());

  printBanner (std::cout, name, description, url);

  // print messages e.g. version, input etc.
  InputConfig input = readCmdLine (args);

  MeshInit* meshInit = initMesh (input);

  GlobalVec g(meshInit->getTotalNumDof ());

  const std::vector<AVI*>& aviList = meshInit->getAVIVec ();
  for (size_t i = 0; i < aviList.size (); ++i) {
    assert (aviList[i]->getOperation ().getFields ().size () == meshInit->getSpatialDim());
  }


  initGlobalVec (*meshInit, g);


  // derived classes may have some data to initialze before running the loop
  initRemaining (*meshInit, g);



  printf ("PAVI %s version\n", getVersion ().c_str ());
  printf ("input mesh: %d elements, %d nodes\n", meshInit->getNumElements (), meshInit->getNumNodes ());


  Galois::StatTimer t;
  t.start ();

  // don't write to files when measuring time
  runLoop (*meshInit, g, false);

  t.stop ();

  verify (input, *meshInit, g);

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


void AVIabstractMain::simulate (AVI* avi, MeshInit& meshInit,
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
  // temporary matrices
  int nrows = meshInit.getSpatialDim ();
  int ncols = meshInit.getNodesPerElem ();

  LocalVec l(nrows, ncols);

  const std::vector<AVI*>& aviList = meshInit.getAVIVec ();

  for (size_t i = 0; i < aviList.size (); ++i) {
    assert (aviList[i]->getOperation ().getFields ().size () == meshInit.getSpatialDim());
  }


  std::priority_queue<AVI*, std::vector<AVI*>, AVIReverseComparator> pq;
  for (std::vector<AVI*>::const_iterator i = aviList.begin (), e = aviList.end (); i != e; ++i) {
    pq.push (*i);
  }


  int iter = 0;
  while (!pq.empty ()) {

    AVI* avi = pq.top ();
    pq.pop ();

    assert (avi != NULL);

    AVIabstractMain::simulate (avi, meshInit, g, l, createSyncFiles);


    if (avi->getNextTimeStamp () <= meshInit.getSimEndTime ()) {
      pq.push (avi);
    }

    ++iter;
  }


  // printf ("iterations = %d, time taken (in ms) = %d, average time per iter = %g\n", iter, time, ((double)time)/iter);
  printf ("iterations = %d\n", iter);

}
#endif // AVI_ABSTRACT_MAIN_H_ 

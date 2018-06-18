/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef MESHINIT_H_
#define MESHINIT_H_

#include <vector>
#include <string>
#include <algorithm>
#include <exception>

#include <cassert>
#include <cmath>
#include <cstdio>

#include <boost/noncopyable.hpp>

#include "AuxDefs.h"
#include "StandardAVI.h"
#include "Element.h"
#include "Femap.h"
#include "Material.h"
#include "CoordConn.h"
#include "TriLinearCoordConn.h"
#include "TetLinearCoordConn.h"

class MeshInit : private boost::noncopyable {

public:
  typedef StandardAVI::BCFunc BCFunc;
  typedef StandardAVI::BCImposedType BCImposedType;

private:
  static const double RHO;
  static const double MU;
  static const double LAMBDA;
  static const int PID;

  static const double DELTA;
  static const double T_INIT;

  // length of filenames
  static const size_t MAX_FNAME;

private:
  double simEndTime;
  bool wave;

  //! to be freed
  LocalToGlobalMap* l2gMap;
  CoordConn* cc;
  SimpleMaterial* ile;

  //! vectors to keep track of all the memory
  std::vector<ElementGeometry*> geomVec;
  std::vector<Element*> elemVec;
  std::vector<Residue*> massResidueVec;
  std::vector<DResidue*> operationsVec;
  std::vector<AVI*> aviVec;

  VecSize_t fieldsUsed;

  double writeInc;
  int writeInterval;
  VecSize_t aviWriteInterval;
  FILE* syncFileWriter;

private:
  void stretchInternal(VecDouble& dispOrVel, bool isVel) const;
  void getBCs(const Element& e,
              std::vector<std::vector<BCImposedType>>& itypeMat,
              std::vector<BCFunc>& bcfuncVec) const;

  static bool computeDiffAVI(std::vector<AVI*> listA, std::vector<AVI*> listB,
                             bool printDiff);

  template <typename T>
  static void destroyVecOfPtr(std::vector<T*>& vec) {
    for (typename std::vector<T*>::iterator i = vec.begin(), ei = vec.end();
         i != ei; ++i) {

      delete *i;
      *i = NULL;
    }
  }

  void destroy() {

    destroyVecOfPtr(geomVec);
    destroyVecOfPtr(elemVec);
    destroyVecOfPtr(massResidueVec);
    destroyVecOfPtr(operationsVec);
    destroyVecOfPtr(aviVec);

    delete l2gMap;
    delete cc;
    delete ile;
  }

public:
  /**
   *
   * @param simEndTime
   * @param wave
   */
  MeshInit(double simEndTime, bool wave) : simEndTime(simEndTime), wave(wave) {

    // TODO: writeInc should depend on simEndTime and
    // number of intervals intended
    if (wave) {
      this->writeInc = 0.005; // XXX: from testPAVI2D
    } else {
      this->writeInc = 0.1;
    }

    writeInterval  = 0;
    syncFileWriter = NULL;
  }

  virtual ~MeshInit() { destroy(); }

  /**
   *
   * main function to call after creating an instance. This
   * initializes all the data structures by reading this file
   *
   * @param fileName
   * @param ndiv: number of times the mesh (read in from the file) should be
   * subdivided
   */
  void initializeMesh(const std::string& fileName, int ndiv);

  virtual size_t getSpatialDim() const = 0;

  virtual size_t getNodesPerElem() const = 0;

  bool isWave() const { return wave; }

  double getSimEndTime() const { return simEndTime; }

  //! Number of elements in the mesh
  int getNumElements() const { return cc->getNumElements(); }

  //! number of nodes (vertices) in the mesh
  int getNumNodes() const { return cc->getNumNodes(); }

  //! number of nodes times the dimensionality
  unsigned int getTotalNumDof() const { return l2gMap->getTotalNumDof(); }

  const std::vector<AVI*>& getAVIVec() const { return aviVec; }

  //! mapping function from local per element vectors (for target functions) to
  //! global vectors this tells what indices in the global vector each element
  //! contributes to

  const LocalToGlobalMap& getLocalToGlobalMap() const { return *l2gMap; }

  /**
   * setup initial conditions
   * to be called before starting the simulation loop
   *
   * @param disp: global dispalcements vector
   */
  void setupDisplacements(VecDouble& disp) const {
    stretchInternal(disp, false);
  }

  /**
   * setup initial conditions
   * to be called before starting the simulation loop
   *
   * @param vel: global velocities vector
   */
  void setupVelocities(VecDouble& vel) const { stretchInternal(vel, true); }

  /**
   * Write the values in global vectors corresponding to this avi element
   * to a file at regular intervals
   *
   * @param avi
   * @param Qval
   * @param Vbval
   * @param Tval
   */
  void writeSync(const AVI& avi, const VecDouble& Qval, const VecDouble& Vbval,
                 const VecDouble& Tval);

  /**
   * Compare state of avi vector against other object
   * Use for verification between different versions
   *
   * @param that
   */
  bool cmpState(const MeshInit& that) const {
    return computeDiffAVI(this->aviVec, that.aviVec, false);
  }

  /**
   * Compare state of avi vector against other object
   * Use for verification between different versions
   * and also print out the differences
   *
   * @param that
   */
  void printDiff(const MeshInit& that) const {
    computeDiffAVI(this->aviVec, that.aviVec, true);
  }

  void writeMeshCenters(const char* outFileName = "mesh-centers.csv") const;

  void writeMesh(const char* polyFileName  = "mesh-poly.csv",
                 const char* coordFileName = "mesh-coord.csv") const;

protected:
  //! functions to compute boundary condtions
  //! @param coord
  virtual BCFunc getBCFunc(const double* coord) const = 0;

  //! returns the correct derived type of CoordConn
  virtual CoordConn* makeCoordConn() const = 0;

  //! parametric node numbering of an element (triangle or tetrahedron)
  virtual const double* getParam() const = 0;

  //! internal function used by @see setupDisplacements
  //! @param coord
  //! @param f
  virtual double initDisp(const double* coord, int f) const = 0;

  //! internal function used by @see setupVelocities
  //! @param coord
  //! @param f
  virtual double initVel(const double* coord, int f) const = 0;

  //! number of fields often the same as dimensionality
  virtual size_t numFields() const = 0;
};

class TriMeshInit : public MeshInit {
private:
  static double topBC(int f, int a, double t) {
    if (f == 0) {
      return (0.1 * cos(t));
    } else {
      return (0.0);
    }
  }

  static double botBC(int f, int a, double t) { return (0.0); }

public:
  static const double PARAM[];

  TriMeshInit(double simEndTime, bool wave = false)
      : MeshInit(simEndTime, wave) {}

  virtual size_t getSpatialDim() const { return TriLinearTraits::SPD; }
  virtual size_t getNodesPerElem() const {
    return TriLinearTraits::NODES_PER_ELEM;
  }

protected:
  virtual CoordConn* makeCoordConn() const { return new TriLinearCoordConn(); }

  virtual const double* getParam() const { return PARAM; }

  virtual size_t numFields() const { return TriLinearTraits::NFIELDS; }

  virtual BCFunc getBCFunc(const double* coord) const {
    if (coord[0] == 0.0) {
      return botBC;
    } else if (coord[0] == 10.0) {
      return topBC;
    } else {
      return NULL;
    }
  }

  virtual double initDisp(const double* coord, int f) const {
    double stretch;
    if (f == 0) {
      // XXX: some weird code???
      stretch = coord[0] * 0.2 - 1.0;
      stretch = coord[0] * 0.01 - 0.05;
    } else {
      stretch = coord[1] * 0.2 - 1.0;
      stretch = coord[1] * 0.01 - 0.05;
    }

    return stretch;
  }

  virtual double initVel(const double* coord, int f) const {
    if (coord[0] == 0.0) {
      return 0.1;
    } else {
      return 0.0;
    }
  }
};

class TetMeshInit : public MeshInit {
private:
  static double topBC(int f, int a, double t) {
    if (f == 2) {
      return (0.1 * sin(t));
    } else {
      return (0.0);
    }
  }

  static double botBC(int f, int a, double t) { return (0.0); }

public:
  static const double PARAM[];

  TetMeshInit(double simEndTime, bool wave = false)
      : MeshInit(simEndTime, wave) {}

  virtual size_t getSpatialDim() const { return TetLinearTraits::SPD; }
  virtual size_t getNodesPerElem() const {
    return TetLinearTraits::NODES_PER_ELEM;
  }

protected:
  virtual CoordConn* makeCoordConn() const { return new TetLinearCoordConn(); }

  virtual const double* getParam() const { return PARAM; }

  virtual size_t numFields() const { return TetLinearTraits::NFIELDS; }

  virtual BCFunc getBCFunc(const double* coord) const {
    if (fabs(coord[0]) < 0.01) {
      return botBC;
    } else if (fabs(coord[0] - 5.0) < 0.01) {
      return topBC;
    } else {
      return NULL;
    }
  }

  virtual double initDisp(const double* coord, int f) const {
    double stretch = coord[f] * 0.10 - 0.25;
    return stretch;
  }

  virtual double initVel(const double* coord, int f) const { return 0.0; }
};

#endif /* MESHINIT_H_ */

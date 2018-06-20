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

#include "MeshInit.h"

const double TriMeshInit::PARAM[] = {1, 0, 0, 1, 0, 0};
const double TetMeshInit::PARAM[] = {1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};

const double MeshInit::RHO    = 1.0;
const double MeshInit::MU     = 0.5;
const double MeshInit::LAMBDA = 0.0;
const int MeshInit::PID       = 0;

const double MeshInit::DELTA  = 0.1;
const double MeshInit::T_INIT = 0.0;

const size_t MeshInit::MAX_FNAME = 1024;

/**
 *
 * @param fileName
 * @param ndiv
 *          number of times to subdivide the initial mesh
 */
void MeshInit::initializeMesh(const std::string& fileName, int ndiv) {

  FemapInput input(fileName.c_str());

  this->cc = makeCoordConn();

  cc->initFromFileData(input);

  for (int i = 0; i < ndiv; ++i) {
    cc->subdivide();
  }

  fieldsUsed.resize(numFields());
  // fileds numbered from 0..N-1
  for (size_t i = 0; i < fieldsUsed.size(); ++i) {
    fieldsUsed[i] = i;
  }

  geomVec.clear();
  elemVec.clear();

  for (size_t i = 0; i < cc->getNumElements(); ++i) {

    Element* elem = cc->makeElem(i);

    elemVec.push_back(elem);
    geomVec.push_back(const_cast<ElementGeometry*>(&(elem->getGeometry())));
  }

  this->l2gMap = new StandardP1nDMap(elemVec);

  this->ile = new NeoHookean(LAMBDA, MU, RHO);

  massResidueVec.clear();
  operationsVec.clear();

  for (std::vector<Element*>::const_iterator i = elemVec.begin();
       i != elemVec.end(); ++i) {
    Residue* m = new DiagonalMassForSW(*(*i), *ile, fieldsUsed);
    massResidueVec.push_back(m);

    DResidue* sw = new StressWork(*(*i), *ile, fieldsUsed);
    operationsVec.push_back(sw);
  }

  size_t totalDof = l2gMap->getTotalNumDof();

  VecDouble massVec(totalDof, 0.0);

  VecDouble dofArray(totalDof, 0.0);

  Residue::assemble(massResidueVec, *l2gMap, dofArray, massVec);

  aviVec.clear();

  for (size_t i = 0; i < elemVec.size(); ++i) {
    const Element* e = elemVec[i];

    std::vector<BCFunc> bcfuncVec(getNodesPerElem());

    std::vector<std::vector<StandardAVI::BCImposedType>> itypeMat(
        getSpatialDim(), std::vector<StandardAVI::BCImposedType>(
                             getNodesPerElem(), StandardAVI::ZERO));

    getBCs(*e, itypeMat, bcfuncVec);

    // TODO: AVI init broken here
    AVI* avi = new StandardAVI(*l2gMap, *operationsVec[i], massVec, i, itypeMat,
                               bcfuncVec, DELTA, T_INIT);

    this->aviVec.push_back(avi);

    assert(operationsVec[i]->getFields().size() == fieldsUsed.size());
  }

  this->aviWriteInterval = VecSize_t(getNumElements(), 0);
}

void MeshInit::getBCs(const Element& e,
                      std::vector<std::vector<BCImposedType>>& itypeMat,
                      std::vector<BCFunc>& bcfuncVec) const {

  const double* param = getParam();
  double* coord       = new double[this->getSpatialDim()];

  for (size_t a = 0; a < getNodesPerElem(); ++a) {
    e.getGeometry().map(param + this->getSpatialDim() * a, coord);

    BCFunc bc = getBCFunc(coord);

    BCImposedType itypeVal = StandardAVI::ZERO;
    if (bc == NULL) {
      itypeVal = StandardAVI::ZERO;
    } else {
      itypeVal = StandardAVI::ONE;
    }

    bcfuncVec[a] = bc; // XXX: sometimes BCFunc is NULL

    for (size_t i = 0; i < this->getSpatialDim(); ++i) {
      itypeMat[i][a] = itypeVal;
    }
  }

  delete[] coord;
}

/**
 *
 * @param avi
 * @param Qval
 *          displacement
 * @param Vbval
 *          velocity
 * @param Tval
 *          time
 */
void MeshInit::writeSync(const AVI& avi, const VecDouble& Qval,
                         const VecDouble& Vbval, const VecDouble& Tval) {
  // end time of the write interval for element 'avi'
  double interEnd =
      (this->aviWriteInterval[avi.getGlobalIndex()] * this->writeInc);

  // when the first time update time of 'a' goes past the current write
  // interval
  // we dump some state into a file
  if (avi.getNextTimeStamp() > interEnd) {
    ++this->aviWriteInterval[avi.getGlobalIndex()];

    // if 'a' is the first to enter a new write interval
    // then open a new file
    // and also close the old file
    if (avi.getNextTimeStamp() > (this->writeInterval * this->writeInc)) {

      if (syncFileWriter != NULL) {
        // will be true after the first interval
        fclose(syncFileWriter);
        syncFileWriter = NULL; // being defensive ...

        printf("myid = %d, done with syncfiles for interval = %d, simulation "
               "time for interval = %g\n",
               PID, (this->writeInterval - 1),
               (this->writeInterval * this->writeInc));
        // TODO: measure
        // syncfile writing
        // time per interval
      }

      char syncFileName[MAX_FNAME];
      sprintf(syncFileName, "sync.%d_%d.dat", this->writeInterval, PID);

      this->syncFileWriter = fopen(syncFileName, "w");

      if (syncFileWriter == NULL) {
        std::cerr << "Failed to open log file for writing: " << syncFileName
                  << std::endl;
        abort();
      }

      // increment to define the end limit for the new interval.
      ++this->writeInterval;
    }

    assert(this->syncFileWriter != NULL);

    const VecSize_t& conn = avi.getGeometry().getConnectivity();

    for (size_t aa = 0; aa < conn.size(); ++aa) {
      GlobalNodalIndex nodeNum = conn[aa];

      fprintf(syncFileWriter, "%zd %zd ", avi.getGlobalIndex(), nodeNum);

      int idx = -1;
      for (size_t f = 0; f < avi.getGeometry().getEmbeddingDimension(); ++f) {
        idx = l2gMap->map(f, aa, avi.getGlobalIndex());

        // XXX: commented out and printing vector values instead
        // double pos = Qval[idx] + Vbval[idx] *
        // (aviWriteInterval[avi.getGlobalIndex ()] * this->writeInc -
        // Tval[idx]); fprintf (syncFileWriter, "%12.10f ", pos);
        fprintf(syncFileWriter, "%12.10f ", Qval[idx]);

        fprintf(syncFileWriter, "%12.10f ", Vbval[idx]);
      }

      fprintf(syncFileWriter, "%12.10f \n", Tval[idx]);
    }
  }
}

void MeshInit::stretchInternal(VecDouble& dispOrVel, bool isVel) const {
  // int localsize = this->getSpatialDim () * this->getNodesPerElem () *
  // this->elemVec.size ();

  double* coord = new double[this->getSpatialDim()];
  double stretch;

  const double* param = getParam();

  for (size_t e = 0, i = 0; e < this->elemVec.size(); ++e) {
    for (size_t f = 0; f < this->numFields(); ++f) {
      for (size_t a = 0; a < this->getNodesPerElem(); ++a, ++i) {

        elemVec[e]->getGeometry().map(param + this->numFields() * a, coord);
        if (isVel) {
          stretch = this->initVel(coord, f);
        } else {
          stretch = this->initDisp(coord, f);
        }

        size_t index     = l2gMap->map(f, a, e);
        dispOrVel[index] = stretch;
      }
    }
  }

  delete[] coord;
}

// makes a copy of the arguments to sort them etc.
bool MeshInit::computeDiffAVI(std::vector<AVI*> listA, std::vector<AVI*> listB,
                              bool printDiff) {
  bool result = false;

  const char* nameA = "this->aviList";
  const char* nameB = "that.aviList";

  if (listA.size() != listB.size()) {
    result = false;
    if (printDiff) {
      fprintf(stderr,
              "Comparing lists of different sizes, %s.size() = %zd, %s.size() "
              "= %zd\n",
              nameA, listA.size(), nameB, listB.size());
    }
  } else {
    // sort in increasing order of next update time
    AVIComparator aviCmp;

    std::sort(listA.begin(), listA.end(), aviCmp);
    std::sort(listB.begin(), listB.end(), aviCmp);

    result = true;
    for (size_t i = 0; i < listA.size(); ++i) {
      const AVI& aviA = *listA[i];
      const AVI& aviB = *listB[i];

      double diff = fabs(aviA.getTimeStamp() - aviB.getTimeStamp());
      if (diff > TOLERANCE) {
        result = false;
        if (printDiff) {
          fprintf(stderr,
                  "(%s[%zd] = (id=%zd,time:%g)) != (%s[%zd]= (id=%zd,time=%g)) "
                  " diff=%g\n",
                  nameA, i, aviA.getGlobalIndex(), aviA.getTimeStamp(), nameB,
                  i, aviB.getGlobalIndex(), aviB.getTimeStamp(), diff);
        } else {
          break; // no use continuing on if not printing
        }
      }
    } // end for
  }

  return result;
}

void MeshInit::writeMeshCenters(const char* outFileName) const {

  if (getSpatialDim() != 2) {
    std::cerr << "Mesh plotting implemented for 2D elements only" << std::endl;
    abort();
  }

  FILE* plotFile = fopen(outFileName, "w");

  if (plotFile == NULL) {
    abort();
  }

  VecDouble center(getSpatialDim(), 0);

  fprintf(plotFile, "center_x, center_y, timestamp\n");
  for (std::vector<AVI*>::const_iterator i  = getAVIVec().begin(),
                                         ei = getAVIVec().end();
       i != ei; ++i) {

    const AVI& avi = **i;

    std::fill(center.begin(), center.end(), 0.0);
    avi.getElement().getGeometry().computeCenter(center);

    fprintf(plotFile, "%g, %g, %g\n", center[0], center[1],
            avi.getNextTimeStamp());
  }

  fclose(plotFile);
}

void MeshInit::writeMesh(const char* polyFileName,
                         const char* coordFileName) const {

  FILE* polyFile = fopen(polyFileName, "w");

  if (polyFile == NULL) {
    abort();
  }

  for (size_t i = 0; i < cc->getNodesPerElem(); ++i) {
    fprintf(polyFile, "node%zd, ", i);
  }
  fprintf(polyFile, "timestamp\n");

  for (std::vector<AVI*>::const_iterator i  = getAVIVec().begin(),
                                         ei = getAVIVec().end();
       i != ei; ++i) {

    const AVI& avi = **i;

    const VecSize_t& conn = avi.getElement().getGeometry().getConnectivity();

    for (size_t j = 0; j < conn.size(); ++j) {
      fprintf(polyFile, "%zd, ", conn[j]);
    }

    fprintf(polyFile, "%g\n", avi.getNextTimeStamp());
  }

  fclose(polyFile);

  FILE* coordFile = fopen(coordFileName, "w");

  if (coordFile == NULL) {
    abort();
  }

  for (size_t i = 0; i < cc->getNodesPerElem(); ++i) {
    fprintf(coordFile, "dim%zd", i);

    if (i < cc->getNodesPerElem() - 1) {
      fprintf(coordFile, ", ");
    } else {
      fprintf(coordFile, "\n");
    }
  }

  const VecDouble& coord = cc->getCoordinates();
  for (size_t i = 0; i < coord.size(); i += cc->getSpatialDim()) {
    for (size_t j = i; j < i + cc->getSpatialDim(); ++j) {
      if (j != i) { // not first iter
        fprintf(coordFile, ", ");
      }
      fprintf(coordFile, "%g", coord[j]);
    }

    fprintf(coordFile, "\n");
  }

  fclose(coordFile);
}

/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef CLUSTER_LIGHT_H
#define CLUSTER_LIGHT_H

#include "LeafLight.h"
#include "NodeWrapper.h"
#include "galois/gstl.h"
#include <assert.h>

class ClusterLight : public AbstractLight {
private:
  AbstractLight* m_left;
  AbstractLight* m_right;
  Point3 boxRadius;

public:
  ClusterLight() : boxRadius(0), coneDirection(0) {}

  // TODO: remove
  /*
  void setBox(double minX, double maxX, double minY, double maxY, double minZ,
              double maxZ) {
    myLoc.set(0.5f * (minX + maxX), 0.5f * (minY + maxY), 0.5f * (minZ + maxZ));
    boxRadius.set(0.5f * (maxX - minX), 0.5f * (maxY - minY),
                  0.5f * (maxZ - minZ));
  }
  */
  void setBox(Point3& min, Point3& max) {
    myLoc.set(min);
    myLoc.add(max);
    myLoc.scale(0.5);
    boxRadius.set(max);
    boxRadius.sub(min);
    boxRadius.scale(0.5);
  }

  void setChildren(AbstractLight* inLeft, AbstractLight* inRight,
                   double repRandomNum) {
    m_left  = inLeft;
    m_right = inRight;
    setSummedIntensity(*m_left, *m_right);
    // setCombinedFlags(m_left, m_right);
    // we only apply clamping to nodes that are low in the tree
    std::vector<double>& ranVec =
        repRandomNums[(int)(repRandomNum * numRepRandomNums)];
    if (globalMultitime) {
      assert(false && "Should  not have time true!");
      std::abort();
      //      int numReps = endTime - startTime + 1;
      //      if (reps == null || reps.length < numReps) {
      //        reps = new LeafLight[numReps];
      //      } else {
      //        for (int j = numReps; j < reps.length; j++) {
      //          reps[j] = null;
      //        } //fill unused values will nulls
      //      }
      //      if (m_left.isLeaf()) {
      //        LeafLight leftLeaf = (LeafLight) m_left;
      //        if (m_right.isLeaf()) {
      //          chooseRepsWithTime(reps, this, ranVec, leftLeaf, (LeafLight)
      //          m_right);
      //        } else {
      //          chooseRepsWithTime(reps, this, ranVec, (ClusterLight)
      //          m_right, leftLeaf); //note: operation is symmectric so we
      //          just interchange the children in the call
      //        }
      //      } else {
      //        ClusterLight leftClus = (ClusterLight) m_left;
      //        if (m_right.isLeaf()) {
      //          chooseRepsWithTime(reps, this, ranVec, leftClus, (LeafLight)
      //          m_right);
      //        } else {
      //          chooseRepsWithTime(reps, this, ranVec, leftClus,
      //          (ClusterLight) m_right);
      //        }
      //      }
    } else {
      if (reps.size() == 0 || reps.size() != (unsigned int)globalNumReps) {
        reps.clear();
        reps.resize(globalNumReps);
      }
      if (m_left->isLeaf()) {
        LeafLight* leftLeaf = (LeafLight*)m_left;
        if (m_right->isLeaf()) {
          chooseRepsNoTime(reps, *this, ranVec, *leftLeaf,
                           (LeafLight&)*m_right);
        } else {
          chooseRepsNoTime(reps, *this, ranVec, (ClusterLight&)*m_right,
                           *leftLeaf); // note: operation is symmectric so we
                                       // just interchange the children in the
                                       // call
        }
      } else {
        ClusterLight* leftClus = (ClusterLight*)m_left;
        if (m_right->isLeaf()) {
          chooseRepsNoTime(reps, *this, ranVec, *leftClus,
                           (LeafLight&)*m_right);
        } else {
          chooseRepsNoTime(reps, *this, ranVec, *leftClus,
                           (ClusterLight&)*m_right);
        }
      }
    }
  }

  template <typename V1, typename V2>
  static void chooseRepsNoTime(V1& repArr, AbstractLight& parent, V2& ranVec,
                               LeafLight& left, LeafLight& right) {
    double totalInten = parent.getScalarTotalIntensity();
    double leftInten  = left.getScalarTotalIntensity();
    double nextTest   = ranVec[0] * totalInten;
    for (unsigned int i = 0; i < repArr.size() - 1; i++) {
      double test = nextTest;
      nextTest    = ranVec[i + 1] * totalInten;
      repArr[i]   = (test < leftInten) ? &left : &right;
    }
    repArr[repArr.size() - 1] = (nextTest < leftInten) ? &left : &right;
  }

  template <typename V1, typename V2>
  static void chooseRepsNoTime(V1& repArr, AbstractLight& parent, V2& ranVec,
                               ClusterLight& left, LeafLight& right) {
    double totalInten = parent.getScalarTotalIntensity();
    double leftInten  = left.getScalarTotalIntensity();
    double nextTest   = ranVec[0] * totalInten;
    for (unsigned int i = 0; i < repArr.size() - 1; i++) {
      double test = nextTest;
      nextTest    = ranVec[i + 1] * totalInten;
      repArr[i]   = (test < leftInten) ? (left.reps[i]) : &right;
    }
    repArr[repArr.size() - 1] =
        (nextTest < leftInten) ? (left.reps[repArr.size() - 1]) : &right;
  }

  template <typename V1, typename V2>
  static void chooseRepsNoTime(V1& repArr, AbstractLight& parent, V2& ranVec,
                               ClusterLight& left, ClusterLight& right) {
    double totalInten = parent.getScalarTotalIntensity();
    double leftInten  = left.getScalarTotalIntensity();
    double nextTest   = ranVec[0] * totalInten;
    for (unsigned int i = 0; i < repArr.size() - 1; i++) {
      double test = nextTest;
      nextTest    = ranVec[i + 1] * totalInten;
      repArr[i]   = (test < leftInten) ? (left.reps[i]) : (right.reps[i]);
    }
    repArr[repArr.size() - 1] = (nextTest < leftInten)
                                    ? (left.reps[repArr.size() - 1])
                                    : (right.reps[repArr.size() - 1]);
  }

  void setDirectionCone(double dirX, double dirY, double dirZ,
                        double inConeCos) {
    coneDirection.set(dirX, dirY, dirZ);
    coneCos = inConeCos;
  }

  float getConeCos() { return coneCos; }

  template <typename V1, typename V2>
  void findConeDirsRecursive(V1& coordArr, V2& tempClusterArr) {
    // TODO : Fix this. NodeWrapper::CONE_RECURSE_DEPTH - 1 = 3
    findConeDirsRecursive(*m_left, coordArr, 0, tempClusterArr, 3);
    findConeDirsRecursive(*m_right, coordArr, 0, tempClusterArr, 3);
  }

  template <typename V1, typename V2>
  static int findConeDirsRecursive(AbstractLight& node, V1& fArr, int numDirs,
                                   V2& cArr, int recurseDepth) {
    if (!node.isLeaf()) {
      ClusterLight& clus = (ClusterLight&)node;
      if (clus.coneCos == 1.0) {
        numDirs =
            addConeDir(fArr, numDirs, clus.coneDirection.getX(),
                       clus.coneDirection.getY(), clus.coneDirection.getZ());
      } else if (recurseDepth <= 0) {
        // find first empty slot and add this cluster there
        for (int i = 0;; i++) {
          if (cArr[i] == NULL) {
            cArr[i] = &clus;
            if (cArr[i + 1] != NULL) {
              assert(false);
            }
            break;
          }
        }
      } else {
        numDirs = findConeDirsRecursive(*(clus.m_left), fArr, numDirs, cArr,
                                        recurseDepth - 1);
        numDirs = findConeDirsRecursive(*(clus.m_right), fArr, numDirs, cArr,
                                        recurseDepth - 1);
      }
    } else {
      LeafLight& light = (LeafLight&)node;
      numDirs = addConeDir(fArr, numDirs, light.getDirX(), light.getDirY(),
                           light.getDirZ());
    }
    return numDirs;
  }

  template <typename V1>
  static int addConeDir(V1& fArr, int numDirs, double x, double y, double z) {
    // only add direction if it does not match any existing directions
    for (int i = 0; i < 3 * numDirs; i++) {
      if ((fArr[i] == x) && (fArr[i + 1] == y) && (fArr[i + 2] == z)) {
        return numDirs;
      }
    }
    int index       = 3 * numDirs;
    fArr[index]     = x;
    fArr[index + 1] = y;
    fArr[index + 2] = z;
    return numDirs + 1;
  }

  bool isLeaf() { return false; }

  int size() {
    // only leafs are counted
    return m_left->size() + m_right->size();
  }
};

#endif /* CLUSTER_LIGHT_H */

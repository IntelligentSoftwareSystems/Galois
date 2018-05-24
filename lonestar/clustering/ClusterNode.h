/** Agglomerative Clustering -*- C++ -*-
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
 * @section Description
 *
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */
#ifndef CLUSTERNODE_H_
#define CLUSTERNODE_H_

#include "LeafNode.h"
#include "NodeWrapper.h"
#include "galois/gstl.h"
#include <assert.h>

class ClusterNode : public AbstractNode {
private:
  AbstractNode* leftChild;
  AbstractNode* rightChild;
  GVector<LeafNode*> reps;
  Point3 boxRadius;
  Point3 coneDirection;
  double coneCos;

public:
  ClusterNode() : boxRadius(0), coneDirection(0) {}
  virtual ~ClusterNode() { reps.clear(); }
  void setBox(double minX, double maxX, double minY, double maxY, double minZ,
              double maxZ) {
    myLoc.set(0.5f * (minX + maxX), 0.5f * (minY + maxY), 0.5f * (minZ + maxZ));
    boxRadius.set(0.5f * (maxX - minX), 0.5f * (maxY - minY),
                  0.5f * (maxZ - minZ));
  }
  void setBox(Point3& min, Point3& max) {
    myLoc.set(min);
    myLoc.add(max);
    myLoc.scale(0.5);
    boxRadius.set(max);
    boxRadius.sub(min);
    boxRadius.scale(0.5);
  }

  void setChildren(AbstractNode* inLeft, AbstractNode* inRight,
                   double repRandomNum) {
    leftChild  = inLeft;
    rightChild = inRight;
    setSummedIntensity(*leftChild, *rightChild);
    // setCombinedFlags(leftChild, rightChild);
    // we only apply clamping to nodes that are low in the tree
    std::vector<double>* ranVec =
        repRandomNums[(int)(repRandomNum * numRepRandomNums)];
    if (globalMultitime) {
      assert(false && "Should  not have time true!");
      //      int numReps = endTime - startTime + 1;
      //      if (reps == null || reps.length < numReps) {
      //        reps = new LeafNode[numReps];
      //      } else {
      //        for (int j = numReps; j < reps.length; j++) {
      //          reps[j] = null;
      //        } //fill unused values will nulls
      //      }
      //      if (leftChild.isLeaf()) {
      //        LeafNode leftLeaf = (LeafNode) leftChild;
      //        if (rightChild.isLeaf()) {
      //          chooseRepsWithTime(reps, this, ranVec, leftLeaf, (LeafNode)
      //          rightChild);
      //        } else {
      //          chooseRepsWithTime(reps, this, ranVec, (ClusterNode)
      //          rightChild, leftLeaf); //note: operation is symmectric so we
      //          just interchange the children in the call
      //        }
      //      } else {
      //        ClusterNode leftClus = (ClusterNode) leftChild;
      //        if (rightChild.isLeaf()) {
      //          chooseRepsWithTime(reps, this, ranVec, leftClus, (LeafNode)
      //          rightChild);
      //        } else {
      //          chooseRepsWithTime(reps, this, ranVec, leftClus, (ClusterNode)
      //          rightChild);
      //        }
      //      }
    } else {
      if (reps.size() == 0 || reps.size() != (unsigned int)globalNumReps) {
        reps.clear();
        reps.resize(globalNumReps);
      }
      if (leftChild->isLeaf()) {
        LeafNode* leftLeaf = (LeafNode*)leftChild;
        if (rightChild->isLeaf()) {
          chooseRepsNoTime(reps, *this, ranVec, *leftLeaf,
                           (LeafNode&)*rightChild);
        } else {
          chooseRepsNoTime(reps, *this, ranVec, (ClusterNode&)*rightChild,
                           *leftLeaf); // note: operation is symmectric so we
                                       // just interchange the children in the
                                       // call
        }
      } else {
        ClusterNode* leftClus = (ClusterNode*)leftChild;
        if (rightChild->isLeaf()) {
          chooseRepsNoTime(reps, *this, ranVec, *leftClus,
                           (LeafNode&)*rightChild);
        } else {
          chooseRepsNoTime(reps, *this, ranVec, *leftClus,
                           (ClusterNode&)*rightChild);
        }
      }
    }
  }

  static void chooseRepsNoTime(GVector<LeafNode*>& repArr, AbstractNode& parent,
                               std::vector<double>* ranVec, LeafNode& left,
                               LeafNode& right) {
    double totalInten = parent.getScalarTotalIntensity();
    double leftInten  = left.getScalarTotalIntensity();
    double nextTest   = (*ranVec)[0] * totalInten;
    for (unsigned int i = 0; i < repArr.size() - 1; i++) {
      double test = nextTest;
      nextTest    = (*ranVec)[i + 1] * totalInten;
      repArr[i]   = (test < leftInten) ? &left : &right;
    }
    repArr[repArr.size() - 1] = (nextTest < leftInten) ? &left : &right;
  }

  static void chooseRepsNoTime(GVector<LeafNode*>& repArr, AbstractNode& parent,
                               std::vector<double>* ranVec, ClusterNode& left,
                               LeafNode& right) {
    double totalInten = parent.getScalarTotalIntensity();
    double leftInten  = left.getScalarTotalIntensity();
    double nextTest   = (*ranVec)[0] * totalInten;
    for (unsigned int i = 0; i < repArr.size() - 1; i++) {
      double test = nextTest;
      nextTest    = (*ranVec)[i + 1] * totalInten;
      repArr[i]   = (test < leftInten) ? (left.reps[i]) : &right;
    }
    repArr[repArr.size() - 1] =
        (nextTest < leftInten) ? (left.reps[repArr.size() - 1]) : &right;
  }

  static void chooseRepsNoTime(GVector<LeafNode*>& repArr, AbstractNode& parent,
                               std::vector<double>* ranVec, ClusterNode& left,
                               ClusterNode& right) {
    double totalInten = parent.getScalarTotalIntensity();
    double leftInten  = left.getScalarTotalIntensity();
    double nextTest   = (*ranVec)[0] * totalInten;
    for (unsigned int i = 0; i < repArr.size() - 1; i++) {
      double test = nextTest;
      nextTest    = (*ranVec)[i + 1] * totalInten;
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

  void findConeDirsRecursive(GVector<double>* coordArr,
                             GVector<ClusterNode*>& tempClusterArr) {
    // TODO : Fix this. NodeWrapper::CONE_RECURSE_DEPTH - 1 = 3
    findConeDirsRecursive(*leftChild, coordArr, 0, tempClusterArr, 3);
    findConeDirsRecursive(*rightChild, coordArr, 0, tempClusterArr, 3);
  }

  static int findConeDirsRecursive(AbstractNode& node, GVector<double>* fArr,
                                   int numDirs, GVector<ClusterNode*>& cArr,
                                   int recurseDepth) {
    if (!node.isLeaf()) {
      ClusterNode& clus = (ClusterNode&)node;
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
        numDirs = findConeDirsRecursive(*(clus.leftChild), fArr, numDirs, cArr,
                                        recurseDepth - 1);
        numDirs = findConeDirsRecursive(*(clus.rightChild), fArr, numDirs, cArr,
                                        recurseDepth - 1);
      }
    } else {
      LeafNode& light = (LeafNode&)node;
      numDirs = addConeDir(fArr, numDirs, light.getDirX(), light.getDirY(),
                           light.getDirZ());
    }
    return numDirs;
  }

  static int addConeDir(GVector<double>* fArr, int numDirs, double x, double y,
                        double z) {
    // only add direction if it does not match any existing directions
    for (int i = 0; i < 3 * numDirs; i++) {
      if (((*fArr)[i] == x) && ((*fArr)[i + 1] == y) && ((*fArr)[i + 2] == z)) {
        return numDirs;
      }
    }
    int index          = 3 * numDirs;
    (*fArr)[index]     = x;
    (*fArr)[index + 1] = y;
    (*fArr)[index + 2] = z;
    return numDirs + 1;
  }

  bool isLeaf() { return false; }

  int size() {
    // only leafs are counted
    return leftChild->size() + rightChild->size();
  }
};

#endif /* CLUSTERNODE_H_ */

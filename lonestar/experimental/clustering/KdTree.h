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

#ifndef KDTREE_H_
#define KDTREE_H_

#include "KdCell.h"
#include "NodeWrapper.h"
#include "Point3.h"
#include "PotentialCluster.h"
#include <limits>

struct KdTreeFactory {

  using Alloc = galois::FixedSizeAllocator<KdCell>;
  Alloc m_alloc;

  template <typename... Args>
  KdCell* createCell(Args&& args...) {

    KdCell* c = m_alloc.allocate(1);
    assert(c && "alloc failed");
    m_alloc.construct(c, std::forward<Args>(args)...);
    return c;
  }

  void destroyCell(KdCell* c) {
    m_alloc.destruct(c);
    m_alloc.deallocate(c, 1);
  }
};

class KdTree : public KdCell {

protected:
  double minLightIntensity;
  double maxConeCosine;
  Point3 minHalfSize;

  KdTree() : KdCell(), minHalfSize(std::numeric_limits<double>::max()) {
    minLightIntensity = std::numeric_limits<double>::max();
    maxConeCosine     = -1.0f;
  }
  KdTree(int st, double sv) : KdCell(st, sv), minHalfSize(0) {
    minLightIntensity = 0;
    maxConeCosine     = -1.0f;
  }

public:
  static KdTree* createTree(GVector<NodeWrapper*>& inPoints) {
    KdTree* factory = new KdTree();
    KdTree* root    = (KdTree*)KdTree::subDivide(inPoints, 0, inPoints.size(),
                                              NULL, *factory);
    delete factory;
    return root;
  }

  virtual KdCell* createNewBlankCell(int inSplitType, double inSplitValue) {
    return new KdTree(inSplitType, inSplitValue);
  }

  static void getAll(KdCell& tree, GVector<NodeWrapper*>& allLeaves) {
    tree.getAll(allLeaves);
  }

  bool notifyPointAdded(NodeWrapper& nw, bool inChange) {
    if (inChange) {
      double b3         = nw.getLight().getScalarTotalIntensity();
      minLightIntensity = (minLightIntensity >= b3) ? b3 : minLightIntensity;
      maxConeCosine     = (maxConeCosine >= nw.getConeCosine())
                          ? maxConeCosine
                          : nw.getConeCosine();
      double b2 = nw.getHalfSizeX();
      double minHalfSizeX =
          (minHalfSize.getX() >= b2) ? b2 : minHalfSize.getX();
      double b1 = nw.getHalfSizeY();
      double minHalfSizeY =
          (minHalfSize.getY() >= b1) ? b1 : minHalfSize.getY();
      double b            = nw.getHalfSizeZ();
      double minHalfSizeZ = (minHalfSize.getZ() >= b) ? b : minHalfSize.getZ();
      minHalfSize.set(minHalfSizeX, minHalfSizeY, minHalfSizeZ);

    } else {
      double newIntensity = nw.getLight().getScalarTotalIntensity();
      if (minLightIntensity > newIntensity) {
        minLightIntensity = newIntensity;
        inChange          = true;
      }
      if (maxConeCosine < nw.getConeCosine()) {
        maxConeCosine = nw.getConeCosine();
        inChange      = true;
      }
      inChange |= minHalfSize.setIfMin(nw.getHalfSizeX(), nw.getHalfSizeY(),
                                       nw.getHalfSizeZ());
    }
    return inChange;
  }

  NodeWrapper* findBestMatch(NodeWrapper& inLight) {
    PotentialCluster cluster(inLight);
    if (splitType == LEAF) {
      findNearestRecursive(cluster);
    } else if (splitType == KdCell::SPLIT_X) {
      recurse(cluster, inLight.getLocationX());
    } else if (splitType == KdCell::SPLIT_Y) {
      recurse(cluster, inLight.getLocationY());
    } else if (splitType == KdCell::SPLIT_Z) {
      recurse(cluster, inLight.getLocationZ());
    } else {
      assert(false && "Invalid split type!");
    }
    NodeWrapper* res = cluster.closest;
    return res;
  }

  void findNearestRecursive(PotentialCluster& potentialCluster) {
    if (couldBeCloser(potentialCluster) == false) {
      return;
    }
    const NodeWrapper& from = potentialCluster.original;
    if (splitType == KdCell::LEAF) {
      // if it is a leaf then compute potential cluster size with each
      // individual light or cluster
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        if (pointList[i] != NULL &&
            pointList[i]->equals(potentialCluster.original) == false) {
          double size =
              NodeWrapper::potentialClusterSize(from, *(pointList[i]));
          if (size < potentialCluster.clusterSize) {
            potentialCluster.closest     = pointList[i];
            potentialCluster.clusterSize = size;
          }
        }
      }
    } else if (splitType == KdCell::SPLIT_X) {
      recurse(potentialCluster, from.getLocationX());
    } else if (splitType == KdCell::SPLIT_Y) {
      recurse(potentialCluster, from.getLocationY());
    } else if (splitType == KdCell::SPLIT_Z) {
      recurse(potentialCluster, from.getLocationZ());
    } else {
      assert(false && "Invalid split type in find nearest recursive");
    }
  }

  void recurse(PotentialCluster& potentialCluster, double which) {
    if (which <= splitValue) {
      if (leftChild != NULL && leftChild->removeFromTree == false)
        ((KdTree*)leftChild)->findNearestRecursive(potentialCluster);
      if (rightChild != NULL && rightChild->removeFromTree == false)
        ((KdTree*)rightChild)->findNearestRecursive(potentialCluster);
    } else {
      if (rightChild != NULL && rightChild->removeFromTree == false)
        ((KdTree*)rightChild)->findNearestRecursive(potentialCluster);
      if (leftChild != NULL && leftChild->removeFromTree == false)
        ((KdTree*)leftChild)->findNearestRecursive(potentialCluster);
    }
  }

  /**
   * Determines if any element of this cell could be closer to the the cluster,
   * outCluster, using the metrics defined in inBuilder.
   *
   * @param outCluster the cluster to test
   * @return true if an element could be closer, false otherwise
   */
  bool couldBeCloser(PotentialCluster& outCluster) {
    // first check to see if we can prove that none of our contents could be
    // closer than the current closest
    const NodeWrapper& from = outCluster.original;
    // compute minumum offset to bounding box
    double a2 =
        min.getX() - from.getLocationX() >= from.getLocationX() - max.getX()
            ? min.getX() - from.getLocationX()
            : from.getLocationX() - max.getX();
    // more than twice as fast as Math.max(a,0)
    double dx = (a2 >= 0) ? a2 : 0;
    double a1 =
        (min.getY() - from.getLocationY() >= from.getLocationY() - max.getY())
            ? min.getY() - from.getLocationY()
            : from.getLocationY() - max.getY();
    double dy = a1 >= 0 ? a1 : 0;
    double a =
        (min.getZ() - from.getLocationZ() >= from.getLocationZ() - max.getZ())
            ? min.getZ() - from.getLocationZ()
            : from.getLocationZ() - max.getZ();
    double dz = a >= 0 ? a : 0;
    // expand distance by half size of from's bounding box (distance is min to
    // center of box)  and by half the minimum bounding box extents of any node
    // in this cell
    dx += from.getHalfSizeX() + minHalfSize.getX();
    dy += from.getHalfSizeY() + minHalfSize.getY();
    dz += from.getHalfSizeZ() + minHalfSize.getZ();
    // cone must be at least as big as the larger of from's and the smallest in
    // this cell
    double coneCos = (maxConeCosine >= from.getConeCosine())
                         ? from.getConeCosine()
                         : maxConeCosine;
    // minimum cluster intensity would be from's intensity plus smallest
    // intensity inside this cell
    double intensity =
        minLightIntensity + from.getLight().getScalarTotalIntensity();
    Point3 diff(dx, dy, dz);

    double testSize = NodeWrapper::clusterSizeMetric(diff, coneCos, intensity);
    // return if our contents could be closer and so need to be checked
    // extra factor of 0.9999 is to correct for any roundoff error in computing
    // minimum size
    return (outCluster.clusterSize >= 0.9999 * testSize);
  }

private:
};

#endif /* KDTREE_H_ */

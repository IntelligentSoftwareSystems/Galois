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

#ifndef NODEWRAPPER_H_
#define NODEWRAPPER_H_

#include "Box3d.h"
#include "ClusterNode.h"
#include "LeafNode.h"
#include "Point3.h"
#include <math.h>

class NodeWrapper : public Box3d {
public:
  static const int CONE_RECURSE_SIZE;
  static const double GLOBAL_SCENE_DIAGONAL;

private:
  AbstractNode& light;
  Box3d direction;
  double coneCosine;
  Point3 location;
  Point3 coneDirection;
  const int descendents;
  GVector<ClusterNode*> coneClusters;
  const bool cleanLight;
  NodeWrapper *_l, *_r;

public:
  NodeWrapper(LeafNode& inNode)
      : light(inNode), location(0), coneDirection(0), descendents(1),
        cleanLight(false) {
    setBox(inNode.getPoint());
    direction.setBox(inNode.getDirection());
    coneCosine = 1.0f;
    coneDirection.set(inNode.getDirection());
    location.set(getMin());
    location.add(getMax());
    location.scale(0.5f);
    _l = _r = NULL;
  }

  NodeWrapper(NodeWrapper& pLeft, NodeWrapper& pRight,
              GVector<double>* coordArr, GVector<ClusterNode*>& tempClusterArr)
      : light(*(new ClusterNode())), location(0), coneDirection(0),
        descendents(pLeft.descendents + pRight.descendents), cleanLight(true) {
    NodeWrapper *l = &pLeft, *r = &pRight;
    if ((pLeft.location.getX() > pRight.location.getX()) ||
        ((pLeft.location.getX() == pRight.location.getX()) &&
         (pLeft.location.getY() > pRight.location.getY())) ||
        ((pLeft.location.getX() == pRight.location.getX()) &&
         (pLeft.location.getY() == pRight.location.getY()) &&
         (pLeft.location.getZ() > pRight.location.getZ()))) {
      l = &pRight;
      r = &pLeft;
    }
    addBox(*r);
    addBox(*l);
    location.set(max);
    location.add(min);
    location.scale(0.5);
    ((ClusterNode&)light).setBox(min, max);
    ((ClusterNode&)light)
        .setChildren(&l->light, &r->light,
                     ((double)rand()) / std::numeric_limits<double>::max());
    coneCosine = computeCone(*l, *r, ((ClusterNode&)light));
    if (coneCosine > -0.9f) {
      direction.addBox(l->direction);
      direction.addBox(r->direction);
      ((ClusterNode&)light).findConeDirsRecursive(coordArr, tempClusterArr);
      int numClus = 0;
      for (; tempClusterArr[numClus] != NULL; numClus++) {
      }
      if (numClus > 0) {
        this->coneClusters.resize(numClus);
        for (int j = 0; j < numClus; j++) {
          coneClusters[j]   = tempClusterArr[j];
          tempClusterArr[j] = NULL;
        }
      }
    }
    _l = l;
    _r = r;
  }

  ~NodeWrapper() {
    if (cleanLight) {
      delete (ClusterNode*)(&light);
    }
  }

  static double computeCone(const NodeWrapper& a, const NodeWrapper& b,
                            ClusterNode& cluster) {
    if (a.direction.isInitialized() == false ||
        b.direction.isInitialized() == false)
      return -1.0f;
    Point3 min(a.direction.getMin());
    min.setIfMin(b.direction.getMin());
    Point3 max(a.direction.getMax());
    max.setIfMax(b.direction.getMax());
    Point3 temp(max);
    temp.sub(min);
    double radiusSq = temp.getLen();
    temp.set(max);
    temp.add(min);
    double centerSq = temp.getLen();
    if (centerSq < 0.01) {
      return -1.0f;
    }
    double invLen = 1.0f / sqrt(centerSq);
    double minCos = (centerSq + 4.0f - radiusSq) * 0.25f * invLen;
    if (minCos < -1.0f) {
      minCos = -1.0f;
    }
    temp.scale(invLen);
    cluster.setDirectionCone(temp.getX(), temp.getY(), temp.getZ(), minCos);
    return minCos;
  }

  static double computeCone(const NodeWrapper& a, const NodeWrapper& b) {
    if (a.direction.isInitialized() == false ||
        b.direction.isInitialized() == false)
      return -1.0f;
    Point3 min(a.direction.getMin());
    min.setIfMin(b.direction.getMin());
    Point3 max(a.direction.getMax());
    max.setIfMax(b.direction.getMax());
    Point3 temp(max);
    temp.sub(min);
    double radiusSq = temp.getLen();
    temp.set(max);
    temp.add(min);
    double centerSq = temp.getLen();
    if (centerSq < 0.01) {
      return -1.0f;
    }
    double invLen = 1.0f / sqrt(centerSq);
    double minCos = (centerSq + 4.0f - radiusSq) * 0.25f * invLen;
    if (minCos < -1.0f) {
      minCos = -1.0f;
    }
    temp.scale(invLen);
    return minCos;
  }

  AbstractNode& getLight() const { return light; }

  double getLocationX() const { return location.getX(); }
  double getLocationY() const { return location.getY(); }

  double getLocationZ() const { return location.getZ(); }
  double getConeCosine() const { return coneCosine; }
  double getHalfSizeX() const { return max.getX() - location.getX(); }
  double getHalfSizeY() const { return max.getY() - location.getY(); }
  double getHalfSizeZ() const { return max.getZ() - location.getZ(); }

  const Point3& getLocation() const { return location; }

  bool equals(const NodeWrapper& other) {
    bool retVal = true;
    if (this->direction.equals(other.direction) == false)
      retVal &= false;
    if (this->coneCosine != other.coneCosine)
      retVal &= false;
    if (this->location.equals(other.location) == false)
      retVal &= false;
    if (this->coneDirection.equals(other.coneDirection) == false)
      retVal &= false;
    if (this->direction.equals(other.direction) == false)
      retVal &= false;
    // TODO : Add light comparison logic here!
    return retVal;
  }

  static double potentialClusterSize(const NodeWrapper& a, NodeWrapper& b) {
    Point3 max(a.max);
    max.setIfMax(b.max);
    Point3 min(a.min);
    min.setIfMin(b.min);
    Point3 diff(max);
    diff.sub(min);
    double minCos = computeCone(a, b);
    double maxIntensity =
        a.light.getScalarTotalIntensity() + b.light.getScalarTotalIntensity();
    return clusterSizeMetric(diff, minCos, maxIntensity);
  }

  /**
   * Compute a measure of the size of a light cluster
   */
  static double clusterSizeMetric(Point3& size, double cosSemiAngle,
                                  double intensity) {
    double len2        = size.getLen();
    double angleFactor = (1 - cosSemiAngle) * GLOBAL_SCENE_DIAGONAL;
    double res         = intensity * (len2 + angleFactor * angleFactor);
    return res;
  }
  /**
   *
   */
  friend std::ostream& operator<<(std::ostream& s, const NodeWrapper& node);
};

const int NodeWrapper::CONE_RECURSE_SIZE        = 4;
const double NodeWrapper::GLOBAL_SCENE_DIAGONAL = 2.0;

std::ostream& operator<<(std::ostream& s, const NodeWrapper& node) {
  s << "NW::[" << node.location << "] ConeClus :: ";
  for (unsigned int i = 0; i < node.coneClusters.size(); i++) {
    if (node.coneClusters[i] != NULL)
      s << "" << (*node.coneClusters[i]) << ",";
  }
  if (node._l != NULL)
    s << "{LEFT " << *node._l << "}";
  if (node._r != NULL)
    s << "{RIGHT " << *node._r << "}";
  s << std::endl;
  return s;
}
#endif /* NODEWRAPPER_H_ */

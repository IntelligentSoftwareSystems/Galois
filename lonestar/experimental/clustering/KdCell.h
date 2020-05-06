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

#ifndef KDCELL_H_
#define KDCELL_H_

#include "NodeWrapper.h"
#include "Point3.h"
#include "galois/gstl.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <limits>

class KdCell {
public:
  const static int LEAF;
  const static int SPLIT_X;
  const static int SPLIT_Y;
  const static int SPLIT_Z;
  const static int MAX_POINTS_IN_CELL;
  bool removeFromTree;

protected:
  Point3 min;
  Point3 max;
  const int m_splitType;
  const double m_splitValue;
  KdCell* m_leftChild;
  KdCell* m_rightChild;
  galois::gstl::Vector<NodeWrapper*> m_points;

public:
  KdCell()
      : min(std::numeric_limits<double>::max()),
        max(-1 * std::numeric_limits<double>::max()), m_splitType(LEAF),
        m_splitValue(std::numeric_limits<double>::max()) {
    m_points.resize(MAX_POINTS_IN_CELL);
    m_leftChild    = NULL;
    m_rightChild   = NULL;
    removeFromTree = false;
  }
  KdCell(int inSplitType, double inSplitValue)
      : min(0), max(0), m_splitType(inSplitType), m_splitValue(inSplitValue) {
    if (m_splitType == LEAF)
      m_points.resize(MAX_POINTS_IN_CELL);
    else
      m_points.resize(0);
    m_leftChild = m_rightChild = NULL;
    removeFromTree             = false;
  }
  virtual ~KdCell() {}

  bool equals(KdCell& other) {
    if (m_splitType != other.m_splitType)
      return false;
    if (m_splitValue != other.m_splitValue)
      return false;
    if (min.equals(other.min) == false)
      return false;
    if (max.equals(other.max) == false)
      return false;
    if (m_splitType == KdCell::LEAF)
      return m_leftChild->equals(*m_leftChild) &&
             m_rightChild->equals(*m_rightChild);
    if (m_points.size() != other.m_points.size())
      return false;
    for (unsigned int i = 0; i < m_points.size(); i++) {
      if (m_points[i] != NULL && other.m_points[i] != NULL) {
        if (m_points[i]->equals(*other.m_points[i]) == false)
          return false;
      }
      if (m_points[i] != other.m_points[i])
        return false;
    }
    return true;
  }

  // TODO: figure out how to use allocator here
  virtual KdCell* createNewBlankCell(int m_splitType, double m_splitValue) {
    std::cout << "KDCELL CALLED !!!!! " << std::endl;
    return (new KdCell(m_splitType, m_splitValue));
  }

  // TODO: figure out how to use allocator here
  static void cleanupTree(KdCell* root) {
    if (root->m_splitType == LEAF) {
      delete root;
      return;
    }
    if (root->m_leftChild != NULL)
      cleanupTree(root->m_leftChild);
    if (root->m_rightChild != NULL)
      cleanupTree(root->m_rightChild);
    delete root;
  }

  static KdCell* subDivide(galois::gstl::Vector<NodeWrapper*>& list, int offset,
                           const int size, galois::gstl::Vector<double>* arr,
                           KdCell& factory) {
    KdCell* toReturn;
    if (size <= KdCell::MAX_POINTS_IN_CELL) {

      toReturn     = factory.createNewBlankCell(KdCell::LEAF,
                                            std::numeric_limits<double>::max());
      KdCell& cell = *toReturn;
      for (int i = 0; i < size; i++) {
        cell.m_points[i] = list[offset + i];
      }
      for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
          if (i != j) {
            if (cell.m_points[i]->equals(*cell.m_points[j]))
              assert(false);
          }
        }
      }
      cell.computeBoundingBoxFromPoints(list, size);
      cell.notifyContentsRebuilt(true);
    } else {
      bool shouldClean = false;
      if (arr == NULL) {
        arr         = new galois::gstl::Vector<double>(size);
        shouldClean = true;
      }
      Point3 min(std::numeric_limits<double>::max());
      Point3 max(-std::numeric_limits<double>::max());
      for (int i = offset; i < size; i++) {
        min.setIfMin(list[i]->getMin());
        max.setIfMax(list[i]->getMax());
      }
      Point3 diff(max);
      diff.sub(min);
      int splitTypeUsed     = -1, splitType0, splitType1, splitType2;
      double splitValueUsed = -1;
      if (diff.getZ() > diff.getX() && diff.getZ() > diff.getY()) {
        splitType0      = KdCell::SPLIT_Z;
        bool comparCond = diff.getX() > diff.getY();
        splitType1      = comparCond ? KdCell::SPLIT_X : KdCell::SPLIT_Y;
        splitType2      = comparCond ? KdCell::SPLIT_Y : KdCell::SPLIT_X;
      } else if (diff.getY() > diff.getX()) {
        splitType0      = KdCell::SPLIT_Y;
        bool comparCond = diff.getX() > diff.getZ();
        splitType1      = comparCond ? KdCell::SPLIT_X : KdCell::SPLIT_Z;
        splitType2      = comparCond ? KdCell::SPLIT_Z : KdCell::SPLIT_X;
      } else {
        splitType0      = KdCell::SPLIT_X;
        bool comparCond = diff.getY() > diff.getZ();
        splitType1      = comparCond ? KdCell::SPLIT_Y : KdCell::SPLIT_Z;
        splitType2      = comparCond ? KdCell::SPLIT_Z : KdCell::SPLIT_Y;
      }

      splitTypeUsed  = splitType0;
      splitValueUsed = computeSplitValue(list, offset, size, splitType0, arr);
      if (splitValueUsed == std::numeric_limits<double>::max()) {
        splitTypeUsed  = splitType1;
        splitValueUsed = computeSplitValue(list, offset, size, splitType1, arr);
        if (splitValueUsed == std::numeric_limits<double>::max()) {
          splitTypeUsed = splitType2;
          splitValueUsed =
              computeSplitValue(list, offset, size, splitType2, arr);
        }
      }
      // Unable to find a good split along any axis!
      if (splitValueUsed == std::numeric_limits<double>::max()) {
        assert(false && "Unable to find a valid split across any dimension!");
      }
      int leftCountForSplit =
          splitList(list, offset, size, splitValueUsed, splitTypeUsed);
      if (leftCountForSplit <= 1 || leftCountForSplit >= size - 1) {
        assert(false && "Invalid split");
      }
      toReturn     = factory.createNewBlankCell(splitTypeUsed, splitValueUsed);
      KdCell& cell = *toReturn;
      cell.max.set(max);
      cell.min.set(min);
      cell.m_leftChild =
          subDivide(list, offset, leftCountForSplit, arr, factory);
      cell.m_rightChild = subDivide(list, offset + leftCountForSplit,
                                    size - leftCountForSplit, arr, factory);
      // Clean up on exit.
      if (shouldClean == true)
        delete arr;
    }
    return toReturn;
  }

  bool notifyContentsRebuilt(bool inChange) { return inChange; }

  static double computeSplitValue(galois::gstl::Vector<NodeWrapper*>& list,
                                  // TODO: create leaf node
                                  int offset, int size, int pSplitType,
                                  galois::gstl::Vector<double>* arr) {
    for (int i = 0; i < size; i++) {
      (*arr)[i] = findSplitComponent(*(list[offset + i]), pSplitType);
    }
    return findMedianGapSplit(arr, size);
    // TODO: create leaf node
  }

  static double findSplitComponent(NodeWrapper& n, int pSplitType) {
    if (pSplitType == KdCell::SPLIT_X)
      return n.getLocationX();
    if (pSplitType == KdCell::SPLIT_Y)
      return n.getLocationY();
    if (pSplitType == KdCell::SPLIT_Z)
      return n.getLocationZ();
    assert(false && "Invalid m_splitType requested in findSplitComponent");
    abort();
    return 0.0;
  }

  static double findMedianGapSplit(galois::gstl::Vector<double>* arr,
                                   int size) {
    sort(arr->begin(), arr->begin() + size);
    int start = ((size - 1) >> 1) - ((size + 7) >> 3);
    int end   = (size >> 1) + ((size + 7) >> 3);
    if (start == end) {
      // should never happen
      assert(false && "Start==End in findMedianSplit, should not happen!");
    }
    double largestGap   = 0;
    double m_splitValue = 0;
    double nextValue    = (*arr)[start];
    for (int i = start; i < end; i++) {
      double curValue = nextValue; // ie val[i]
      nextValue       = (*arr)[i + 1];
      if ((nextValue - curValue) > largestGap) {
        largestGap   = nextValue - curValue;
        m_splitValue = 0.5f * (curValue + nextValue);
        if (m_splitValue == nextValue) {
          m_splitValue = curValue;
        } // if not between then choose smaller value
      }
    }
    if (largestGap <= 0) {
      // indicate that the attempt to find a good split value failed
      m_splitValue = std::numeric_limits<double>::max();
    }
    return m_splitValue;
  }

  static int splitList(galois::gstl::Vector<NodeWrapper*>& list, int startIndex,
                       int size, double pSplitValue, const int pSplitType) {
    int lo = startIndex;
    int hi = startIndex + size - 1;
    // split into a low group that contains all points <= the split value and
    // a high group with all the points > the split value
    // note: after splitting, (lo - startIndex) will be the size of the low
    // group
    while (lo <= hi) {
      while (lo <= hi &&
             pSplitValue >= findSplitComponent(*(list[lo]), pSplitType)) {
        //				cout << "Lo[" << findSplitComponent(*(list[lo]),
        // pSplitType)
        //						<< "]";
        lo++;
      }
      while (lo <= hi &&
             pSplitValue < findSplitComponent(*(list[hi]), pSplitType)) {
        //				cout << "Hi[" << findSplitComponent(*(list[hi]),
        // pSplitType)
        //						<< "]";
        hi--;
      }
      if (lo < hi) {
        int index1        = lo++;
        int index2        = hi--;
        NodeWrapper* temp = list[index1];
        list[index1]      = list[index2];
        list[index2]      = temp;
      }
    }
    return lo - startIndex;
  }

  bool contains(NodeWrapper& point) {
    if (m_splitType == KdCell::LEAF) {
      // look for it in list of points
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        NodeWrapper* myNode = m_points[i];
        if (myNode != NULL && (*myNode).equals(point) == true) {
          return true;
        }
      }
      return false;
    } else {
      // otherwise its an interior node, so find which child should contain the
      // point
      double val    = findSplitComponent(point, m_splitType);
      KdCell* child = val <= m_splitValue ? m_leftChild : m_rightChild;
      if (child != NULL)
        return child->contains(point);
      return false;
    }
  }

  void getAll(galois::gstl::Vector<NodeWrapper*>& allLeaves) {
    if (this->m_splitType == KdCell::LEAF) {
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        if (m_points[i] != NULL)
          allLeaves.push_back(m_points[i]);
      }
    } else {
      m_leftChild->getAll(allLeaves);
      m_rightChild->getAll(allLeaves);
    }
  }

  bool remove(NodeWrapper& nw) {
    bool treeChanged = false;
    treeChanged      = removeInternal(nw, NULL, NULL);
    std::cout << "===================AFTER REMOVAL================" << *this
              << "=====================================" << std::endl;
    return treeChanged;
  }

  bool removeInternal(NodeWrapper& nw, KdCell* parent, KdCell* grandParent) {
    bool treeChanged = false;
    // Leaf Node!
    if (this->m_splitType == KdCell::LEAF) {
      int numPoints     = 0;
      int indexToDelete = -1;
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        if (m_points[i] != NULL) {
          if (m_points[i]->equals(nw) == true) {
            indexToDelete = i;
          }
          numPoints++;
        }
      }
      // If we found a match, delete the node.
      if (indexToDelete != -1) {
        if (numPoints == 1 && parent != NULL && grandParent != NULL) {
          std::cout << "About to Updated subnode :: " << *grandParent
                    << std::endl;
        }
        m_points[indexToDelete] = NULL;
        std::cout << "Removing " << nw << std::endl;
        treeChanged = recomputeLeafBoundingBoxIfChanges();
        treeChanged |= notifyContentsRebuilt(treeChanged);
        if (numPoints == 1 && parent != NULL && grandParent != NULL) {
          KdCell* otherChild;
          if (parent->m_leftChild->equals(*this)) {
            otherChild = m_rightChild;
          } else {
            otherChild = m_leftChild;
          }

          if (grandParent->m_leftChild->equals(*parent)) {
            grandParent->m_leftChild = otherChild;
          } else {
            grandParent->m_rightChild = otherChild;
          }
          this->removeFromTree   = true;
          parent->removeFromTree = true;
          std::cout << "Updated subnode :: " << *grandParent << std::endl;
        }
      }
    }
    // Interior node.
    else {
      double nodeSplitAxisValue = findSplitComponent(nw, m_splitType);
      KdCell* child =
          nodeSplitAxisValue <= m_splitValue ? m_leftChild : m_rightChild;
      treeChanged = child->removeInternal(nw, this, parent);
      std::cout << "BEFORE EX " << *this << std::endl;
      if (treeChanged == true && removeFromTree == false) {
        treeChanged |= recomputeParentBoundingBoxIfChanges();
        notifyContentsRebuilt(treeChanged);
      }
    }
    return treeChanged;
  }

  bool add(NodeWrapper& nw) { return add(NULL, this, nw); }

  static bool add(KdCell* parent, KdCell* current, NodeWrapper& nw) {
    bool treeChanged = false;
    if (current->m_splitType == KdCell::LEAF) {
      bool canInsert = false;
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        if (current->m_points[i] == NULL) {
          current->m_points[i] = &nw;
          canInsert            = true;
          break;
        }
      }
      // If we could not insert in there, we need to split this.
      if (canInsert == false) {
        if (parent == NULL) {
          assert(false && "Cannot split root node, in addNode");
        } else {
          galois::gstl::Vector<NodeWrapper*> newList(
              KdCell::MAX_POINTS_IN_CELL + 1);
          for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
            for (int j = 0; j < MAX_POINTS_IN_CELL; j++) {
              if (i != j) {
                if (current->m_points[i]->equals(*current->m_points[j]))
                  assert(false && "Sharing!!");
              }
            }
          }
          for (int i = 0; i < MAX_POINTS_IN_CELL; i++)
            newList[i] = current->m_points[i];
          newList[MAX_POINTS_IN_CELL] = &nw;
          KdCell* newCell             = subDivide(
              newList, 0, KdCell::MAX_POINTS_IN_CELL + 1, NULL, *current);
          if (parent->m_leftChild == current) {
            parent->m_leftChild = newCell;
          } else if (parent->m_rightChild == current) {
            parent->m_rightChild = newCell;
          }
          canInsert = true;
          delete current;
        }
      }
      treeChanged = canInsert;
    }
    // Internal node.
    else {
      double nodeSplitAxisValue = findSplitComponent(nw, current->m_splitType);
      treeChanged               = (nodeSplitAxisValue <= current->m_splitValue)
                        ? add(current, current->m_leftChild, nw)
                        : add(current, current->m_rightChild, nw);
      if (treeChanged) {
        bool change = current->addToBoundingBoxIfChanged(nw);
        change      = current->notifyPointAdded(nw, change);
      }
    }
    return treeChanged;
  }

private:
  bool notifyPointAdded(NodeWrapper& nw, bool inChange) { return inChange; }
  bool addToBoundingBoxIfChanged(NodeWrapper& nw) {
    bool retVal = min.setIfMin(nw.getLocation());
    retVal |= max.setIfMax(nw.getLocation());
    return retVal;
  }

  void computeBoundingBoxFromPoints(galois::gstl::Vector<NodeWrapper*>& list,
                                    int size) {
    Point3 newMin(std::numeric_limits<double>::max());
    Point3 newMax(-std::numeric_limits<double>::max());
    for (int i = 0; i < size; i++) {
      newMin.setIfMin(list[i]->getLocation());
      newMax.setIfMax(list[i]->getLocation());
    }
    min.set(newMin);
    max.set(newMax);
  }

  bool recomputeLeafBoundingBoxIfChanges() {
    Point3 newMin(std::numeric_limits<double>::max());
    Point3 newMax(-std::numeric_limits<double>::max());
    for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
      if (m_points[i] != NULL) {
        newMin.setIfMin(m_points[i]->getMin());
        newMax.setIfMax(m_points[i]->getMax());
      }
    }
    return updateBoundingBox(newMin, newMax);
  }

  bool recomputeParentBoundingBoxIfChanges() {
    Point3 newMin(m_leftChild->min);
    newMin.setIfMin(m_rightChild->min);
    Point3 newMax(m_leftChild->max);
    newMax.setIfMax(m_rightChild->max);
    return updateBoundingBox(newMin, newMax);
  }
  bool updateBoundingBox(Point3& newMin, Point3& newMax) {
    bool retVal = false;
    retVal      = min.setIfMin(newMin);
    retVal |= max.setIfMax(newMax);
    return retVal;
  }

  friend std::ostream& operator<<(std::ostream& s, KdCell& cell);
};

const int KdCell::SPLIT_X            = 0;
const int KdCell::SPLIT_Y            = 1;
const int KdCell::SPLIT_Z            = 2;
const int KdCell::LEAF               = 3;
const int KdCell::MAX_POINTS_IN_CELL = 4;

std::ostream& operator<<(std::ostream& s, KdCell& cell) {
  if (cell.m_splitType == KdCell::LEAF) {
    s << "Leaf ::[";
    for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
      if (cell.m_points[i] != NULL)
        s << *cell.m_points[i] << ",";
    }
    s << "]" << std::endl;
  } else {
    s << "InnerNode(" << cell.m_splitType << "," << cell.m_splitValue;
    if (cell.m_leftChild != NULL)
      s << ") \nLEFT::[" << (*cell.m_leftChild);
    else
      s << " NO-LEFT ";
    if (cell.m_rightChild != NULL)
      s << "]\nRIGHT::[" << (*cell.m_rightChild);
    else
      s << " NO-RIGHT";
    s << "]";
  }
  return s;
}
#endif /* KDCELL_H_ */

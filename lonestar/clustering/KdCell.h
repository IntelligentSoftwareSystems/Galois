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
#ifndef KDCELL_H_
#define KDCELL_H_
#include "NodeWrapper.h"
#include "Point3.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <limits>
#include <vector>
using namespace std;

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
  const int splitType;
  const double splitValue;
  KdCell* leftChild;
  KdCell* rightChild;
  vector<NodeWrapper*> pointList;

public:
  KdCell()
      : min(std::numeric_limits<double>::max()),
        max(-1 * std::numeric_limits<double>::max()), splitType(LEAF),
        splitValue(numeric_limits<double>::max()) {
    pointList.resize(MAX_POINTS_IN_CELL);
    leftChild      = NULL;
    rightChild     = NULL;
    removeFromTree = false;
  }
  KdCell(int inSplitType, double inSplitValue)
      : min(0), max(0), splitType(inSplitType), splitValue(inSplitValue) {
    if (splitType == LEAF)
      pointList.resize(MAX_POINTS_IN_CELL);
    else
      pointList.resize(0);
    leftChild = rightChild = NULL;
    removeFromTree         = false;
  }
  virtual ~KdCell() {}

  bool equals(KdCell& other) {
    if (splitType != other.splitType)
      return false;
    if (splitValue != other.splitValue)
      return false;
    if (min.equals(other.min) == false)
      return false;
    if (max.equals(other.max) == false)
      return false;
    if (splitType == KdCell::LEAF)
      return leftChild->equals(*leftChild) && rightChild->equals(*rightChild);
    if (pointList.size() != other.pointList.size())
      return false;
    for (unsigned int i = 0; i < pointList.size(); i++) {
      if (pointList[i] != NULL && other.pointList[i] != NULL) {
        if (pointList[i]->equals(*other.pointList[i]) == false)
          return false;
      }
      if (pointList[i] != other.pointList[i])
        return false;
    }
    return true;
  }
  /**
   *
   */
  virtual KdCell* createNewBlankCell(int splitType, double splitValue) {
    cout << "KDCELL CALLED !!!!! " << endl;
    return (new KdCell(splitType, splitValue));
  }
  /**
   *
   */
  static void cleanupTree(KdCell* root) {
    if (root->splitType == LEAF) {
      delete root;
      return;
    }
    if (root->leftChild != NULL)
      cleanupTree(root->leftChild);
    if (root->rightChild != NULL)
      cleanupTree(root->rightChild);
    delete root;
  }
  /***
   *
   */
  static KdCell* subDivide(vector<NodeWrapper*>& list, int offset,
                           const int size, vector<double>* arr,
                           KdCell& factory) {
    KdCell* toReturn;
    if (size <= KdCell::MAX_POINTS_IN_CELL) {

      toReturn     = factory.createNewBlankCell(KdCell::LEAF,
                                            numeric_limits<double>::max());
      KdCell& cell = *toReturn;
      for (int i = 0; i < size; i++) {
        cell.pointList[i] = list[offset + i];
      }
      for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
          if (i != j) {
            if (cell.pointList[i]->equals(*cell.pointList[j]))
              assert(false);
          }
        }
      }
      cell.computeBoundingBoxFromPoints(list, size);
      cell.notifyContentsRebuilt(true);
    } else {
      bool shouldClean = false;
      if (arr == NULL) {
        arr         = new vector<double>(size);
        shouldClean = true;
      }
      Point3 min(std::numeric_limits<float>::max());
      Point3 max(-std::numeric_limits<float>::max());
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
      //			cout<<
      //"================================================================"<<
      // endl;  Perform splitting, iteratively on type0, type1, type2, whichever
      // suceeds.
      splitTypeUsed  = splitType0;
      splitValueUsed = computeSplitValue(list, offset, size, splitType0, arr);
      if (splitValueUsed == numeric_limits<float>::max()) {
        splitTypeUsed  = splitType1;
        splitValueUsed = computeSplitValue(list, offset, size, splitType1, arr);
        if (splitValueUsed == numeric_limits<float>::max()) {
          splitTypeUsed = splitType2;
          splitValueUsed =
              computeSplitValue(list, offset, size, splitType2, arr);
        }
      }
      // Unable to find a good split along any axis!
      if (splitValueUsed == numeric_limits<float>::max()) {
        assert(false && "Unable to find a valid split across any dimension!");
      }
      //			cout << "Before :" << offset << " , " << size << " , value
      //::"
      //					<< splitValueUsed << " type:" << splitTypeUsed <<
      // endl;
      int leftCountForSplit =
          splitList(list, offset, size, splitValueUsed, splitTypeUsed);
      //			cout << "Splitting at " << offset << " , " <<
      // leftCountForSplit
      //					<< " , " << size << " , value ::" << splitValueUsed
      //					<< " type:" << splitTypeUsed << endl;
      if (leftCountForSplit <= 1 || leftCountForSplit >= size - 1) {
        //				for (int i = 0; i < size; i++)
        //					cout << "NW In split fault " << *list[offset + i] <<
        // endl; 				cout << "Failed at " << offset << " , " <<
        // leftCountForSplit
        //						<< " , " << size << " , value ::" <<
        // splitValueUsed
        //						<< " type:" << splitTypeUsed << endl;
        assert(false && "Invalid split");
      }
      toReturn     = factory.createNewBlankCell(splitTypeUsed, splitValueUsed);
      KdCell& cell = *toReturn;
      cell.max.set(max);
      cell.min.set(min);
      cell.leftChild = subDivide(list, offset, leftCountForSplit, arr, factory);
      cell.rightChild = subDivide(list, offset + leftCountForSplit,
                                  size - leftCountForSplit, arr, factory);
      //			cout << "created inner node" << cell;
      // Clean up on exit.
      if (shouldClean == true)
        delete arr;
    }
    return toReturn;
  }
  /**
   *
   */
  bool notifyContentsRebuilt(bool inChange) { return inChange; }
  /**
   *
   */
  static double computeSplitValue(vector<NodeWrapper*>& list, int offset,
                                  int size, int pSplitType,
                                  vector<double>* arr) {
    for (int i = 0; i < size; i++) {
      (*arr)[i] = findSplitComponent(*(list[offset + i]), pSplitType);
    }
    //		cout << "SplitVal ::[ " << pSplitType << "]";
    //		for (int i = 0; i < size; i++) {
    //			cout << "["<<*list[offset+i]<<" , "<<(*arr)[i] << ",]";
    //		}
    //		cout << endl;

    return findMedianGapSplit(arr, size);
  }
  /**
   *
   */
  static double findSplitComponent(NodeWrapper& n, int pSplitType) {
    if (pSplitType == KdCell::SPLIT_X)
      return n.getLocationX();
    if (pSplitType == KdCell::SPLIT_Y)
      return n.getLocationY();
    if (pSplitType == KdCell::SPLIT_Z)
      return n.getLocationZ();
    assert(false && "Invalid splitType requested in findSplitComponent");
    abort();
    return 0.0;
  }
  /**
   *
   */
  static double findMedianGapSplit(vector<double>* arr, int size) {

    //		cout << "Pre sort Median ::[ ";
    //		for (int i = 0; i < size; i++) {
    //			cout << (*arr)[i] << ",";
    //		}
    //		cout << "]" << endl;
    sort(arr->begin(), arr->begin() + size);
    //		cout << "Sorted Median ::[ ";
    //		for (int i = 0; i < size; i++) {
    //			cout << (*arr)[i] << ",";
    //		}
    //		cout << "]" << endl;
    int start = ((size - 1) >> 1) - ((size + 7) >> 3);
    int end   = (size >> 1) + ((size + 7) >> 3);
    if (start == end) {
      // should never happen
      assert(false && "Start==End in findMedianSplit, should not happen!");
    }
    double largestGap = 0;
    double splitValue = 0;
    double nextValue  = (*arr)[start];
    for (int i = start; i < end; i++) {
      double curValue = nextValue; // ie val[i]
      nextValue       = (*arr)[i + 1];
      if ((nextValue - curValue) > largestGap) {
        largestGap = nextValue - curValue;
        splitValue = 0.5f * (curValue + nextValue);
        if (splitValue == nextValue) {
          splitValue = curValue;
        } // if not between then choose smaller value
      }
    }
    if (largestGap <= 0) {
      // indicate that the attempt to find a good split value failed
      splitValue = numeric_limits<float>::max();
    }
    return splitValue;
  }
  /**
   *
   */
  static int splitList(vector<NodeWrapper*>& list, int startIndex, int size,
                       double pSplitValue, const int pSplitType) {
    //		for(int i=startIndex;i<size;i++){
    //			cout<<"NW to split :: "<<*list[i];
    //		}
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
  /**
   *
   */

  bool contains(NodeWrapper& point) {
    if (splitType == KdCell::LEAF) {
      // look for it in list of points
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        NodeWrapper* myNode = pointList[i];
        if (myNode != NULL && (*myNode).equals(point) == true) {
          return true;
        }
      }
      return false;
    } else {
      // otherwise its an interior node, so find which child should contain the
      // point
      float val     = findSplitComponent(point, splitType);
      KdCell* child = val <= splitValue ? leftChild : rightChild;
      if (child != NULL)
        return child->contains(point);
      return false;
    }
  }
  /**
   *
   */
  void getAll(vector<NodeWrapper*>& allLeaves) {
    if (this->splitType == KdCell::LEAF) {
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        if (pointList[i] != NULL)
          allLeaves.push_back(pointList[i]);
      }
    } else {
      leftChild->getAll(allLeaves);
      rightChild->getAll(allLeaves);
    }
  }
  /**
   *
   */
  bool remove(NodeWrapper& nw) {
    bool treeChanged = false;
    treeChanged      = removeInternal(nw, NULL, NULL);
    cout << "===================AFTER REMOVAL================" << *this
         << "=====================================" << endl;
    return treeChanged;
  }
  /**
   *
   */
  bool removeInternal(NodeWrapper& nw, KdCell* parent, KdCell* grandParent) {
    bool treeChanged = false;
    // Leaf Node!
    if (this->splitType == KdCell::LEAF) {
      int numPoints     = 0;
      int indexToDelete = -1;
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        if (pointList[i] != NULL) {
          if (pointList[i]->equals(nw) == true) {
            indexToDelete = i;
          }
          numPoints++;
        }
      }
      // If we found a match, delete the node.
      if (indexToDelete != -1) {
        if (numPoints == 1 && parent != NULL && grandParent != NULL) {
          cout << "About to Updated subnode :: " << *grandParent << endl;
        }
        pointList[indexToDelete] = NULL;
        cout << "Removing " << nw << endl;
        treeChanged = recomputeLeafBoundingBoxIfChanges();
        treeChanged |= notifyContentsRebuilt(treeChanged);
        if (numPoints == 1 && parent != NULL && grandParent != NULL) {
          //						cout<<"About to Updated subnode :: " <<
          //*grandParent<<endl;
          KdCell* otherChild;
          if (parent->leftChild->equals(*this)) {
            otherChild = rightChild;
          } else {
            otherChild = leftChild;
          }

          if (grandParent->leftChild->equals(*parent)) {
            grandParent->leftChild = otherChild;
          } else {
            grandParent->rightChild = otherChild;
          }
          this->removeFromTree   = true;
          parent->removeFromTree = true;
          cout << "Updated subnode :: " << *grandParent << endl;
        }
      }
    }
    // Interior node.
    else {
      double nodeSplitAxisValue = findSplitComponent(nw, splitType);
      KdCell* child = nodeSplitAxisValue <= splitValue ? leftChild : rightChild;
      treeChanged   = child->removeInternal(nw, this, parent);
      cout << "BEFORE EX " << *this << endl;
      if (treeChanged == true && removeFromTree == false) {
        treeChanged |= recomputeParentBoundingBoxIfChanges();
        notifyContentsRebuilt(treeChanged);
      }
    }
    return treeChanged;
  }
  /**
   *
   */
  bool add(NodeWrapper& nw) { return add(NULL, this, nw); }
  /**
   *
   */
  static bool add(KdCell* parent, KdCell* current, NodeWrapper& nw) {
    bool treeChanged = false;
    if (current->splitType == KdCell::LEAF) {
      bool canInsert = false;
      for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
        if (current->pointList[i] == NULL) {
          current->pointList[i] = &nw;
          canInsert             = true;
          break;
        }
      }
      // If we could not insert in there, we need to split this.
      if (canInsert == false) {
        if (parent == NULL) {
          assert(false && "Cannot split root node, in addNode");
        } else {
          vector<NodeWrapper*> newList(KdCell::MAX_POINTS_IN_CELL + 1);
          for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
            for (int j = 0; j < MAX_POINTS_IN_CELL; j++) {
              if (i != j) {
                if (current->pointList[i]->equals(*current->pointList[j]))
                  assert(false && "Sharing!!");
              }
            }
          }
          for (int i = 0; i < MAX_POINTS_IN_CELL; i++)
            newList[i] = current->pointList[i];
          newList[MAX_POINTS_IN_CELL] = &nw;
          KdCell* newCell             = subDivide(
              newList, 0, KdCell::MAX_POINTS_IN_CELL + 1, NULL, *current);
          if (parent->leftChild == current) {
            parent->leftChild = newCell;
          } else if (parent->rightChild == current) {
            parent->rightChild = newCell;
          }
          canInsert = true;
          delete current;
        }
      }
      treeChanged = canInsert;
    }
    // Internal node.
    else {
      double nodeSplitAxisValue = findSplitComponent(nw, current->splitType);
      treeChanged               = (nodeSplitAxisValue <= current->splitValue)
                        ? add(current, current->leftChild, nw)
                        : add(current, current->rightChild, nw);
      if (treeChanged) {
        bool change = current->addToBoundingBoxIfChanged(nw);
        change      = current->notifyPointAdded(nw, change);
      }
    }
    return treeChanged;
  }

private:
  /**
   *
   */
  bool notifyPointAdded(NodeWrapper& nw, bool inChange) { return inChange; }
  /**
   *
   */
  bool addToBoundingBoxIfChanged(NodeWrapper& nw) {
    bool retVal = min.setIfMin(nw.getLocation());
    retVal |= max.setIfMax(nw.getLocation());
    return retVal;
  }

  void computeBoundingBoxFromPoints(vector<NodeWrapper*>& list, int size) {
    Point3 newMin(numeric_limits<double>::max());
    Point3 newMax(-numeric_limits<double>::max());
    for (int i = 0; i < size; i++) {
      newMin.setIfMin(list[i]->getLocation());
      newMax.setIfMax(list[i]->getLocation());
    }
    min.set(newMin);
    max.set(newMax);
  }

  bool recomputeLeafBoundingBoxIfChanges() {
    Point3 newMin(numeric_limits<float>::max());
    Point3 newMax(-numeric_limits<float>::max());
    for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
      if (pointList[i] != NULL) {
        newMin.setIfMin(pointList[i]->getMin());
        newMax.setIfMax(pointList[i]->getMax());
      }
    }
    return updateBoundingBox(newMin, newMax);
  }
  bool recomputeParentBoundingBoxIfChanges() {
    Point3 newMin(leftChild->min);
    newMin.setIfMin(rightChild->min);
    Point3 newMax(leftChild->max);
    newMax.setIfMax(rightChild->max);
    return updateBoundingBox(newMin, newMax);
  }
  bool updateBoundingBox(Point3& newMin, Point3& newMax) {
    bool retVal = false;
    retVal      = min.setIfMin(newMin);
    retVal |= max.setIfMax(newMax);
    return retVal;
  }
  friend ostream& operator<<(ostream& s, KdCell& cell);
};
const int KdCell::SPLIT_X            = 0;
const int KdCell::SPLIT_Y            = 1;
const int KdCell::SPLIT_Z            = 2;
const int KdCell::LEAF               = 3;
const int KdCell::MAX_POINTS_IN_CELL = 4;

ostream& operator<<(ostream& s, KdCell& cell) {
  if (cell.splitType == KdCell::LEAF) {
    s << "Leaf ::[";
    for (int i = 0; i < KdCell::MAX_POINTS_IN_CELL; i++) {
      if (cell.pointList[i] != NULL)
        s << *cell.pointList[i] << ",";
    }
    s << "]" << std::endl;
  } else {
    s << "InnerNode(" << cell.splitType << "," << cell.splitValue;
    if (cell.leftChild != NULL)
      s << ") \nLEFT::[" << (*cell.leftChild);
    else
      s << " NO-LEFT ";
    if (cell.rightChild != NULL)
      s << "]\nRIGHT::[" << (*cell.rightChild);
    else
      s << " NO-RIGHT";
    s << "]";
  }
  return s;
}
#endif /* KDCELL_H_ */

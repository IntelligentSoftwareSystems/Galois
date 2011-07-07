/*
 * KdCell.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include<iostream>
#include<stdlib.h>
#include<limits>
#include<math.h>
#include<assert.h>
#include<algorithm>
#include<vector>
#include"KdTreeConflictManager.h"
#include "Galois/Runtime/Context.h"

#ifndef KDCELL_H_
#define KDCELL_H_

#define MAX_POINTS_IN_CELL 4
#define RETRY_LIMIT 100
#define SPLIT_X 0
#define SPLIT_Y 1
#define SPLIT_Z  2
#define LEAF  3
using namespace std;
class KdCell : public GaloisRuntime::Lockable{
public:
	// only set for the root
//	KdTreeConflictManager * cm;

private:
	bool removedFromTree;

protected:
	//bounding box of points contained in this cell (and all descendents)
	float xMin;
	float yMin;
	float zMin;
	float xMax;
	float yMax;
	float zMax;
	//X,Y,Z, or LEAF
	const int splitType;
	//pointers points if this is a leaf node
	std::vector<NodeWrapper*> * pointList;
	//split value if its not a leaf cell
	const float splitValue;
	//else if its not a leaf node, we need children
	KdCell *leftChild;
	KdCell *rightChild;
public:
	//set to true when a node is removed from the main kdtree
	bool isEqual(KdCell * other) {
		if (xMin != other->xMin)
			return false;
		if (yMin != other->yMin)
			return false;
		if (zMin != other->zMin)
			return false;
		if (xMax != other->xMax)
			return false;
		if (yMax != other->yMax)
			return false;
		if (zMax != other->zMax)
			return false;
		if (splitType != other->splitType)
			return false;
		if (this->splitType == LEAF)
			return true;
		return leftChild->isEqual(other->leftChild) && rightChild->isEqual(
				other->rightChild);
	}
public:
	/**
	 * Create a new empty KDTree
	 */
	KdCell() :
		splitType(LEAF), splitValue(std::numeric_limits<float>::max()) {
		xMin = yMin = zMin = std::numeric_limits<float>::max();
		xMax = yMax = zMax = (-1 * std::numeric_limits<float>::max());
		pointList = new std::vector<NodeWrapper*>(MAX_POINTS_IN_CELL);
		for (int i = 0; i < MAX_POINTS_IN_CELL; i++)
			(*pointList)[i] = NULL;
		leftChild = NULL;
		rightChild = NULL;
		removedFromTree=false;
	}

	//special constructor used internally
protected:
	KdCell(int inSplitType, float inSplitValue) :
		splitType(inSplitType), splitValue(inSplitValue) {
		leftChild=rightChild=NULL;
		//we don't set the bounding box as we assume it will be set next
		pointList = (inSplitType == LEAF) ? (new std::vector<NodeWrapper*>(MAX_POINTS_IN_CELL)) : NULL;
		if(inSplitType==LEAF)
			for(int i=0;i<MAX_POINTS_IN_CELL;i++)
				(*pointList)[i]=NULL;
		removedFromTree=false;
	}
public:
	/**
	 * We provide this factory method so that KDCell can be subclassed.  Returns a new
	 * uninitialized cell (also tried to reuse any preallocated array for holding children)
	 * Used during cell subdivision.
	 */
	virtual KdCell * createNewBlankCell(int inSplitType, float inSplitValue) {
		return new KdCell(inSplitType, inSplitValue);
	}

	//These methods are provided in case KDCell is subclassed.  Will be called after KDCell
	//has already been updated for the relevant operation
	//Because we want to prune unnecessary updates, these methods are passed a boolean
	//stating if the cell's statistics (eg, bounding box, etc) are known to have actually
	//changed and should return a boolean indicating if they found any changes.

	virtual bool notifyPointAdded(NodeWrapper *inPoint, bool inChanged) {
		return inChanged;
	}

	virtual bool notifyContentsRebuilt(bool inChanged) {
		return inChanged;
	}

	/**
	 * Check to see if adding this point changes the bounding box and enlarge
	 * bounding box if necessary.  Returns if the bounding box changed.
	 */
private:
	bool addToBoundingBoxIfChanges(NodeWrapper *cluster) {
		float x = cluster->getX();
		float y = cluster->getY();
		float z = cluster->getZ();
		bool retval = false;
		if (x < xMin) {
			xMin = x;
			retval = true;
		}
		if (x > xMax) {
			xMax = x;
			retval = true;
		}
		if (y < yMin) {
			yMin = y;
			retval = true;
		}
		if (y > yMax) {
			yMax = y;
			retval = true;
		}
		if (z < zMin) {
			zMin = z;
			retval = true;
		}
		if (z > zMax) {
			zMax = z;
			retval = true;
		}
		return retval;
	}

	bool recomputeLeafBoundingBoxIfChanges() {
		float xMinNew = std::numeric_limits<float>::max();
		float yMinNew = std::numeric_limits<float>::max();
		float zMinNew = std::numeric_limits<float>::max();
		float xMaxNew = (-1 * std::numeric_limits<float>::max());
		float yMaxNew = (-1 * std::numeric_limits<float>::max());
		float zMaxNew = (-1 * std::numeric_limits<float>::max());
		for (std::vector<NodeWrapper*>::iterator it = pointList->begin(),
				itEnd = pointList->end(); it != itEnd; ++it) {
			if (*it == NULL) {
				continue;
			}
			NodeWrapper * pt = (*it);
			float x = pt->getX();
			float y = pt->getY();
			float z = pt->getZ();
			xMinNew = std::min(x, xMinNew);
			yMinNew = std::min(y, yMinNew);
			zMinNew = std::min(z, zMinNew);
			xMaxNew = std::max(x, xMaxNew);
			yMaxNew = std::max(y, yMaxNew);
			zMaxNew = std::max(z, zMaxNew);
		}
		return updateBoundingBox(xMinNew, yMinNew, zMinNew, xMaxNew, yMaxNew,
				zMaxNew);
	}

	bool recomputeParentBoundingBoxIfChanges() {
		KdCell *left = leftChild;
		KdCell *right = rightChild;
		float xMinNew = std::min(left->xMin, right->xMin);
		float xMaxNew = std::max(left->xMax, right->xMax);
		float yMinNew = std::min(left->yMin, right->yMin);
		float yMaxNew = std::max(left->yMax, right->yMax);
		float zMinNew = std::min(left->zMin, right->zMin);
		float zMaxNew = std::max(left->zMax, right->zMax);
		return updateBoundingBox(xMinNew, yMinNew, zMinNew, xMaxNew, yMaxNew,
				zMaxNew);
	}

	bool updateBoundingBox(float xMinNew, float yMinNew, float zMinNew,
			float xMaxNew, float yMaxNew, float zMaxNew) {
		bool retval = false;
		if (xMinNew != xMin) {
			xMin = xMinNew;
			retval = true;
		}
		if (xMaxNew != xMax) {
			xMax = xMaxNew;
			retval = true;
		}
		if (yMinNew != yMin) {
			yMin = yMinNew;
			retval = true;
		}
		if (yMaxNew != yMax) {
			yMax = yMaxNew;
			retval = true;
		}
		if (zMinNew != zMin) {
			zMin = zMinNew;
			retval = true;
		}
		if (zMaxNew != zMax) {
			zMax = zMaxNew;
			retval = true;
		}
		return retval;
	}

	/**
	 * Computes this cells bounding box to just contain the specified points
	 */
	void computeBoundingBoxFromPoints(std::vector<NodeWrapper*> *list, int size) {
		float xMinNew = std::numeric_limits<float>::max();
		float yMinNew = std::numeric_limits<float>::max();
		float zMinNew = std::numeric_limits<float>::max();
		float xMaxNew = (-1 * std::numeric_limits<float>::max());
		float yMaxNew = (-1 * std::numeric_limits<float>::max());
		float zMaxNew = (-1 * std::numeric_limits<float>::max());
		for (int i = 0; i < size; i++) {
			float x = (*list)[i]->getX();
			float y = (*list)[i]->getY();
			float z = (*list)[i]->getZ();
			xMinNew = std::min(x, xMinNew);
			yMinNew = std::min(y, yMinNew);
			zMinNew = std::min(z, zMinNew);
			xMaxNew = std::max(x, xMaxNew);
			yMaxNew = std::max(y, yMaxNew);
			zMaxNew = std::max(z, zMaxNew);
		}
		xMin = xMinNew;
		xMax = xMaxNew;
		yMin = yMinNew;
		yMax = yMaxNew;
		zMin = zMinNew;
		zMax = zMaxNew;
	}

	/**
	 * Return the appropriate splitting component (x,y, or z) which is relevant for this node
	 */
	static float findSplitComponent(NodeWrapper * cluster, int splitType) {
		switch (splitType) {
		case SPLIT_X:
			return cluster->getX();
		case SPLIT_Y:
			return cluster->getY();
		case SPLIT_Z:
			return cluster->getZ();
		default:
			std::cout << "Error in findSplitComponent!!!" << std::endl;
		}
		assert(false&&"Invalid split type");
	}

	/**
	 * Given a list of points, and a split plane defined by the splitType and splitValue,
	 * partition the list into points below (<=) and above (>) the plane.  Returns the number of points
	 * which fell below the plane.
	 */
private:
	static int splitList(std::vector<NodeWrapper*>* &list, int startIndex,int size, float splitValue, int splitType) {
		int lo = startIndex;
		int hi = startIndex + size - 1;
		//split into a low group that contains all points <= the split value and
		//a high group with all the points > the split value
		//note: after splitting, (lo - startIndex) will be the size of the low group
		while (lo <= hi) {
			while (lo <= hi && splitValue >= findSplitComponent((*list)[lo],
					splitType)) {
				lo++;
			}
			while (lo <= hi && splitValue < findSplitComponent((*list)[hi],
					splitType)) {
				hi--;
			}
			if (lo < hi) {
				int index1 = lo++;
				int index2 = hi--;
				NodeWrapper * temp = (*list)[index1];
				(*list)[index1] = (*list)[index2];
				(*list)[index2] = temp;
			}
		}
		return lo - startIndex;
	}

	/**
	 * Sets the contents of this cell to the specified list and size.  Then sudivides the cell as
	 * necessary to build an appropriate hierarchy.  Val is a temporary array of floats that
	 * we can pass in to reduce the allocation of additional temporary space.
	 */
protected:
	static KdCell* subdivide(std::vector<NodeWrapper*> *&list, int offset,int size, float *floatArr, KdCell * factory) {
		//		std::cout << "Starting subdivision with list size:: " << list->size() << ", off:" << offset << ", size: " << size << std::endl;//", floatArr:"<<floatArr->size()<<""<<std::endl;
		if (size <= MAX_POINTS_IN_CELL) {
			//If less than or equal to 4 nodes, then create a new bounding box and return it.
			KdCell * cell = factory->createNewBlankCell(LEAF,std::numeric_limits<float>::max());
			//			std::cout<<"Copying to a new BlankCell "<<std::endl;
			for (int i = 0; i < size; i++) {
				(*(cell->pointList))[i] = (*list)[offset + i];
			}
			cell->computeBoundingBoxFromPoints(cell->pointList, size);
			cell->notifyContentsRebuilt(true);
			return cell;
		}
		//otherwise its an interior node and we need to choose a split plane
		if (floatArr == NULL) {
			floatArr = new float[size];
		}
		//compute bounding box of points
		float xMin = std::numeric_limits<float>::max();
		float yMin = std::numeric_limits<float>::max();
		float zMin = std::numeric_limits<float>::max();
		float xMax = (-1 * std::numeric_limits<float>::max());
		float yMax = (-1 * std::numeric_limits<float>::max());
		float zMax = (-1 * std::numeric_limits<float>::max());

		for (int i = offset; i < size + offset; i++) {
			float x = (*list)[i]->getX();
			float y = (*list)[i]->getY();
			float z = (*list)[i]->getZ();
			xMin = min(x, xMin);
			yMin = min(y, yMin);
			zMin = min(z, zMin);
			xMax = max(x, xMax);
			yMax = max(y, yMax);
			zMax = max(z, zMax);
		}
		//choose split plane
		float sx = xMax - xMin;
		float sy = yMax - yMin;
		float sz = zMax - zMin;
		int type;
		float value;
		int type0, type1, type2;
		//		std::cout<<"In subdivide X:"<<sx<<" Y:"<<sy<<" Z:"<<sz<<std::endl;
		if (sz > sx && sz > sy) {
			type0 = SPLIT_Z;
			bool cond = sx > sy;
			type1 = cond ? SPLIT_X : SPLIT_Y;
			type2 = cond ? SPLIT_Y : SPLIT_X;
		} else if (sy > sx) {
			type0 = SPLIT_Y;
			bool cond = sx > sz;
			type1 = cond ? SPLIT_X : SPLIT_Z;
			type2 = cond ? SPLIT_Z : SPLIT_X;
		} else {
			type0 = SPLIT_X;
			bool cond = sy > sz;
			type1 = cond ? SPLIT_Y : SPLIT_Z;
			type2 = cond ? SPLIT_Z : SPLIT_Y;
		}
		type = type0;
		value = computeSplitValue(list, offset, size, type0, floatArr);
		if (value == std::numeric_limits<float>::max()) {
			//attempt to split failed so try another axis
			type = type1;
			value = computeSplitValue(list, offset, size, type1, floatArr);
			if (value == std::numeric_limits<float>::max()) {
				type = type2;
				value = computeSplitValue(list, offset, size, type2, floatArr);
			}
		}
		if (value == std::numeric_limits<float>::max()) {
			//throw new RuntimeException("badness splittype:" + type + " value:" + value + " size:" + size + " sx:" + sx+ " sy:" + sy + " sz:" + sz);
			std::cout << "badness splittype:" << type << " value:" << value
					<< " size:" << size << " sx:" << sx << " sy:" << sy
					<< " sz:" << sz << std::endl;
			assert(false);
		}
		int leftCount = splitList(list, offset, size, value, type);
		if (leftCount <= 1 || leftCount >= size - 1) {
			//throw new RuntimeException("badness splittype:" + type + " value:" + value + " leftCount:" + leftCount+ " rightCount: " + (size - leftCount) + " sx:" + sx + " sy:" + sy + " sz:" + sz);
			std::cout << "badness splittype:" << type << " value:" << value
					<< " leftCount:" << leftCount << " rightCount: " << (size
					- leftCount) << " sx:" << sx << " sy:" << sy << " sz:"
					<< sz;
			//TODO FIX THIS!!!
			assert(false);
		}
		KdCell *cell = factory->createNewBlankCell(type, value);
		cell->xMin = xMin;
		cell->xMax = xMax;
		cell->yMin = yMin;
		cell->yMax = yMax;
		cell->zMin = zMin;
		cell->zMax = zMax;
		cell->leftChild = (subdivide(list, offset, leftCount, floatArr, factory));
		cell->rightChild = (subdivide(list, offset + leftCount, size - leftCount, floatArr, factory));
		cell->notifyContentsRebuilt(true);
		return cell;
	}

	static float computeSplitValue(std::vector<NodeWrapper*>*& list, int offset,
			int size, int splitType, float*& floatArr) {
		for (int i = 0; i < size; i++) {
			floatArr[i] = findSplitComponent((*list)[offset + i], splitType);
		}
		return findMedianGapSplit(floatArr, size);
	}

	/**
	 * Given an array of floats, sorts the list, finds the largest gap in values
	 * near the median, and returns a value in the middle of that gap
	 */
private:
	static float findMedianGapSplit(float *& val, int size) {
		//this is not very efficient at the moment, there are faster median finding algorithms
		sort(val, val + size);
		int start = ((size - 1) >> 1) - ((size + 7) >> 3);
		int end = (size >> 1) + ((size + 7) >> 3);
		if (start == end) {
			//should never happen
			assert(false&&"Error in findMedianGapSplit");
		}
		float largestGap = 0;
		float splitValue = 0;
		float nextValue = val[start];
		for (int i = start; i < end; i++) {
			float curValue = nextValue; //ie val[i]
			nextValue = val[i + 1];
			if ((nextValue - curValue) > largestGap) {
				largestGap = nextValue - curValue;
				splitValue = 0.5f * (curValue + nextValue);
				if (splitValue == nextValue) {
					splitValue = curValue;
				} //if not between then choose smaller value
			}
		}
		if (largestGap <= 0) {
			//indicate that the attempt to find a good split value failed
			return std::numeric_limits<float>::max();
		}
		return splitValue;
	}

public:
	bool add(NodeWrapper *inPoint) {
		acquire(this);
//		acquire(inPoint);
		int ret = addPoint(inPoint, NULL);
		if (ret == -1) {
			std::cout << "Retrying to add" << std::endl;
		} else if (ret == 0 || ret == 1) {
			//			std::cout<<"Added node "<<(*inPoint)<<std::endl;
			return true;
		} else {
			//		 throw new RuntimeException();
			assert(false&& "Unable to add point!");
		}
		//		 throw new RuntimeException("repeated retries of concurrent op still failed");
		return false;
	}

	//return value is true if child stats changed (and so need to potentially update this node)

private:
	int addPoint(NodeWrapper *cluster, KdCell *parent) {
		if (splitType == LEAF) {
			//      synchronized (this)
			{
				if (removedFromTree) {
					//this leaf node is no longer in the tree
					return -1;
				}
				const int numPoints = pointList->size();
				for (int i = 0; i < numPoints; i++) {
					if ((*pointList)[i] == NULL) {
						(*pointList)[i] = cluster;
						bool changed = addToBoundingBoxIfChanges(cluster);
						return notifyPointAdded(cluster, changed) ? 1 : 0;
					}
				}
				//if we get here the point list was full so we need to subdivide the node
				std::vector<NodeWrapper*> *fullList = new std::vector<NodeWrapper*>(numPoints + 1);
				for (int i = 0; i < numPoints; i++)
					(*fullList)[i] = (*pointList)[i];
				(*fullList)[numPoints] = cluster;
				KdCell *subtree = subdivide(fullList, 0, numPoints + 1, NULL,this);
				//substitute refined subtree for ourself by changing parent's child ptr
				//        synchronized (parent)
				{
					if (parent->removedFromTree) {
						//if parent no longer valid, retry from beginning
						return -1;
					}
					if (parent->leftChild == this) {
						parent->leftChild = subtree;
					} else if (parent->rightChild == this) {
						parent->rightChild = subtree;
					} else {
						//pointer was changed by someone else
						//            throw new RuntimeException();
						assert(false && "Error in addPint, parent");
					}
					this->removedFromTree = true;
				}
			}
			//assume changed as its not easy to check for changes when refining leaf to subtree
			return 1;
		}
		//its an interior node, so see which child should receive this new point
		float val = findSplitComponent(cluster, splitType);
		KdCell *child = val <= splitValue ? leftChild : rightChild;
		int status = child->addPoint(cluster, this);
		if (status == 1) {
			//      synchronized (this)
			{
				if (removedFromTree) {
					return 1;
				}
				//if node is no longer in the tree, tell parent to check for changes, but don't bother updating this node
				bool changed = addToBoundingBoxIfChanges(cluster);
				changed = notifyPointAdded(cluster, changed);
				status = changed ? 1 : 0;
			}
		}
		return status;
	}

public:
	//	bool remove(NodeWrapper *cluster) {
	//		//    return remove(cluster, MethodFlag.ALL);
	//		return remove(cluster, 'a');
	//	}

	/**
	 * Remove a ClusterKDWrapper from the octree.  Returns true if found and removed
	 * and false otherwise.  Will un-subdivide if count is low enough but does not
	 * trigger rebalancing of the tree.
	 */
	//TODO fix this!
	//  bool remove(NodeWrapper * cluster, byte flags) {
	bool remove(NodeWrapper *&cluster) {
		//		std::cout << "Removing node " << (*cluster) << std::endl;
		acquire(this);
		int ret = 0;
		for (int i = 0; i < 5; i++) {
			ret = removePoint(cluster, NULL, NULL);
			if (ret == -2) {
//				std::cout<<"Unable to remove "<<*cluster;
				return false;
//				assert(false&&"cannot remove cluster");
			} else if (ret == -1) {
				std::cout << "Retrying to remove" << std::endl;
			} else if (ret == 0 || ret == 1) {
				//			std::cout << "Done Removing node " << (*cluster) << std::endl;
				return true;
			} else {
				assert(false&&"Runtime exception");
			}
		}
//		std::cout << "Returned " << ret << std::endl;
		assert(false&&"remove failed after repeated retries");

	}

private:
	int removePoint(NodeWrapper *&inRemove, KdCell *parent, KdCell *grandparent) {
		if (splitType == LEAF) {
			//look for it in list of points
			//      synchronized (this)
			{
				//				std::cout<<"Removing at Leaf";
				if (removedFromTree) {
					//this leaf node is no longer in the tree
					return -1;
				}
				int index = -1;
				int count = 0;
				for (int i = 0; i < (int) pointList->size(); i++) {
					if ((*pointList)[i] != NULL) {
						if ((*pointList)[i]->isEqual(inRemove)) {
							index = i;
						}
						//TODO Verify this is correct?
						if ((*pointList)[i] != NULL) {
							count++;
						}
					}
				}
				if (index < 0) {
					// instead of throwing NoSuchElementException
//					std::cout << "Unable to remove :: " << (*inRemove)<< std::endl;
					return -2;
				}
				if (count == 1 && parent != NULL && grandparent != NULL) {
					//snip parent and this node out of the tree and replace with parent's other child
					//          synchronized (parent)
					{
						//            synchronized (grandparent)
						{
							if (parent->removedFromTree|| grandparent->removedFromTree) {
								//tree structure status, so retry op
								return -1;
							}
							KdCell *otherChild = NULL;
							if ((parent->leftChild)->isEqual(this)) {
								otherChild = parent->rightChild;
							} else if ((parent->rightChild)->isEqual(this)) {
								otherChild = parent->leftChild;
							} else {
								//                throw new RuntimeException();
								assert(false);
							}
							this->removedFromTree = true;
							parent->removedFromTree = true;
							if ((grandparent->leftChild)->isEqual(parent)) {
								grandparent->leftChild = otherChild;
							} else if ((grandparent->rightChild)->isEqual(
									parent)) {
								grandparent->rightChild = otherChild;
							} else {
								//                throw new RuntimeException();
								assert(false);
							}
							return 1;
						}
					}
				}
				//once found, remove the point and recompute our bounding box
				//TODO : Fix this?
				//				delete (*pointList)[index];
				(*pointList)[index] = NULL;
				bool changed = recomputeLeafBoundingBoxIfChanges();
				changed = notifyContentsRebuilt(changed);
				return changed ? 1 : 0;
			}
		}
		//otherwise its an interior node, so find which child should contain the point
		float val = findSplitComponent(inRemove, splitType);
		KdCell *child = val <= splitValue ? leftChild : rightChild;
		int status = child->removePoint(inRemove, this, parent);
		if (status == 1) {
			//      synchronized (this)
			{
				if (removedFromTree) {
					return 1;
				}
				//if node is no longer in the tree, tell parent to check for changes, but don't bother updating this node
				bool changed = recomputeParentBoundingBoxIfChanges();
				status = notifyContentsRebuilt(changed) ? 1 : 0;
			}
		} else if (status == 0) {
			//not sure this check is necessary, but leaving it in as a precaution for now
			status = removedFromTree ? 1 : 0;
		}
		return status;
	}

public:
	NodeWrapper *getAny(double ranNum) {
		NodeWrapper *retval = internalGetAny(ranNum);
		return retval;
	}

	/**
	 * Returns some point from the kdtree.  Will not return null unless there are no points left
	 * in the tree.  Random number helps randomize the selection from the tree, but there is no
	 * guarantee about exactly which point will be returned
	 */
	NodeWrapper *internalGetAny(double ranNum) {
		NodeWrapper *retval = NULL;
		if (splitType == LEAF) {
			int length = pointList->size();
			int i = (int) (ranNum * length);
			for (int j = 0; j < length; j++) {
				retval = (*pointList)[i];
				if (retval != NULL) {
					return retval;
				}
				i = (i + 1) % length;
			}
		} else {
			if (ranNum < 0.5) {
				ranNum *= 2;
				retval = leftChild->internalGetAny(ranNum);
				if (retval == NULL) {
					retval = rightChild->internalGetAny(ranNum);
				}
			} else {
				ranNum = 2 * ranNum - 1.0;
				retval = rightChild->internalGetAny(ranNum);
				if (retval == NULL) {
					retval = leftChild->internalGetAny(ranNum);
				}
			}
		}
		return retval;
	}

public:
	bool contains(NodeWrapper *inPoint) {
		acquire (this);
		return internalContains(inPoint);

	}

	bool internalContains(NodeWrapper *point) {
		if (splitType == LEAF) {
			//look for it in list of points
			//      for (NodeWrapper aPointList : pointList) {
			for (unsigned int i = 0; i < pointList->size(); i++) {
				NodeWrapper* aPointList = (*pointList)[i];
				if (aPointList == NULL)
					continue;
				if (aPointList->isEqual(point)) {
					return true;
				}
			}
			return false;
		}
		//otherwise its an interior node, so find which child should contain the point
		float val = findSplitComponent(point, splitType);
		KdCell *child = val <= splitValue ? leftChild : rightChild;
		return child->internalContains(point);
	}

	/**
	 * Perform a variety of consistency checks on the tree and throws an error if any of them fail
	 * This method is not concurrent safe.
	 */
	virtual bool isOkay() {
		if (removedFromTree) {
			//        throw new IllegalStateException("removed flag set for node still in tree");
			std::cout << "ERR !! removed flag set for node still in tree"
					<< std::endl;
		}
		if (splitType == LEAF) {
			if (leftChild != NULL || rightChild != NULL) {
				//        throw new IllegalStateException("leaf has child");
				std::cout << "ERR !! leaf has child" << std::endl;
			}
			if (pointList->size() != MAX_POINTS_IN_CELL) {
				//        throw new IllegalStateException("point list is wrong size");
				std::cout << "ERR !! point list is wrong size" << std::endl;
			}
			//check that the bounding box is right
			float xMinNew = std::numeric_limits<float>::max();
			float yMinNew = std::numeric_limits<float>::max();
			float zMinNew = std::numeric_limits<float>::max();
			float xMaxNew = (-1 * std::numeric_limits<float>::max());
			float yMaxNew = (-1 * std::numeric_limits<float>::max());
			float zMaxNew = (-1 * std::numeric_limits<float>::max());
			//      for (NodeWrapper aPointList : pointList) {
			for (unsigned int i = 0; i < pointList->size(); i++) {
				NodeWrapper * aPointList = (*pointList)[i];
				if (aPointList == NULL) {
					continue;
				}
				float x = aPointList->getX();
				float y = aPointList->getY();
				float z = aPointList->getZ();
				xMinNew = min(x, xMinNew);
				yMinNew = min(y, yMinNew);
				zMinNew = min(z, zMinNew);
				xMaxNew = max(x, xMaxNew);
				yMaxNew = max(y, yMaxNew);
				zMaxNew = max(z, zMaxNew);
			}
			if (xMin != xMinNew || yMin != yMinNew || zMin != zMinNew) {
				assert(false&&"bad bounding box");
			}
			if (xMax != xMaxNew || yMax != yMaxNew || zMax != zMaxNew) {
				assert(false&&"bad bounding box");
			}
		} else { //its an interior node
			leftChild->isOkay();
			rightChild->isOkay();
			if (pointList != NULL) {
				assert(false&&"split nodes should not contain points");
			}
			if (xMin != min(leftChild->xMin, rightChild->xMin)) {
				assert(false&&"bad bounding box");
			}
			if (yMin != min(leftChild->yMin, rightChild->yMin)) {
				assert(false&&"bad bounding box");
			}
			if (zMin != min(leftChild->zMin, rightChild->zMin)) {
				assert(false&&"bad bounding box");
			}
			if (xMax != max(leftChild->xMax, rightChild->xMax)) {
				assert(false&&"bad bounding box");
			}
			if (yMax != max(leftChild->yMax, rightChild->yMax)) {
				assert(false&&"bad bounding box");
			}
			if (zMax != max(leftChild->zMax, rightChild->zMax)) {
				assert(false&&"bad bounding box");
			}
			switch (splitType) {
			case SPLIT_X:
				if (leftChild->xMax > splitValue || rightChild->xMin
						< splitValue) {
					assert(false&&"incorrect split");
				}
				break;
			case SPLIT_Y:
				if (leftChild->yMax > splitValue || rightChild->yMin
						< splitValue) {
					assert(false&&"incorrect split");
				}
				break;
			case SPLIT_Z:
				if (leftChild->zMax > splitValue || rightChild->zMin
						< splitValue) {
					assert(false&&"incorrect split");
				}
				break;
			default:
				assert(false&&"bad split type");
			}
		}
		return true;
	}
	KdCell & operator=(KdCell & other){
		assert(false&&"Not implemented");
	}
	friend std::ostream& operator<<(std::ostream &s, KdCell & c);
};
std::ostream& operator<<(std::ostream &s, KdCell & c) {
	if (c.splitType == LEAF) {
		s<<"Leaf : " << (*c.pointList)[0] << ", " << (*c.pointList)[1] << ", "<< (*c.pointList)[2] << ", " << (*c.pointList)[3];
	}
	else {
		s<<"SUBTREE [\nLeft " << *c.leftChild << " \n Right " <<*c.rightChild<< "\nSUBTREEEND]";
	}
	return s;
}
#endif /* KDCELL_H_ */

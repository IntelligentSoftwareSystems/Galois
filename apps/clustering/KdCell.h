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

#ifndef KDCELL_H_
#define KDCELL_H_

#define MAX_POINTS_IN_CELL 4
#define RETRY_LIMIT 100
#define SPLIT_X 0
#define SPLIT_Y 1
#define SPLIT_Z  2
#define LEAF  3
using namespace std;
class KdCell {
public:
	// private Logger logger = Logger.getLogger("apps.clustering");
	// private boolean isFineLoggable = Logger.getLogger("apps.clustering").isLoggable(Level.FINE);

	// only set for the root
	KdTreeConflictManager * cm;

private:
	//	static const int MAX_POINTS_IN_CELL;//= 4;
	//	static const int RETRY_LIMIT;// = 100;
	bool removedFromTree;

public:
	//	static const int SPLIT_X;// = 0;
	//	static const int SPLIT_Y;// = 1;
	//	static const int SPLIT_Z;// = 2;
	//	static const int LEAF;// = 3;

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
	//set to true when a node is removed from the main kdtree
public:
	/**
	 * Create a new empty KDTree
	 */
	KdCell() :
		splitType(LEAF), splitValue(std::numeric_limits<float>::max()) {
		xMin = yMin = zMin = std::numeric_limits<float>::max();
		xMax = yMax = zMax = std::numeric_limits<float>::min();
		pointList = new std::vector<NodeWrapper*>(MAX_POINTS_IN_CELL);
		leftChild = NULL;
		rightChild = NULL;
	}

	//special constructor used internally

	KdCell(int inSplitType, float inSplitValue) :
		splitType(inSplitType), splitValue(inSplitValue) {
		//we don't set the bounding box as we assume it will be set next
		pointList = (inSplitType == LEAF) ? new std::vector<NodeWrapper*>(
				MAX_POINTS_IN_CELL) : NULL;
	}

	/**
	 * We provide this factory method so that KDCell can be subclassed.  Returns a new
	 * uninitialized cell (also tried to reuse any preallocated array for holding children)
	 * Used during cell subdivision.
	 */
	KdCell *createNewBlankCell(int inSplitType, float inSplitValue) {
		return new KdCell(inSplitType, inSplitValue);
	}

	//These methods are provided in case KDCell is subclassed.  Will be called after KDCell
	//has already been updated for the relevant operation
	//Because we want to prune unnecessary updates, these methods are passed a boolean
	//stating if the cell's statistics (eg, bounding box, etc) are known to have actually
	//changed and should return a boolean indicating if they found any changes.

	bool notifyPointAdded(NodeWrapper *inPoint, bool inChanged) {
		return inChanged;
	}

	bool notifyContentsRebuilt(bool inChanged) {
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
		float xMinNew = std::numeric_limits<float>::max(), yMinNew =
				std::numeric_limits<float>::max(), zMinNew =
				std::numeric_limits<float>::max();
		float xMaxNew = std::numeric_limits<float>::min(), yMaxNew =
				std::numeric_limits<float>::min(), zMaxNew =
				std::numeric_limits<float>::min();
		for (std::vector<NodeWrapper*>::iterator it = pointList->begin(),
				itEnd = pointList->end(); it != itEnd; ++it) {
			if (*it == NULL) {
				continue;
			}
			NodeWrapper *pt = *it;
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
		float xMinNew = std::numeric_limits<float>::max(), yMinNew =
				std::numeric_limits<float>::max(), zMinNew =
				std::numeric_limits<float>::max();
		float xMaxNew = std::numeric_limits<float>::min(), yMaxNew =
				std::numeric_limits<float>::min(), zMaxNew =
				std::numeric_limits<float>::min();
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
		float retVal = 0;
		switch (splitType) {
		case SPLIT_X:
//			return cluster->getX();
//			std::cout << "X  ";
			retVal  = cluster->getX();
			break;
		case SPLIT_Y:
//			std::cout << "Y  ";
			retVal  =  cluster->getY();
			break;
		case SPLIT_Z:
//			std::cout << "Z  ";
			retVal  =  cluster->getZ();
			break;
		default:
			std::cout << "Error in findSplitComponent!!!" << std::endl;
		}
//		std::cout<<"Find split component " << retVal << std::endl;
		return retVal;
	}

	/**
	 * Given a list of points, and a split plane defined by the splitType and splitValue,
	 * partition the list into points below (<=) and above (>) the plane.  Returns the number of points
	 * which fell below the plane.
	 */
private:
	static int splitList(std::vector<NodeWrapper*>* list, int startIndex,
			int size, float splitValue, int splitType) {
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
	static KdCell* subdivide(std::vector<NodeWrapper*> *list, int offset,int size, std::vector<float> *floatArr, KdCell *factory) {
		if (size <= MAX_POINTS_IN_CELL) {
			KdCell * cell = factory->createNewBlankCell(LEAF,std::numeric_limits<float>::max());
			//System.arraycopy(list, offset, cell.pointList, 0, size);
			//list->clear();
			for (int i = 0; i < size; i++)
				(*(cell->pointList))[i] = (*list)[offset + i];
			cell->computeBoundingBoxFromPoints(cell->pointList, size);
			cell->notifyContentsRebuilt(true);
			return cell;
		}
		//otherwise its an interior node and we need to choose a split plane
		if (floatArr == NULL) {
			floatArr = new std::vector<float>(size);
		}
		//compute bounding box of points
		float xMin = std::numeric_limits<float>::max(),
			  yMin = std::numeric_limits<float>::max(),
			  zMin = std::numeric_limits<float>::max();
		float xMax = std::numeric_limits<float>::min(),
			  yMax = std::numeric_limits<float>::min(),
			  zMax = std::numeric_limits<float>::min();
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
			assert(false);
		}
		KdCell *cell = factory->createNewBlankCell(type, value);
		cell->xMin = xMin;
		cell->xMax = xMax;
		cell->yMin = yMin;
		cell->yMax = yMax;
		cell->zMin = zMin;
		cell->zMax = zMax;
		cell->leftChild = subdivide(list, offset, leftCount, floatArr, factory);
		cell->rightChild = subdivide(list, offset + leftCount,
				size - leftCount, floatArr, factory);
		cell->notifyContentsRebuilt(true);
		return cell;
	}

	static float computeSplitValue(std::vector<NodeWrapper*>* list, int offset,int size, int splitType, std::vector<float>* floatArr) {
		for (int i = 0; i < size; i++) {
			(*floatArr)[i] = findSplitComponent((*list)[offset + i], splitType);
		}
		return findMedianGapSplit(floatArr, size);
	}

	/**
	 * Given an array of floats, sorts the list, finds the largest gap in values
	 * near the median, and returns a value in the middle of that gap
	 */
private:
	static float findMedianGapSplit(std::vector<float> * val, int size) {
		//this is not very efficient at the moment, there are faster median finding algorithms
		sort(val->begin(), val->end());
//		std::cout<<" Sorted";
//		for(std::vector<float>::iterator it = val->begin(), itEnd = val->end();it!=itEnd;++it)
//			std::cout<<" "<<(*it)<<",";
//		std::cout<<std::endl;
//		assert(false);
		//Arrays.sort(val, 0, size);
		int start = ((size - 1) >> 1) - ((size + 7) >> 3);
		int end = (size >> 1) + ((size + 7) >> 3);
		if (start == end) {
			//should never happen
			//throw new RuntimeException();
			std::cout << "Error in findMedianGapSplit" << std::endl;
		}
		float largestGap = 0;
		float splitValue = 0;
		float nextValue = (*val)[start];
		for (int i = start; i < end; i++) {
			float curValue = nextValue; //ie val[i]
			nextValue = (*val)[i + 1];
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
	bool add(NodeWrapper *inAdd) {
		//return add(inAdd, MethodFlag.ALL);
		return add(inAdd,'a');
	}

	/**
	 * Add a ClusterKDWrapper to the kdtree, subdividing if necessary
	 */
	//TODO Finish this!!
public:
	bool add(NodeWrapper * inPoint, unsigned char flags) {
		/*
		 if (GaloisRuntime.needMethodFlag(flags, MethodFlag.CHECK_CONFLICT)) {
		 cm.addRemoveProlog(inPoint);
		 }
		 if (GaloisRuntime.needMethodFlag(flags, MethodFlag.SAVE_UNDO)) {
		 GaloisRuntime.getRuntime().onUndo(Iteration.getCurrentIteration(), new Lambda0Void() {
		 //@Override
		 public void call() {
		 remove(inPoint, MethodFlag.NONE);
		 }
		 });
		 }
		 */
		for (int i = 0; i < RETRY_LIMIT; i++) {
			int ret = addPoint(inPoint, NULL);
			if (ret == -1) {
				//		 if (isFineLoggable) {
				//		 logger.info("retrying addPoint");
				//		 }
			} else if (ret == 0 || ret == 1) {
				return true;
			} else {
				//		 throw new RuntimeException();
			}
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
				std::vector<NodeWrapper*> fullList(numPoints + 1);
				for (int i = 0; i < numPoints; i++)
					fullList[i] = (*pointList)[i];
				//        System.arraycopy(pointList, 0, fullList, 0, numPoints);
				fullList[numPoints] = cluster;
				KdCell *subtree = subdivide(&fullList, 0, numPoints + 1, NULL,
						this);
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
						std::cout << "Error in addPint, parent" << std::endl;
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
	bool remove(NodeWrapper *cluster) {
		//    return remove(cluster, MethodFlag.ALL);
		return remove(cluster,'a');
	}

	/**
	 * Remove a ClusterKDWrapper from the octree.  Returns true if found and removed
	 * and false otherwise.  Will un-subdivide if count is low enough but does not
	 * trigger rebalancing of the tree.
	 */
	//TODO fix this!
	//  bool remove(NodeWrapper * cluster, byte flags) {
	bool remove(NodeWrapper * cluster, unsigned char flags) {
		/*
		 if (GaloisRuntime.needMethodFlag(flags, MethodFlag.CHECK_CONFLICT)) {
		 cm.addRemoveProlog(cluster);
		 }
		 if (GaloisRuntime.needMethodFlag(flags, MethodFlag.SAVE_UNDO)) {
		 GaloisRuntime.getRuntime().onUndo(Iteration.getCurrentIteration(), new Lambda0Void() {
		 //					@Override
		 public void call() {
		 add(cluster, MethodFlag.NONE);
		 }
		 });
		 }*/
		for (int i = 0; i < RETRY_LIMIT; i++) {
			int ret = removePoint(cluster, NULL, NULL);
			if (ret == -2) {
				assert(false&&"cannot remove cluster");
				//		 throw new RuntimeException("cannot remove cluster");
			} else if (ret == -1) {
				//		 if (isFineLoggable) {
				//		 logger.fine("retrying removal");
				//		 }
			} else if (ret == 0 || ret == 1) {
				return true;
			} else {
				//		 throw new RuntimeException();
				assert(false&&"Runtimeexception");
			}
		}
		assert(false&&"remove failed after repeated retries");
		//		 throw new RuntimeException("remove failed after repeated retries");

	}

private:
	int removePoint(NodeWrapper *inRemove, KdCell *parent, KdCell *grandparent) {
		if (splitType == LEAF) {
			//look for it in list of points
			//      synchronized (this)
			{
				if (removedFromTree) {
					//this leaf node is no longer in the tree
					return -1;
				}
				int index = -1;
				int count = 0;
				for (int i = 0; i < (int)pointList->size(); i++) {
					if ((*pointList)[i] == inRemove) {
						index = i;
					}
					if ((*pointList)[i] != NULL) {
						count++;
					}
				}
				if (index < 0) {
					// instead of throwing NoSuchElementException
					return -2;
				}
				if (count == 1 && parent != NULL && grandparent != NULL) {
					//snip parent and this node out of the tree and replace with parent's other child
					//          synchronized (parent)
					{
						//            synchronized (grandparent)
						{
							if (parent->removedFromTree
									|| grandparent->removedFromTree) {
								//tree structure status, so retry op
								return -1;
							}
							KdCell *otherChild = NULL;
							if (parent->leftChild == this) {
								otherChild = parent->rightChild;
							} else if (parent->rightChild == this) {
								otherChild = parent->leftChild;
							} else {
								//                throw new RuntimeException();
								assert(false);
							}
							this->removedFromTree = true;
							parent->removedFromTree = true;
							if (grandparent->leftChild == parent) {
								grandparent->leftChild = otherChild;
							} else if (grandparent->rightChild == parent) {
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
		//    return getAny(ranNum, MethodFlag.ALL);
		return getAny(ranNum);
	}
	//TODO Fix this!!!
	//  NodeWrapper *getAny(double ranNum, byte flags) {
	NodeWrapper *getAny(double ranNum, unsigned char flags) {
		//		bool checkConflict = GaloisRuntime.needMethodFlag(flags, MethodFlag.CHECK_CONFLICT);
		//		 KdTreeConflictManager.LocalEntryLog finishedTail = checkConflict ? cm.readBestMatchProlog() : null;
		NodeWrapper *retval = internalGetAny(ranNum);
		//		 if (checkConflict)
		//		 {
		//		 cm.readEpilog(retval, finishedTail);
		//		 }
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
		//    return contains(inPoint, MethodFlag.ALL);
		return contains(inPoint,'a');
	}

	//  bool contains(NodeWrapper *point, byte flags) {
	//TODO Fix this!!!
	bool contains(NodeWrapper *point, unsigned char flags) {
		//    if (GaloisRuntime.needMethodFlag(flags, MethodFlag.CHECK_CONFLICT)) {
		//      cm.addRemoveProlog(point);
		//    }
		bool retval = internalContains(point);
		return retval;
	}

	bool internalContains(NodeWrapper *point) {
		if (splitType == LEAF) {
			//look for it in list of points
			//      for (NodeWrapper aPointList : pointList) {
			for (unsigned int i = 0; i < pointList->size(); i++) {
				NodeWrapper* aPointList = (*pointList)[i];
				if (aPointList == point) {
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
	bool isOkay() {
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
			float xMinNew = std::numeric_limits<float>::max(), yMinNew =
					std::numeric_limits<float>::max(), zMinNew =
					std::numeric_limits<float>::max();
			float xMaxNew = std::numeric_limits<float>::min(), yMaxNew =
					std::numeric_limits<float>::min(), zMaxNew =
					std::numeric_limits<float>::min();
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
				//        throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
			if (xMax != xMaxNew || yMax != yMaxNew || zMax != zMaxNew) {
				//        throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
		} else { //its an interior node
			leftChild->isOkay();
			rightChild->isOkay();
			if (pointList != NULL) {
				//          throw new IllegalStateException("split nodes should not contain points");
				assert(false&&"split nodes should not contain points");
			}
			if (xMin != min(leftChild->xMin, rightChild->xMin)) {
				//          throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
			if (yMin != min(leftChild->yMin, rightChild->yMin)) {
				//        throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
			if (zMin != min(leftChild->zMin, rightChild->zMin)) {
				//          throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
			if (xMax != max(leftChild->xMax, rightChild->xMax)) {
				//        throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
			if (yMax != max(leftChild->yMax, rightChild->yMax)) {
				//        throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
			if (zMax != max(leftChild->zMax, rightChild->zMax)) {
				//        throw new IllegalStateException("bad bounding box");
				assert(false&&"bad bounding box");
			}
			switch (splitType) {
			case SPLIT_X:
				if (leftChild->xMax > splitValue || rightChild->xMin
						< splitValue) {
					//            throw new IllegalStateException("incorrect split");
					assert(false&&"incorrect split");
				}
				break;
			case SPLIT_Y:
				if (leftChild->yMax > splitValue || rightChild->yMin
						< splitValue) {
					//          throw new IllegalStateException("incorrect split");
					assert(false&&"incorrect split");
				}
				break;
			case SPLIT_Z:
				if (leftChild->zMax > splitValue || rightChild->zMin
						< splitValue) {
					//            throw new IllegalStateException("incorrect split");
					assert(false&&"incorrect split");
				}
				break;
			default:
				//        throw new IllegalStateException("bad split type");
				assert(false&&"bad split type");
			}
		}
		return true;
	}
};
//const int KdCell::MAX_POINTS_IN_CELL = 4;
//const int KdCell::RETRY_LIMIT = 100;
//const int KdCell::SPLIT_X = 0;
//const int KdCell::SPLIT_Y = 1;
//const int KdCell::SPLIT_Z = 2;
//const int KdCell::LEAF = 3;
#endif /* KDCELL_H_ */

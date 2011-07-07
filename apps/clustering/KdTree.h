/** Unordered Agglomerative Clustering -*- C++ -*-
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
 * @author Rashid Kaleem <rashid@cs.utexas.edu>
 */
#include"KdCell.h"
#include"PotentialCluster.h"
#ifndef KDTREE_H_
#define KDTREE_H_
class KdTree: public KdCell {

//minimum intensity of any light or cluster in this node, needed as part of cluster size metric
private:
	float minLightIntensity;
	float maxConeCos;
	float minHalfSizeX;
	float minHalfSizeY;
	float minHalfSizeZ;

private:
	KdTree():KdCell() {
		minLightIntensity = std::numeric_limits<float>::max();
		maxConeCos = -1.0f;
		minHalfSizeX = std::numeric_limits<float>::max();
		minHalfSizeY = std::numeric_limits<float>::max();
		minHalfSizeZ = std::numeric_limits<float>::max();
	}
	//special constructor used internally when space for point list has already been allocated
	KdTree(int inSplitType, float inSplitValue) :
		KdCell(inSplitType, inSplitValue) {
	}
private:
	void findNearestRecursive(PotentialCluster *&potentialCluster) {
		acquire(this);
		if (!couldBeCloser(potentialCluster)) {
			return;
		}
		NodeWrapper *from = potentialCluster->original;
		if (splitType == LEAF) {
			//if it is a leaf then compute potential cluster size with each individual light or cluster
			for (unsigned int i = 0; i < pointList->size(); i++) {
				NodeWrapper *aPointList = (*pointList)[i];
				if (aPointList != NULL && aPointList!= potentialCluster->original) {
					double size = NodeWrapper::potentialClusterSize(*from,*aPointList);
					if (size < potentialCluster->clusterSize) {
						potentialCluster->closest = aPointList;
						potentialCluster->clusterSize = size;
					}
				}
			}
		} else if (splitType == SPLIT_X) {
			recurse(potentialCluster, from->getX());
		} else if (splitType == SPLIT_Y) {
			recurse(potentialCluster, from->getY());
		} else if (splitType == SPLIT_Z) {
			recurse(potentialCluster, from->getZ());
		} else {
			assert(false&&"Badness error in findNearestRecursive....");
		}
	}

	void recurse(PotentialCluster *& potentialCluster, float which) {
		//if its a interior node recurse on the closer child first
		if (which <= splitValue) {
			((KdTree*) leftChild)->findNearestRecursive(potentialCluster);
			((KdTree*) rightChild)->findNearestRecursive(potentialCluster);
		} else {
			((KdTree*) rightChild)->findNearestRecursive(potentialCluster);
			((KdTree*) leftChild)->findNearestRecursive(potentialCluster);
		}
	}

	/**
	 * Determines if any element of this cell could be closer to the the cluster, outCluster, using
	 * the metrics defined in inBuilder.
	 *
	 * @param outCluster the cluster to test
	 * @param inBuilder  the builder defining closeness
	 * @return true if an element could be closer, false otherwise
	 */
protected:
	bool couldBeCloser(PotentialCluster *& outCluster) {
		//first check to see if we can prove that none of our contents could be closer than the current closest
		NodeWrapper * from = outCluster->original;
		//compute minumum offset to bounding box
		float a2 = xMin - from->getX() >= from->getX() - xMax ? xMin
				- from->getX() : from->getX() - xMax;
		//more than twice as fast as Math.max(a,0)
		float dx = (a2 >= 0) ? a2 : 0;
		float a1 = (yMin - from->getY() >= from->getY() - yMax) ? yMin
				- from->getY() : from->getY() - yMax;
		float dy = a1 >= 0 ? a1 : 0;
		float a = (zMin - from->getZ() >= from->getZ() - zMax) ? zMin
				- from->getZ() : from->getZ() - zMax;
		float dz = a >= 0 ? a : 0;
		//expand distance by half size of from's bounding box (distance is min to center of box)
		//and by half the minimum bounding box extents of any node in this cell
		float t = from->getHalfSizeX();
		dx += t+ minHalfSizeX;
		dy += from->getHalfSizeY() + minHalfSizeY;
		dz += from->getHalfSizeZ() + minHalfSizeZ;
		//cone must be at least as big as the larger of from's and the smallest in this cell
		float coneCos = (maxConeCos >= from->coneCos) ? from->coneCos
				: maxConeCos;
		//minimum cluster intensity would be from's intensity plus smallest intensity inside this cell
		float intensity = minLightIntensity
				+ from->light->getScalarTotalIntensity();
		double testSize = NodeWrapper::clusterSizeMetric(dx, dy, dz, coneCos,
				intensity);
		//return if our contents could be closer and so need to be checked
		//extra factor of 0.9999 is to correct for any roundoff error in computing minimum size
		return (outCluster->clusterSize >= 0.9999 * testSize);
	}

	/*--- Methods needed to implement as an extended KDCell in a KDTree ---*/

	/**
	 * We provide this factory method so that KDCell can be subclassed.  Returns a new
	 * uninitialized cell (also tried to reuse any preallocated array for holding children)
	 * Used during cell subdivision.
	 */

	virtual KdCell *createNewBlankCell(int inSplitType, float inSplitValue) {
		return new KdTree(inSplitType, inSplitValue);
	}

	virtual bool notifyContentsRebuilt(bool changed) {
		//must recompute the min light intensity since the cells contents have changed
		if (splitType == LEAF) {
			float newMinInten = numeric_limits<float>::max();
			float newMaxCos = -1.0f;
			float newMinHX = numeric_limits<float>::max(), newMinHY =
					numeric_limits<float>::max(), newMinHZ = numeric_limits<
					float>::max();
			for (unsigned int i = 0; i < pointList->size(); i++) {
				NodeWrapper *aPointList = (*pointList)[i];
				if (aPointList == NULL) {
					continue;
				}
				float b3 = aPointList->light->getScalarTotalIntensity();
				newMinInten = (newMinInten >= b3) ? b3 : newMinInten;
				newMaxCos = (newMaxCos >= aPointList->coneCos) ? newMaxCos: aPointList->coneCos;
				float b2 = aPointList->getHalfSizeX();
				newMinHX = (newMinHX >= b2) ? b2 : newMinHX;
				float b1 = aPointList->getHalfSizeY();
				newMinHY = (newMinHY >= b1) ? b1 : newMinHY;
				float b = aPointList->getHalfSizeZ();
				newMinHZ = (newMinHZ >= b) ? b : newMinHZ;
			}
			if (changed) {
				minLightIntensity = newMinInten;
				maxConeCos = newMaxCos;
				minHalfSizeX = newMinHX;
				minHalfSizeY = newMinHY;
				minHalfSizeZ = newMinHZ;
			} else {
				if (minLightIntensity != newMinInten) {
					minLightIntensity = newMinInten;
					changed = true;
				}
				if (maxConeCos != newMaxCos) {
					maxConeCos = newMaxCos;
					changed = true;
				}
				if (minHalfSizeX != newMinHX) {
					minHalfSizeX = newMinHX;
					changed = true;
				}
				if (minHalfSizeY != newMinHY) {
					minHalfSizeY = newMinHY;
					changed = true;
				}
				if (minHalfSizeZ != newMinHZ) {
					minHalfSizeZ = newMinHZ;
					changed = true;
				}
			}
		} else {
			//its a split node
			KdTree* left = (KdTree*) leftChild;
			KdTree* right = (KdTree*) rightChild;
			if (changed) {
				minLightIntensity = (left->minLightIntensity>= right->minLightIntensity) ? right->minLightIntensity : left->minLightIntensity;
				maxConeCos		  = (left->maxConeCos >= right->maxConeCos) 			 ? left->maxConeCos			: right->maxConeCos;
				minHalfSizeX	  = (left->minHalfSizeX >= right->minHalfSizeX) 		 ? right->minHalfSizeX		: left->minHalfSizeX;
				minHalfSizeY 	  = (left->minHalfSizeY >= right->minHalfSizeY) 		 ? right->minHalfSizeY		: left->minHalfSizeY;
				minHalfSizeZ 	  = (left->minHalfSizeZ >= right->minHalfSizeZ) 		 ? right->minHalfSizeZ		: left->minHalfSizeZ;
			} else {
				float newMinInten = (left->minLightIntensity>= right->minLightIntensity) ? right->minLightIntensity: left->minLightIntensity;
				float newMaxCos = (left->maxConeCos >= right->maxConeCos) ? left->maxConeCos: right->maxConeCos;
				float newMinHX =(left->minHalfSizeX >= right->minHalfSizeX) ? right->minHalfSizeX: left->minHalfSizeX;
				float newMinHY = (left->minHalfSizeY >= right->minHalfSizeY) ? right->minHalfSizeY : left->minHalfSizeY;
				float newMinHZ = (left->minHalfSizeZ >= right->minHalfSizeZ) ? right->minHalfSizeZ : left->minHalfSizeZ;
				if (minLightIntensity != newMinInten) {
					minLightIntensity = newMinInten;
					changed = true;
				}
				if (maxConeCos != newMaxCos) {
					maxConeCos = newMaxCos;
					changed = true;
				}
				if (minHalfSizeX != newMinHX) {
					minHalfSizeX = newMinHX;
					changed = true;
				}
				if (minHalfSizeY != newMinHY) {
					minHalfSizeY = newMinHY;
					changed = true;
				}
				if (minHalfSizeZ != newMinHZ) {
					minHalfSizeZ = newMinHZ;
					changed = true;
				}
			}
		}
		return changed;
	}
	virtual bool notifyPointAdded(NodeWrapper *inPoint, bool changed) {
		if (changed) {
			float b3 = inPoint->light->getScalarTotalIntensity();
			minLightIntensity = (minLightIntensity >= b3) ? b3
					: minLightIntensity;
			maxConeCos = (maxConeCos >= inPoint->coneCos) ? maxConeCos
					: inPoint->coneCos;
			float b2 = inPoint->getHalfSizeX();
			minHalfSizeX = (minHalfSizeX >= b2) ? b2 : minHalfSizeX;
			float b1 = inPoint->getHalfSizeY();
			minHalfSizeY = (minHalfSizeY >= b1) ? b1 : minHalfSizeY;
			float b = inPoint->getHalfSizeZ();
			minHalfSizeZ = (minHalfSizeZ >= b) ? b : minHalfSizeZ;
		} else {
			float newInten = inPoint->light->getScalarTotalIntensity();
			float hx = inPoint->getHalfSizeX();
			float hy = inPoint->getHalfSizeY();
			float hz = inPoint->getHalfSizeZ();
			if (minLightIntensity > newInten) {
				minLightIntensity = newInten;
				changed = true;
			}
			if (maxConeCos < inPoint->coneCos) {
				maxConeCos = inPoint->coneCos;
				changed = true;
			}
			if (minHalfSizeX > hx) {
				minHalfSizeX = hx;
				changed = true;
			}
			if (minHalfSizeY > hy) {
				minHalfSizeY = hy;
				changed = true;
			}
			if (minHalfSizeZ > hz) {
				minHalfSizeZ = hz;
				changed = true;
			}
		}
		return changed;
	}

	/**
	 * Perform a variety of consistency checks on the tree and throws an error if any of them fail
	 */
public:
	bool isOkay() {
		KdCell::isOkay();
		float minLight = numeric_limits<float>::max();
		float maxCos = -1.0f;
		float minHX = numeric_limits<float>::max();
		float minHY = numeric_limits<float>::max();
		float minHZ = numeric_limits<float>::max();
		if (splitType == LEAF) {
			for (unsigned int i = 0; i < pointList->size(); i++) {
				NodeWrapper *aPointList = (*pointList)[i];
				if (aPointList == NULL) {
					continue;
				}
				minLight = std::min(minLight,
						aPointList->light->getScalarTotalIntensity());
				maxCos = std::max(maxCos, aPointList->coneCos);
				minHX = std::min(minHX, aPointList->getHalfSizeX());
				minHY = std::min(minHY, aPointList->getHalfSizeY());
				minHZ = std::min(minHZ, aPointList->getHalfSizeZ());

			}
		} else {
			KdTree* left = (KdTree*) leftChild;
			KdTree* right = (KdTree*) rightChild;
			minLight = std::min(left->minLightIntensity, right->minLightIntensity);
			maxCos = std::max(left->maxConeCos, right->maxConeCos);
			minHX = std::min(left->minHalfSizeX, right->minHalfSizeX);
			minHY = std::min(left->minHalfSizeY, right->minHalfSizeY);
			minHZ = std::min(left->minHalfSizeZ, right->minHalfSizeZ);
		}
		if (minLight != this->minLightIntensity) {
			assert(false&&"bad minimum light intensity");
		}
		if (maxCos != this->maxConeCos) {
			assert(false&&"bad max cone cos");
		}
		if (minHX != this->minHalfSizeX || minHY != this->minHalfSizeY || minHZ != this->minHalfSizeZ) {
			assert(false&&"bad minimum half size");
		}
		return true;
	}
public:
	static KdTree& createTree(std::vector<NodeWrapper*> *& inPoints) {
		KdTree &root = *(KdTree*) subdivide(inPoints, 0, inPoints->size(), NULL, new KdTree());
		return root;
	}
	NodeWrapper* findBestMatch(NodeWrapper *&inLight) {
		//TODO : uncomment if findBestMatch is being called in parallel.
		// right now we only call it in the first process-body, which does
		// not modify the tree.
		//		acquire(this);
		PotentialCluster *cluster = new PotentialCluster(inLight);
		if (splitType == LEAF) {
			findNearestRecursive(cluster);
		} else if (splitType == SPLIT_X) {
			recurse(cluster, inLight->getX());
		} else if (splitType == SPLIT_Y) {
			recurse(cluster, inLight->getY());
		} else if (splitType == SPLIT_Z) {
			recurse(cluster, inLight->getZ());
		} else {
			assert(false&&"Err in findBestMatch");
		}
		return cluster->closest;
	}


};
#endif /* KDTREE_H_ */

/*
 * ClusterNode.h
 *
 *  Created on: Jun 30, 2011
 *      Author: rashid
 */
#ifndef CLUSTERNODE_H_
#define CLUSTERNODE_H_

#include<assert.h>
#include"NodeWrapper.h"
class ClusterNode: public AbstractNode {

private:
	AbstractNode * leftChild;
	AbstractNode * rightChild;
	vector<LeafNode*> *reps;
	float boxRadiusX;
	float boxRadiusY;
	float boxRadiusZ;
	float coneDirX;
	float coneDirY;
	float coneDirZ;
	float coneCos;

public:
	ClusterNode() {
		leftChild = rightChild = NULL;
		reps=NULL;
	}
	friend std::ostream & operator<<(std::ostream & s , ClusterNode & n);

	void setBox(float minX, float maxX, float minY, float maxY, float minZ, float maxZ) {
		x = 0.5f * (minX + maxX);
		y = 0.5f * (minY + maxY);
		z = 0.5f * (minZ + maxZ);
		boxRadiusX = 0.5f * (maxX - minX);
		boxRadiusY = 0.5f * (maxY - minY);
		boxRadiusZ = 0.5f * (maxZ - minZ);
	}

	void setChildren(AbstractNode & inLeft, AbstractNode &inRight, double repRandomNum) {
		leftChild = &inLeft;
		rightChild = &inRight;
		setSummedIntensity(*leftChild, *rightChild);
		setCombinedFlags(*leftChild, *rightChild);
		//we only apply clamping to nodes that are low in the tree
		vector<float> ranVec = (*repRandomNums)[(int) (repRandomNum * repRandomNums->size())];
		if (globalMultitime) {
			int numReps = endTime - startTime + 1;
			if (reps == NULL || ((int) reps->size()) < numReps) {
				reps = new vector<LeafNode*> (numReps);
			} else {
				for (int j = numReps; j < (int) reps->size(); j++) {
					(*reps)[j] = NULL;
				} //fill unused values will nulls
			}
			if (leftChild->isLeaf()) {
				LeafNode & leftLeaf = *((LeafNode*) leftChild);
				if (rightChild->isLeaf()) {
					chooseRepsWithTime(*reps, *this, ranVec, leftLeaf, *(LeafNode*) rightChild);
				} else {
					chooseRepsWithTime(*reps, *this, ranVec, *((ClusterNode*) rightChild), leftLeaf); //note: operation is symmectric so we just interchange the children in the call
				}
			} else {
				ClusterNode &leftClus = *(ClusterNode*) leftChild;
				if (rightChild->isLeaf()) {
					chooseRepsWithTime(*reps, *this, ranVec, leftClus, *(LeafNode*) rightChild);
				} else {
					chooseRepsWithTime(*reps, *this, ranVec, leftClus, *(ClusterNode*) rightChild);
				}
			}
		} else {
			if (reps == NULL || ((int) reps->size()) != globalNumReps) {
				reps = new vector<LeafNode*> (globalNumReps);
			}
			if (leftChild->isLeaf()) {
				LeafNode &leftLeaf = *(LeafNode*) leftChild;
				if (rightChild->isLeaf()) {
					chooseRepsNoTime(*reps, *this, ranVec, leftLeaf, *(LeafNode*) rightChild);
				} else {
					chooseRepsNoTime(*reps, *this, ranVec, *(ClusterNode*) rightChild, leftLeaf); //note: operation is symmectric so we just interchange the children in the call
				}
			} else {
				ClusterNode &leftClus = *(ClusterNode*) leftChild;
				if (rightChild->isLeaf()) {
					chooseRepsNoTime(*reps, *this, ranVec, leftClus, *(LeafNode*) rightChild);
				} else {
					chooseRepsNoTime(*reps, *this, ranVec, leftClus, *(ClusterNode*) rightChild);
				}
			}
		}
	}

	static void chooseRepsNoTime(vector<LeafNode*> &repArr, AbstractNode &parent, vector<float> & ranVec, LeafNode &left, LeafNode &right) {
		float totalInten = parent.getScalarTotalIntensity();
		float leftInten = left.getScalarTotalIntensity();
		float nextTest = ranVec[0] * totalInten;
		for (int i = 0; i < (int) repArr.size() - 1; i++) {
			float test = nextTest;
			nextTest = ranVec[i + 1] * totalInten;
			(repArr)[i] = (test < leftInten) ? &left : &right;
		}
		(repArr)[repArr.size() - 1] = (nextTest < leftInten) ? &left : &right;
	}

	static void chooseRepsNoTime(vector<LeafNode*>&repArr, AbstractNode &parent, vector<float> &ranVec, ClusterNode &left, LeafNode &right) {
		float totalInten = parent.getScalarTotalIntensity();
		float leftInten = left.getScalarTotalIntensity();
		float nextTest = ranVec[0] * totalInten;
		for (int i = 0; i < (int) repArr.size() - 1; i++) {
			float test = nextTest;
			nextTest = ranVec[i + 1] * totalInten;
			repArr[i] = (test < leftInten) ? ((*left.reps)[i]) : &right;
		}
		repArr[repArr.size() - 1] = (nextTest < leftInten) ? ((*left.reps)[repArr.size() - 1]) : &right;
	}

	static void chooseRepsNoTime(vector<LeafNode*> &repArr, AbstractNode &parent, vector<float> &ranVec, ClusterNode &left,
			ClusterNode &right) {
		float totalInten = parent.getScalarTotalIntensity();
		float leftInten = left.getScalarTotalIntensity();
		float nextTest = ranVec[0] * totalInten;
		for (int i = 0; i < (int) repArr.size() - 1; i++) {
			float test = nextTest;
			nextTest = ranVec[i + 1] * totalInten;
			repArr[i] = (test < leftInten) ? ((*left.reps)[i]) : ((*right.reps)[i]);
		}
		repArr[repArr.size() - 1] = (nextTest < leftInten) ? ((*left.reps)[repArr.size() - 1]) : ((*right.reps)[repArr.size() - 1]);
	}

	static void chooseRepsWithTime(vector<LeafNode*> & repArr, AbstractNode & parent, vector<float> & ranVec, LeafNode & left,
			LeafNode & right) {
		int startTime = (int) parent.getStartTime();
		int endTime = (int) parent.getEndTime();
		float parentTotal = parent.getScalarTotalIntensity();
		float leftTotal = left.getScalarTotalIntensity();
		float nextTest = ranVec[startTime] * parentTotal * parent.getRelativeIntensity(startTime);
		float nextLeftInten = leftTotal * left.getRelativeIntensity(startTime);
		for (int t = startTime; t < endTime; t++) {
			float test = nextTest;
			float leftInten = nextLeftInten;
			nextTest = ranVec[t + 1] * parentTotal * parent.getRelativeIntensity(t + 1);
			nextLeftInten = leftTotal * left.getRelativeIntensity(t + 1);
			if (test == 0) {
				repArr[t - startTime] = NULL;
			} else {
				repArr[t - startTime] = (test < leftInten) ? &left : &right;
			}
		}
		if (nextTest == 0) {
			repArr[endTime - startTime] = NULL;
		} else {
			repArr[endTime - startTime] = (nextTest < nextLeftInten) ? &left : &right;
		}
	}

	static void chooseRepsWithTime(vector<LeafNode*> & repArr, AbstractNode &parent, vector<float> &ranVec, ClusterNode &left,
			LeafNode &right) {
		int startTime = parent.getStartTime();
		int endTime = parent.getEndTime();
		float parentTotal = parent.getScalarTotalIntensity();
		float leftTotal = left.getScalarTotalIntensity();
		float nextTest = ranVec[startTime] * parentTotal * parent.getRelativeIntensity(startTime);
		float nextLeftInten = leftTotal * left.getRelativeIntensity(startTime);
		for (int t = startTime; t < endTime; t++) {
			float test = nextTest;
			float leftInten = nextLeftInten;
			nextTest = ranVec[t + 1] * parentTotal * parent.getRelativeIntensity(t + 1);
			nextLeftInten = leftTotal * left.getRelativeIntensity(t + 1);
			if (test == 0) {
				repArr[t - startTime] = NULL;
			} else {
				repArr[t - startTime] = (test < leftInten) ? ((*left.reps)[t - left.getStartTime()]) : &right;
			}
		}
		if (nextTest == 0) {
			repArr[endTime - startTime] = NULL;
		} else {
			repArr[endTime - startTime] = (nextTest < nextLeftInten) ? ((*left.reps)[endTime - left.getStartTime()]) : &right;
		}
	}

	static void chooseRepsWithTime(vector<LeafNode*>& repArr, AbstractNode &parent, vector<float> &ranVec, ClusterNode &left,
			ClusterNode &right) {
		int startTime = parent.getStartTime();
		int endTime = parent.getEndTime();
		float parentTotal = parent.getScalarTotalIntensity();
		float leftTotal = left.getScalarTotalIntensity();
		float nextTest = ranVec[startTime] * parentTotal * parent.getRelativeIntensity(startTime);
		float nextLeftInten = leftTotal * left.getRelativeIntensity(startTime);
		for (int t = startTime; t < endTime; t++) {
			float test = nextTest;
			float leftInten = nextLeftInten;
			nextTest = ranVec[t + 1] * parentTotal * parent.getRelativeIntensity(t + 1);
			nextLeftInten = leftTotal * left.getRelativeIntensity(t + 1);
			if (test == 0) {
				repArr[t - startTime] = NULL;
			} else {
				repArr[t - startTime] = (test < leftInten) ? ((*left.reps)[t - left.getStartTime()]) : ((*right.reps)[t
						- right.getStartTime()]);
			}
		}
		if (nextTest == 0) {
			repArr[endTime - startTime] = NULL;
		} else {
			repArr[endTime - startTime] = (nextTest < nextLeftInten) ? ((*left.reps)[endTime - left.getStartTime()])
					: ((*right.reps)[endTime - right.getStartTime()]);
		}
	}
public:
	float getBoxRadiusX() {
		return boxRadiusX;
	}

	float getBoxRadiusY() {
		return boxRadiusY;
	}

	float getBoxRadiusZ() {
		return boxRadiusZ;
	}

	void setDirectionCone(float dirX, float dirY, float dirZ, float inConeCos) {
		coneDirX = dirX;
		coneDirY = dirY;
		coneDirZ = dirZ;
		coneCos = inConeCos;
	}

	float getConeDirX() {
		return coneDirX;
	}

	float getConeDirY() {
		return coneDirY;
	}

	float getConeDirZ() {
		return coneDirZ;
	}

	float getConeCos() {
		return coneCos;
	}

	bool isLeaf() {
		return false;
	}

	int size() {
		// only leafs are counted
		return leftChild->size() + rightChild->size();
	}

	bool equals(ClusterNode & other) {
		return leftChild->equals(*other.leftChild) && rightChild->equals(*other.rightChild);
	}

	void findConeDirsRecursive(vector<float> & fArr, vector<ClusterNode*>& cArr) {
		//TODO Use NodeWrapper::CONE here
		findConeDirsRecursive(*leftChild, fArr, 0, cArr, 4- 1);
		findConeDirsRecursive(*rightChild, fArr, 0, cArr, 4- 1);
	}

private:
	static int findConeDirsRecursive(AbstractNode &node, vector<float>& fArr, int numDirs, vector<ClusterNode*>& cArr, int recurseDepth) {
		if (!node.isLeaf()) {
			ClusterNode & clus = (ClusterNode&) node;
			if (clus.coneCos == 1.0) {
				numDirs = addConeDir(fArr, numDirs, clus.coneDirX, clus.coneDirY, clus.coneDirZ);
			} else if (recurseDepth <= 0) {
				//find first empty slot and add this cluster there
				for (int i = 0;; i++) {
					if (cArr[i] == NULL) {
						cArr[i] = &clus;
						if (cArr[i + 1] != NULL) {
							assert(false && "throw new RuntimeException()");
						}
						break;
					}
				}
			} else {
				numDirs = findConeDirsRecursive(*clus.leftChild, fArr, numDirs, cArr, recurseDepth - 1);
				numDirs = findConeDirsRecursive(*clus.rightChild, fArr, numDirs, cArr, recurseDepth - 1);
			}
		} else {
			LeafNode &light = (LeafNode&) node;
			numDirs = addConeDir(fArr, numDirs, light.getDirX(), light.getDirY(), light.getDirZ());
		}
		return numDirs;
	}

private:
	static int addConeDir(vector<float>& fArr, int numDirs, float x, float y, float z) {
		//only add direction if it does not match any existing directions
		for (int i = 0; i < 3 * numDirs; i++) {
			if ((fArr[i] == x) && (fArr[i + 1] == y) && (fArr[i + 2] == z)) {
				return numDirs;
			}
		}
		int index = 3 * numDirs;
		fArr[index] = x;
		fArr[index + 1] = y;
		fArr[index + 2] = z;
		return numDirs + 1;
	}
};
std::ostream & operator<<(std::ostream & s , ClusterNode & n){
		s<<"ClusterNode box:["<<n.boxRadiusX<<","<<n.boxRadiusY<<","<<n.boxRadiusZ<<"], Cone:["<<n.coneDirX<<","<<n.coneDirY<<","<<n.coneDirZ<<"]";
		if(n.leftChild!=NULL)
		s<<"L:"<<(*n.leftChild);
		if(n.rightChild!=NULL)
				s<<"L:"<<(*n.rightChild);
		return s;
	}

#endif /* CLUSTERNODE_H_ */

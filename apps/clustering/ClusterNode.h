/*
 * ClusterNode.h
 *
 *  Created on: Jun 20, 2011
 *      Author: rashid
 */
#include<assert.h>
#include"NodeWrapper.h"
#ifndef CLUSTERNODE_H_
#define CLUSTERNODE_H_
class ClusterNode: public AbstractNode {

private:
	AbstractNode* leftChild;
	AbstractNode* rightChild;
	std::vector<LeafNode*> * reps;
	float coneDirX;
	float coneDirY;
	float coneDirZ;
	float coneCos;
public:
	ClusterNode() {
		//TODO Fix this!
		//		leftChild=rightChild = *(new LeafNode(0,0,0,0,0,0));
	}

	void setBox(float minX, float maxX, float minY, float maxY, float minZ,
			float maxZ) {
		x = 0.5f * (minX + maxX);
		y = 0.5f * (minY + maxY);
		z = 0.5f * (minZ + maxZ);
	}

	void setChildren(AbstractNode* inLeft, AbstractNode* inRight,
			double repRandomNum) {
		leftChild = inLeft;
		rightChild = inRight;
		setSummedIntensity(*leftChild, *rightChild);
		setCombinedFlags(*leftChild, *rightChild);
		//we only apply clamping to nodes that are low in the tree
		std::vector<float> ranVec = repRandomNums[(int) (repRandomNum
				* repRandomNums.size())];
		if (globalMultitime) {
			assert(false&& " Should not be here, global multitime is true...");
		} else {
			if (reps == NULL || ((int)reps->size()) != globalNumReps) {
				reps = new std::vector<LeafNode*>(globalNumReps);
			}
			if (leftChild->isLeaf()) {
				LeafNode* leftLeaf = (LeafNode*) leftChild;
				if (rightChild->isLeaf()) {
					chooseRepsNoTime(reps, this, ranVec, leftLeaf,
							(LeafNode*) rightChild);
				} else {
					chooseRepsNoTime(reps, this, ranVec,
							(ClusterNode*) rightChild, leftLeaf); //note: operation is symmectric so we just interchange the children in the call
				}
			} else {
				ClusterNode* leftClus = (ClusterNode*) leftChild;
				if (rightChild->isLeaf()) {
					chooseRepsNoTime(reps, this, ranVec, leftClus,
							(LeafNode*) rightChild);
				} else {
					chooseRepsNoTime(reps, this, ranVec, leftClus,
							(ClusterNode*) rightChild);
				}
			}
		}
	}
private:
	static void chooseRepsNoTime(std::vector<LeafNode*> * repArr,
			AbstractNode* parent, std::vector<float> ranVec, LeafNode *left,
			LeafNode *right) {
		float totalInten = parent->getScalarTotalIntensity();
		float leftInten = left->getScalarTotalIntensity();
		float nextTest = (ranVec)[0] * totalInten;
		for (int i = 0; i < (int)repArr->size() - 1; i++) {
			float test = nextTest;
			nextTest = ranVec[i + 1] * totalInten;
			(*repArr)[i] = (test < leftInten) ? left : right;
		}
		(*repArr)[repArr->size() - 1] = (nextTest < leftInten) ? left : right;
	}

	static void chooseRepsNoTime(std::vector<LeafNode*> * repArr,
			AbstractNode* parent, std::vector<float> ranVec, ClusterNode* left,
			LeafNode* right) {
		float totalInten = parent->getScalarTotalIntensity();
		float leftInten = left->getScalarTotalIntensity();
		float nextTest = ranVec[0] * totalInten;
		for (int i = 0; i < (int)repArr->size() - 1; i++) {
			float test = nextTest;
			nextTest = ranVec[i + 1] * totalInten;
			(*repArr)[i] = (test < leftInten) ? ((*(left->reps))[i]) : right;
		}
		(*repArr)[repArr->size() - 1]
				= (nextTest < leftInten) ? ((*(left->reps))[repArr->size() - 1])
						: (right);
	}

	static void chooseRepsNoTime(std::vector<LeafNode*> *repArr,
			AbstractNode* parent, std::vector<float> ranVec, ClusterNode* left,
			ClusterNode* right) {
		float totalInten = parent->getScalarTotalIntensity();
		float leftInten = left->getScalarTotalIntensity();
		float nextTest = ranVec[0] * totalInten;
		for (int i = 0; i < (int)repArr->size() - 1; i++) {
			float test = nextTest;
			nextTest = ranVec[i + 1] * totalInten;
			(*repArr)[i] = (test < leftInten) ? ((*(left->reps))[i])
					: ((*(right->reps))[i]);
		}
		(*repArr)[repArr->size() - 1]
				= (nextTest < leftInten) ? ((*(left->reps))[repArr->size() - 1])
						: ((*(right->reps))[repArr->size() - 1]);
	}

public:
	void setDirectionCone(float dirX, float dirY, float dirZ, float inConeCos) {
		coneDirX = dirX;
		coneDirY = dirY;
		coneDirZ = dirZ;
		coneCos = inConeCos;
	}

	//TODO Fix this!
	void findConeDirsRecursive(std::vector<float> &fArr, std::vector<ClusterNode*> &cArr) {
//		findConeDirsRecursive(leftChild, fArr, 0, cArr,NodeWrapper::CONE_RECURSE_DEPTH - 1);
//		findConeDirsRecursive(rightChild, fArr, 0, cArr,NodeWrapper::CONE_RECURSE_DEPTH - 1);
		findConeDirsRecursive(leftChild, fArr, 0, cArr,4- 1);
		findConeDirsRecursive(rightChild, fArr, 0, cArr,4- 1);
	}
private:
	static int findConeDirsRecursive(AbstractNode* node,
			std::vector<float>& fArr, int numDirs,
			std::vector<ClusterNode*>& cArr, int recurseDepth) {
		if (!node->isLeaf()) {
			ClusterNode * clus = (ClusterNode*) node;
			if (clus->coneCos == 1.0) {
				numDirs = addConeDir(fArr, numDirs, clus->coneDirX,
						clus->coneDirY, clus->coneDirZ);
			} else if (recurseDepth <= 0) {
				//find first empty slot and add this cluster there
				for (int i = 0;; i++) {
					if (cArr[i] == NULL) {
						cArr[i] = clus;
						if (cArr[i + 1] != NULL) {
							//throw new RuntimeException();
							std::cout << "Error in findConeDirsRecursive"
									<< std::endl;
						}
						break;
					}
				}
			} else {
				numDirs = findConeDirsRecursive(clus->leftChild, fArr, numDirs,
						cArr, recurseDepth - 1);
				numDirs = findConeDirsRecursive(clus->rightChild, fArr,
						numDirs, cArr, recurseDepth - 1);
			}
		} else {
			LeafNode* light = (LeafNode*) node;
			numDirs = addConeDir(fArr, numDirs, light->getDirX(),
					light->getDirY(), light->getDirZ());
		}
		return numDirs;
	}

	static int addConeDir(std::vector<float> &fArr, int numDirs, float x,
			float y, float z) {
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
public:
	bool isLeaf() {
		return false;
	}

	int size() {
		// only leafs are counted
		return leftChild->size() + rightChild->size();
	}
	//TODO : implement this.
//	bool equals(void *obj) {
//		ClusterNode * other = dynamic_cast<ClusterNode*> (obj);
//		if (other != NULL) {
//			return leftChild->equals(other->leftChild) && rightChild->equals(
//					other->rightChild);
//		}
//		return false;
//	}
};
#endif /* CLUSTERNODE_H_ */

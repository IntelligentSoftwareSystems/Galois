/*
 * NodeWrapper.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include"Box3d.h"
#include"ClusterNode.h"
#include<vector>
#include<math.h>

#ifndef NODEWRAPPER_H_
#define NODEWRAPPER_H_
class NodeWrapper : public Box3d {

public:

	static const int CONE_RECURSE_DEPTH;// = 4;
	static const double GLOBAL_SCENE_DIAGONAL;// = 2.0;

	//The position bounding box is stored in the inherited min/max fields
	//If light directions are relevant then they are stored in the dirBox and coneCos

	/**
	 * Wrapper for lights or clusters so we can put them into a kd-tree for fast searching for their
	 * best clustering candidate
	 */
	//light or cluster
	/*const */AbstractNode * light;
	//bounding box of light directions on the unit sphere
	/*const */float coneCos;
	/*const */float x;
	/*const */float y;
	/*const */float z;
	//objects to be used when computing direction cone angles
private:
	Box3d * dirBox;
	//cosine of cone of light directions
	std::vector<float> *coneDirs;
	std::vector<ClusterNode*> *coneClusters;
	/*const */int descendents;
	long rid;

public:
	NodeWrapper(ClusterNode *inNode):light(inNode),x(inNode->getX()),y(inNode->getY()),z(inNode->getZ()),descendents(inNode->size()) {
		coneDirs = NULL;
		coneClusters = NULL;
		//light = inNode;
		//x = inNode.getX();
		//y = inNode.getY();
		//z = inNode.getZ();
		xMin = x - inNode->getBoxRadiusX();
		xMax = x + inNode->getBoxRadiusX();
		yMin = y - inNode->getBoxRadiusY();
		yMax = y + inNode->getBoxRadiusY();
		zMin = z - inNode->getBoxRadiusZ();
		zMax = z + inNode->getBoxRadiusZ();
		coneCos = inNode->getConeCos();
		if (coneCos != 1.0) {
			//throw new RuntimeException("not yet implemented");
			std::cout<<"Error, not implemented yet!"<<std::endl;
		}
		if (dirBox == NULL) {
			dirBox = new Box3d();
		}
		dirBox->addPoint(inNode->getConeDirX(), inNode->getConeDirY(), inNode->getConeDirZ());
		coneDirs = new std::vector<float>(3);
		//coneDirs.resize(3);/// = new float[3];
		(*coneDirs)[0] = inNode->getConeDirX();
		(*coneDirs)[1] = inNode->getConeDirY();
		(*coneDirs)[2] = inNode->getConeDirZ();
	}

	NodeWrapper(LeafNode *inNode):light(inNode) {
		coneDirs = NULL;
		coneClusters = NULL;
		descendents = 1;
		setBox(inNode->getX(), inNode->getY(), inNode->getZ());
		if (dirBox == NULL) {
			dirBox = new Box3d();
		}
		dirBox->setBox(inNode->getDirX(), inNode->getDirY(), inNode->getDirZ());
		coneCos = 1.0f;
		coneDirs = new std::vector<float>(3);
		//coneDirs.resize(3);// = new float[3];
		(*coneDirs)[0] = inNode->getDirX();
		(*coneDirs)[1] = inNode->getDirY();
		(*coneDirs)[2] = inNode->getDirZ();
		x = 0.5f * (xMax + xMin);
		y = 0.5f * (yMax + yMin);
		z = 0.5f * (zMax + zMin);
	}

	NodeWrapper(NodeWrapper* leftChild, NodeWrapper *rightChild, std::vector<float> tempFloatArr, std::vector<ClusterNode*> tempClusterArr) :light(new ClusterNode()){
		if ((leftChild->x > rightChild->x) || ((leftChild->x == rightChild->x) && (leftChild->y > rightChild->y))
				|| ((leftChild->x == rightChild->x) && (leftChild->y == rightChild->y) && (leftChild->z > rightChild->z))) {
			//swap them to make sure we are consistent about which node becomes the left and the right
			//this makes the trees easier to compare for equivalence when checking against another building method
			NodeWrapper *temp = leftChild;
			leftChild = rightChild;
			rightChild = temp;
		}
		addBox(*rightChild);
		addBox(*leftChild);
		x = 0.5f * (xMax + xMin);
		y = 0.5f * (yMax + yMin);
		z = 0.5f * (zMax + zMin);
		descendents = leftChild->descendents + rightChild->descendents;
		//create new cluster
		ClusterNode * cluster = new ClusterNode();
		cluster->setBox(xMin, xMax, yMin, yMax, zMin, zMax);
		cluster->setChildren((leftChild->light), (rightChild->light), rand()/(rand()+1));
		light = cluster;
		//compute direction cone and set it in the cluster
		coneCos = computeCone(leftChild, rightChild, cluster);
		if (coneCos > -0.9f) {
			dirBox = new Box3d();
			dirBox->addBox(* (leftChild->dirBox));
			dirBox->addBox(* (rightChild->dirBox));
			cluster->findConeDirsRecursive(tempFloatArr, tempClusterArr);
			int numClus = 0;
			for (; tempClusterArr[numClus] != NULL; numClus++) {
			}
			if (numClus > 0) {
				coneClusters->resize(numClus);// = new ClusterNode[numClus];
				for (int j = 0; j < numClus; j++) {
					(*coneClusters)[j] = (tempClusterArr[j]);
					tempClusterArr[j] = NULL;
				}
			}
		}
	}

	static float computeCone(NodeWrapper* a, NodeWrapper* b, ClusterNode* outCluster) {
		if (a->dirBox == NULL || b->dirBox == NULL) {
			return -1.0f;
		}
		//we use the circumscribed sphere around the dirBox to compute a conservative bounding cone
		float xMin = a->dirBox->xMin >= b->dirBox->xMin ? b->dirBox->xMin : a->dirBox->xMin;
		float yMin = a->dirBox->yMin >= b->dirBox->yMin ? b->dirBox->yMin : a->dirBox->yMin;
		float zMin = a->dirBox->zMin >= b->dirBox->zMin ? b->dirBox->zMin : a->dirBox->zMin;
		float xMax = a->dirBox->xMax >= b->dirBox->xMax ? a->dirBox->xMax : b->dirBox->xMax;
		float yMax = a->dirBox->yMax >= b->dirBox->yMax ? a->dirBox->yMax : b->dirBox->yMax;
		float zMax = a->dirBox->zMax >= b->dirBox->zMax ? a->dirBox->zMax : b->dirBox->zMax;

		float rad2 = ((xMax - xMin) * (xMax - xMin) + (yMax - yMin) * (yMax - yMin) + (zMax - zMin) * (zMax - zMin));
		float coneX = (xMin + xMax);
		float coneY = (yMin + yMax);
		float coneZ = (zMin + zMax);
		float center2 = coneX * coneX + coneY * coneY + coneZ * coneZ;
		if (center2 < 0.01) {
			return -1.0f; //too large to produce any reasonable cone smaller than the whole sphere
		}
		float invLen = 1.0f / (float) sqrt(center2);
		//given the unit sphere and another sphere width radius^2 (rad2) and center^2 (center2)
		//we can compute the cos(angle) cone defined by its intersection with the unit sphere
		//note we actually keep 4*cetner2 and 4*rad2 to reduce number of multipications needed
		float minCos = (center2 + 4.0f - rad2) * 0.25f * invLen;
		if (minCos < -1.0f) {
			minCos = -1.0f;
		}
		if (outCluster != NULL) {
			outCluster->setDirectionCone(coneX * invLen, coneY * invLen, coneZ * invLen, minCos);
		}
		return minCos;
//		return 0;
	}

	float getX() {
		return x;
	}

	float getY() {
		return y;
	}

	float getZ() {
		return z;
	}

	float getHalfSizeX() {
		return xMax - x;
	}

	float getHalfSizeY() {
		return yMax - y;
	}

	float getHalfSizeZ() {
		return zMax - z;
	}

	static double potentialClusterSize(NodeWrapper* a, NodeWrapper* b) {
		float dx = (a->xMax >= b->xMax ? a->xMax : b->xMax) - (a->xMin >= b->xMin ? b->xMin : a->xMin);
		float dy = (a->yMax >= b->yMax ? a->yMax : b->yMax) - (a->yMin >= b->yMin ? b->yMin : a->yMin);
		float dz = (a->zMax >= b->zMax ? a->zMax : b->zMax) - (a->zMin >= b->zMin ? b->zMin : a->zMin);
		float minCos = computeCone(a, b, NULL);
		float maxIntensity = a->light->getScalarTotalIntensity() + b->light->getScalarTotalIntensity();
		return clusterSizeMetric(dx, dy, dz, minCos, maxIntensity);
//		return 0;
	}

	/**
	 * Compute a measure of the size of a light cluster
	 */
	static double clusterSizeMetric(double xSize, double ySize, double zSize, double cosSemiAngle, double intensity) {
		double len2 = xSize * xSize + ySize * ySize + zSize * zSize;
		double angleFactor = (1 - cosSemiAngle) * GLOBAL_SCENE_DIAGONAL;
		return intensity * (len2 + angleFactor * angleFactor);
//		return 0;
	}
};
const int NodeWrapper::CONE_RECURSE_DEPTH = 4;
const double NodeWrapper::GLOBAL_SCENE_DIAGONAL = 2.0;
#endif /* NODEWRAPPER_H_ */

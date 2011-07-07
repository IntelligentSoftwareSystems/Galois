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

#include "AbstractNode.h"
#ifndef LEAFNODE_H_
#define LEAFNODE_H_
#define MATH_PI 3.14159

class LeafNode: public AbstractNode {
protected:
	//direction of maximum emission
	/*const */
	float dirX;
	/*const */
	float dirY;
	/*const */
	float dirZ;

	/**
	 * Creates a new instance of MLTreeLeafNode
	 */
public:
	LeafNode(float x, float y, float z, float px, float py, float pz) {
		this->x = x;
		this->y = y;
		this->z = z;
		setIntensity(1.0 / MATH_PI, (short) 0);
		dirX = px;
		dirY = py;
		dirZ = pz;
	}

	float getDirX() {
		return dirX;
	}

	float getDirY() {
		return dirY;
	}

	float getDirZ() {
		return dirZ;
	}

	bool isLeaf() {
		return true;
	}

	int size() {
		return 1;
	}
	/*bool operator==(LeafNode & other) {
		return x == other.x && y == other.y && z == other.z && dirX
				== other.dirX && dirY == other.dirY && dirZ == other.dirZ;
	}*/
	bool isEqual(LeafNode * other) {
			return x == other->x && y == other->y && z == other->z && dirX
					== other->dirX && dirY == other->dirY && dirZ == other->dirZ;
		}
	friend std::ostream & operator<<(std::ostream& s, LeafNode & l);
};
std::ostream & operator<<(std::ostream& s, LeafNode & l) {
	s << "Leaf node [" << l.x << "," << l.y << "," << l.z << "]";
	return s;
}
#endif /* LEAFNODE_H_ */

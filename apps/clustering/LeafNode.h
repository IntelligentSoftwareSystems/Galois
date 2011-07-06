/*
 * LeafNode.h
 *
 *  Created on: Jun 30, 2011
 *      Author: rashid
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

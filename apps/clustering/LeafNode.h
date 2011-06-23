/*
 * LeafNode.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */

#include "AbstractNode.h"
#ifndef LEAFNODE_H_
#define LEAFNODE_H_
#define MATH_PI 3.14159

class LeafNode :public AbstractNode{
private:
  //direction of maximum emission
  /*const */float dirX;
  /*const */float dirY;
  /*const */float dirZ;

  /**
   * Creates a new instance of MLTreeLeafNode
   */
public:
  LeafNode(float x, float y, float z, float px, float py, float pz):dirX(px),dirY(py),dirZ(pz) {
    this->x = x;
    this->y = y;
    this->z = z;
    setIntensity(1.0 / MATH_PI, (short) 0);
//    std::cout<<"Creating Leaf node ["<<x<<","<<y<<","<<z<<"]"<<std::endl;
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
  friend std::ostream & operator<<(std::ostream& s, LeafNode & l);
};
std::ostream & operator<<(std::ostream& s, LeafNode & l){
	s<<"Leaf node ["<<l.x<<","<<l.y<<","<<l.z<<"]"<<std::endl;
	return s;
}
#endif /* LEAFNODE_H_ */

/*
 * Box3d.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include<stdlib.h>
#include<limits>
#include<iostream>

#ifndef BOX3D_H_
#define BOX3D_H_
class Box3d {
public:
  float xMin;
  float xMax;
  float yMin;
  float yMax;
  float zMin;
  float zMax;

  /**
   * Creates a new instance of TreeBuilderFloatBox
   */
  Box3d() {
    xMin=yMin=zMin=std::numeric_limits<float>::max();
    xMax=yMax=zMax=std::numeric_limits<float>::min();//Float.MAX_VALUE;
  }

  void setBox(float x, float y, float z) {
    xMin = xMax = x;
    yMin = yMax = y;
    zMin = zMax = z;
  }

  void addBox(const Box3d box) {
    xMin = xMin >= box.xMin ? box.xMin 	: xMin;
    xMax = xMax >= box.xMax ? xMax 		: box.xMax;
    yMin = yMin >= box.yMin ? box.yMin 	: yMin;
    yMax = yMax >= box.yMax ? yMax 		: box.yMax;
    zMin = zMin >= box.zMin ? box.zMin 	: zMin;
    zMax = zMax >= box.zMax ? zMax 		: box.zMax;
  }
  friend std::ostream& operator<<(std::ostream& s, Box3d & b);

} ;
std::ostream& operator<<(std::ostream& s, Box3d & b){
	s<<"Box :["<<b.xMin<<","<<b.yMin<<","<<b.zMin<<"]-["<<b.xMax<<","<<b.yMax<<","<<b.zMax<<"]";
	return s;
}
#endif /* BOX3D_H_ */

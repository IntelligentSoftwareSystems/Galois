/** A box -*- C++ -*-
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
#ifndef BOX3D_H_
#define BOX3D_H_

#include <stdlib.h>
#include <limits>

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

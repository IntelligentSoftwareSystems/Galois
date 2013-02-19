/** Various objects and the scene that contains them -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef SCENE_H
#define SCENE_H

#include <QGLViewer/vec.h>
#include <vector>

class Ball {
public:
  qglviewer::Vec pos;
  qglviewer::Vec vel;

  template<typename T1, typename T2>
  Ball(T1&& p, T2&& v): pos(std::forward<T1>(p)), vel(std::forward<T2>(v)) { }

  void draw();
};

class Lines {
  std::vector<GLfloat> vertices;
  std::vector<GLushort> indices;

public:
  void init(const std::vector<Ball>& balls);
  void draw();
};

class Scene {
  std::vector<Ball> balls;
  Lines lines;
  float radius;
  float deltaT;

  /**
   * Returns if p is in scene box. If p is not in the scene
   * update normal vector of box side that was violated.
   */
  bool inScene(const qglviewer::Vec& p, qglviewer::Vec& n) {
    for (int k = 0; k < 3; ++k) {
      if (p[k] < -radius) {
        n = qglviewer::Vec();
        n[k] = 1;
        return false;
      }
      if (p[k] > radius) {
        n = qglviewer::Vec();
        n[k] = -1;
        return false;
      }
    }
    return true;
  }

public:
  void init(int numBalls, float r, float dt = 0.1);
  void animate();
  void draw();
};

#endif

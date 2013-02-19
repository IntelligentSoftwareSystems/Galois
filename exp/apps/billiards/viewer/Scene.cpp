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
#include <GL/gl.h>
#include <cmath>

#include "Scene.h"

namespace {
class Sphere {
protected:
  std::vector<GLfloat> vertices;
  std::vector<GLfloat> normals;
  std::vector<GLushort> indices;

public:
  Sphere(const float radius, const unsigned int rings, const unsigned int sectors) {
    float R = 1.0/(rings-1);
    float S = 1.0/(sectors-1);

    vertices.resize(rings * sectors * 3);
    normals.resize(rings * sectors * 3);

    auto v = vertices.begin();
    auto n = normals.begin();
    for (unsigned int r = 0; r < rings; ++r) {
      for (unsigned int s = 0; s < sectors; ++s) {
        float y = sin(-M_PI_2 + M_PI * r * R);
        float x = cos(2*M_PI* s * S) * sin(M_PI * r * R);
        float z = sin(2*M_PI* s * S) * sin(M_PI * r * R);

        *v++ = x * radius;
        *v++ = y * radius;
        *v++ = z * radius;
        *n++ = x;
        *n++ = y;
        *n++ = z;
      }
    }

    indices.resize(rings * sectors * 4);
    auto i = indices.begin();
    for (unsigned int r = 0; r < rings - 1; ++r) {
      for (unsigned int s = 0; s < sectors - 1; ++s) {
        *i++ = r * sectors + s;
        *i++ = r * sectors + (s+1);
        *i++ = (r+1) * sectors + (s+1);
        *i++ = (r+1) * sectors + s;
      }
    }
  }

  void draw();
};

  
void Sphere::draw() {
  // TODO: Switch over to using shader programs
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
  glNormalPointer(GL_FLOAT, 0, &normals[0]);
  glDrawElements(GL_QUADS, indices.size(), GL_UNSIGNED_SHORT, &indices[0]);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
}
}

void Ball::draw() {
  static Sphere sphere(0.24, 24, 48);
  glColor3f(0.5f, 0.5f, 0.0f);
  glPushMatrix();
  glTranslatef(pos[0], pos[1], pos[2]);
  sphere.draw();
  glPopMatrix();
}

void Lines::init(const std::vector<Ball>& balls) {
  vertices.resize(balls.size() * 3 * 2);
  indices.resize(balls.size() * 2);
  
  auto v = vertices.begin();
  auto i = indices.begin();
  unsigned int index = 0;
  
  for (const Ball& b : balls) {
    qglviewer::Vec off = b.pos + b.vel;
    for (int k = 0; k < 3; ++k)
      *v++ = b.pos[k];
    for (int k = 0; k < 3; ++k)
      *v++ = off[k];
    for (int k = 0; k < 2; ++k)
      *i++ = index++;
  }
}

void Lines::draw() {
  glDisable(GL_LIGHTING);
  glColor3f(1.0f, 1.0f, 1.0f);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
  glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_SHORT, &indices[0]);
  glDisableClientState(GL_VERTEX_ARRAY);
  glEnable(GL_LIGHTING);
}

void Scene::init(int numBalls, float r, float dt) {
  deltaT = dt;
  radius = r;

  int maxX = radius;
  int maxY = radius;

  for (int x = 0; x <= maxX; ++x) {
    for (int y = 0; y <= maxY; ++y) {
      float v = (x+y)/((float) maxX+maxY);
      float vx = cos(2*M_PI*v);
      float vy = sin(2*M_PI*v);
      float px = x - maxX/2;
      float py = y - maxY/2;
      balls.emplace_back(qglviewer::Vec(px, py, 0.0f), qglviewer::Vec(vx, vy, 0.0f));
    }
  }

  lines.init(balls);
}

void Scene::draw() {
  for (Ball& b : balls) {
    b.draw();
  }

  lines.draw();
}

void Scene::animate() {
  for (Ball& b : balls) {
    b.pos += b.vel * deltaT;
    qglviewer::Vec n;

    if (!inScene(b.pos, n)) {
      b.pos -= b.vel * deltaT;
      b.vel = -2 * (b.vel * n) * n + b.vel;
      b.pos += b.vel * deltaT;
    }
  }

  lines.init(balls);
}

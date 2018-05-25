/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "Scene.h"
#include "Viewer.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <cmath>
#include <fstream>
#include <limits>
#include <algorithm>

//! Project world coordinates to screen based on current GL 
class WorldScreenProjector {
  GLint viewport[4];
  GLdouble mvp[16];

public:
  WorldScreenProjector(qglviewer::Camera* c) {
    glGetIntegerv(GL_VIEWPORT, viewport);
    c->getModelViewProjectionMatrix(mvp);
  }

  qglviewer::Vec project(const qglviewer::Vec& p) {
    GLdouble v[4], vs[4];
    // vs = MVP * V
    v[0] = p[0]; v[1] = p[1]; v[2] = p[2]; v[3] = 1.0;
    vs[0] = mvp[0]*v[0] + mvp[4]*v[1] + mvp[ 8]*v[2] + mvp[12]*v[3];
    vs[1] = mvp[1]*v[0] + mvp[5]*v[1] + mvp[ 9]*v[2] + mvp[13]*v[3];
    vs[2] = mvp[2]*v[0] + mvp[6]*v[1] + mvp[10]*v[2] + mvp[14]*v[3];
    vs[3] = mvp[3]*v[0] + mvp[7]*v[1] + mvp[11]*v[2] + mvp[15]*v[3];

    // Projection to window 
    // see http://www.opengl.org/sdk/docs/man2/xhtml/gluProject.xml
    vs[0] /= vs[3];
    vs[1] /= vs[3];
    vs[2] /= vs[3];
    vs[0] = vs[0] * 0.5 + 0.5;
    vs[1] = vs[1] * 0.5 + 0.5;
    vs[2] = vs[2] * 0.5 + 0.5;
    vs[0] = vs[0] * viewport[2] + viewport[0];
    vs[1] = vs[1] * viewport[3] + viewport[1];

    return qglviewer::Vec(vs[0], viewport[3] - vs[1], vs[2]);
  }
};

// -----------------------------------------------
// Sphere
// -----------------------------------------------

void Sphere::init(float radius, float rings, float sectors) {
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

// -----------------------------------------------
// Balls
// -----------------------------------------------

void Balls::init(size_t numBalls, float ballRadius) {
  sphere.init(ballRadius, 24, 48);

  balls.resize(numBalls);
  colors.resize(numBalls * 3);
  auto c = colors.begin();
  for (size_t i = 0; i < numBalls; ++i) {
    for (size_t k = 0; k < 3; ++k)
      *c++ = rand() / static_cast<float>(RAND_MAX);
  }
}

void Balls::draw(Viewer* v) {
  auto c = colors.begin();
  for (Ball& b : balls) {
    glColor3f(*c++, *c++, *c++);
    glPushMatrix();
    glTranslatef(b.pos[0], b.pos[1], b.pos[2]);
    sphere.draw();
    glPopMatrix();
  }

  lines.draw();
  
  WorldScreenProjector proj(v->camera());

  glDisable(GL_LIGHTING);
  int index = 0;
  for (Ball& b : balls) {
    qglviewer::Vec screen = proj.project(b.pos);
    //qglviewer::Vec screen = v->camera()->projectedCoordinatesOf(b.pos);
    // v->drawText(screen.x, screen.y, QString::number(index++));
  }
  glEnable(GL_LIGHTING);
}

void Balls::animate(float endTime) {
  while (!events.empty()) {
    Event& e = events.front();
    if (e.ball.time > endTime)
      break;
    balls[e.index] = e.ball;
    events.pop_front();
  }

  for (Ball& b : balls) {
    b.pos += b.vel * (endTime - b.time);
    b.time = endTime;
  }

  lines.update(balls);
}

void Balls::initBall(int index, float time, float px, float py, float vx, float vy) {
  if (time == 0.0) {
    balls[index].pos[0] = px;
    balls[index].pos[1] = py;
    balls[index].vel[0] = vx;
    balls[index].vel[1] = vy;
  } else {
    events.emplace_back(index, Ball(qglviewer::Vec(px, py, 0.0), qglviewer::Vec(vx, vy, 0.0), time));
  }
}

void Balls::update() {
  std::sort(events.begin(), events.end(), EventLess());
  lines.update(balls);
}

void Balls::Lines::update(std::vector<Ball> balls) {
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

void Balls::Lines::draw() {
  glDisable(GL_LIGHTING);
  glColor3f(1.0f, 1.0f, 1.0f);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
  glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_SHORT, &indices[0]);
  glDisableClientState(GL_VERTEX_ARRAY);
  glEnable(GL_LIGHTING);
}

void Scene::initCushions (double length, double width) {
  cushions = {
      0.0, 0.0,
      0.0, width,
      length, width,
      length, 0.0,
      0.0, 0.0

  };
}

// TODO: anti-aliasing for lines
void Scene::drawCushions (void) {
  glDisable(GL_LIGHTING);
  glColor3f(1.0f, 1.0f, 1.0f);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(2, GL_DOUBLE, 0, &cushions[0]);
  glDrawArrays(GL_LINE_STRIP, 0, cushions.size ()/2);
  glDisableClientState(GL_VERTEX_ARRAY);
  glEnable(GL_LIGHTING);

}

// -----------------------------------------------
// Scene
// -----------------------------------------------

void Scene::readConfig() {
  std::ifstream infile(configFilename.c_str());
  std::string comma;
  
  float width, length, mass, ballRadius;
  size_t numBalls;

  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  infile >> length >> comma;
  infile >> width >> comma;
  infile >> numBalls >> comma;
  infile >> mass >> comma;
  infile >> ballRadius;

  sceneDim[0] = length;
  sceneDim[1] = width;
  initCushions(length, width);
  balls.init(numBalls, ballRadius);
}

void Scene::readLog() {
  std::ifstream infile(eventLogFilename.c_str());
  std::string comma;

  int index;
  float time, px, py, vx, vy;
  
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  while (infile) {
    infile >> index >> comma;
    infile >> time >> comma;
    infile >> px >> comma;
    infile >> py >> comma;
    infile >> vx >> comma;
    infile >> vy;
    if (infile) {
      balls.initBall(index, time, px, py, vx, vy);
    }
  }
  balls.update();
}

qglviewer::Vec Scene::init(float dt) {
  deltaTime = dt;

  readConfig();
  readLog();
  
  return sceneDim;
}

void Scene::draw(Viewer* v) {
  drawCushions ();
  balls.draw(v);
}

void Scene::animate() {
  currentTime += deltaTime;
  balls.animate(currentTime);
}

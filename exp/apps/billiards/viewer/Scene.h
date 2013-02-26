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
#include <deque>

class Viewer;

//! Object that we are modeling
class Ball {
public:
  qglviewer::Vec pos;
  qglviewer::Vec vel;
  float time;

  Ball(): time(0.0) { }

  template<typename T1, typename T2>
  Ball(T1&& p, T2&& v, float t = 0.0): pos(std::forward<T1>(p)), vel(std::forward<T2>(v)), time(t) { }
};

//! Drawing a sphere
class Sphere {
  std::vector<GLfloat> vertices;
  std::vector<GLfloat> normals;
  std::vector<GLushort> indices;

public:
  void init(float radius, float rings, float sectors);
  void draw();
};

//! Drawing a collection of balls
class Balls {
  //! Prototype to model drawing of all balls
  Sphere sphere;
  std::vector<GLfloat> colors;
  std::vector<Ball> balls;

  class Lines {
    std::vector<GLfloat> vertices;
    std::vector<GLushort> indices;
  public:
    void update(std::vector<Ball> balls);
    void draw();
  };

  struct Event {
    Ball ball;
    int index;

    template<typename T>
    Event(int i, T&& b): ball(std::forward<T>(b)), index(i) { }
  };

  struct EventLess {
    bool operator()(const Event& e1, const Event& e2) {
      return e1.ball.time <= e2.ball.time;
    }
  };

  Lines lines;
  std::deque<Event> events;

public:
  void init(size_t numBalls, float ballRadius);
  void initBall(int index, float time, float px, float py, float vx, float vy);
  //! Call after all init calls have been made
  void update();
  void animate(float endTime);
  void draw(Viewer* v);
};


//! Drawing a scene
class Scene {
  Balls balls;
  qglviewer::Vec sceneDim;
  std::vector<GLdouble> cushions;
  std::string configFilename;
  std::string eventLogFilename;
  float currentTime;
  float deltaTime;

  void initCushions(double length, double width);
  void drawCushions (void);

  void readConfig();
  void readLog();
  
public:
  Scene(const std::string& c, const std::string& e): configFilename(c), eventLogFilename(e), currentTime(0.0) { }

  //! Initializes scene; returns dimensions of scene
  qglviewer::Vec init(float dt = 0.1);
  void animate();
  void draw(Viewer* v);
  float time() { return currentTime; }
};

#endif

#ifndef _COLLIDING_OBJECT_H_
#define _COLLIDING_OBJECT_H_

#include <string>
#include <iostream>

#include <cstdlib>
#include <cstdio>
#include <cmath>

class Event;

class CollidingObject {

public:

  virtual ~CollidingObject (void) {}

  virtual unsigned collCounter (void) const = 0;

  virtual void incrCollCounter (void) = 0;

  // need for objects that don't need to recompute their
  // collisions with other bodies. Usually, it's the moving bodies
  // i.e. balls that need to recompute their collisions with other
  // balls and cushions etc, while cushions and potted balls don't need to
  // do so.
  virtual bool isStationary (void) const = 0;

  virtual unsigned getID (void) const = 0;

  virtual std::string str (void) const = 0 ;

  virtual void simulate (const Event& e) = 0;
};



#endif // _COLLIDING_OBJECT_H_

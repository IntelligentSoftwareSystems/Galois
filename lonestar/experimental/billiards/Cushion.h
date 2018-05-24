#ifndef _CUSHION_H_
#define _CUSHION_H_

#include <string>
#include <iostream>

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "GeomUtils.h"
#include "CollidingObject.h"


class Cushion: public CollidingObject {

public:
  // return velocity is REFLECTION_COEFF * incident velocity
  static const FP REFLECTION_COEFF;

private:
  // A cushion is a straight line between
  // two points 'start' and 'end'
  // where we move from 'start' to 'end' going around
  // the table in counterclockwise direction
  //
  // The cushion is represented by the equation
  // r*end + (1-r)*start, where 0 <= r <= 1;
  unsigned m_id;
  LineSegment m_line;

public:
  Cushion (const unsigned id, const Vec2& start, const Vec2& end) 
    : CollidingObject (), m_id (id), m_line (start, end) {}


  const LineSegment& getLineSegment (void) const {
    return m_line;
  }

  virtual bool isStationary () const { return true; }

  // Collision Counter for Cushion and other stationary objects
  // remains fixed
  virtual unsigned collCounter () const { return 0; }

  virtual unsigned getID () const { return m_id; }

  virtual void incrCollCounter () {}

  virtual void simulate (const Event& e);

  virtual std::string str () const {

    char s [256];
    sprintf (s, "[Cushion-%d, %s]", m_id, m_line.str ().c_str ());
    return s;
  }
};



#endif // _CUSHION_H_

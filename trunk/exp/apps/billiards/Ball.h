/** A Billiard Ball  -*- C++ -*-
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
 * @section Description
 *
 * A Billiard Ball .
 *
 * @author <ahassaan@ices.utexas.edu>
 */



#ifndef _BALL_H_
#define _BALL_H_

#include <iostream>
#include <string>

#include <cassert>

#include "Vec2.h"
#include "FPutils.h"
#include "CollidingObject.h"

class Ball: public CollidingObject {

  unsigned id;

  Vec2 position;
  Vec2 velocity;

  double m_mass;
  double m_radius;
  double timestamp;

  unsigned collisionCounter;


public:
  Ball (
      const unsigned id, 
      const Vec2& position, 
      const Vec2& velocity, 
      double mass, 
      double radius):

    id (id), 
    position (position), 
    velocity (velocity), 
    m_mass (mass), 
    m_radius (radius), 
    timestamp (0.0),
    collisionCounter (0) {
      
      truncateAll ();
    }


private:

  void truncateAll () {
    position = FPutils::truncate (position);
    velocity = FPutils::truncate (velocity);
    m_mass = FPutils::truncate (m_mass);
    m_radius = FPutils::truncate (m_radius);
    timestamp = FPutils::truncate (timestamp);
  }

public:

  virtual bool isStationary () const { return false; }

  virtual unsigned collCounter () const { 
    return collisionCounter;
  }

  virtual unsigned getID () const { return id; }

  virtual void incrCollCounter () {
    ++collisionCounter;
  }

  virtual std::string str () const {
    char s [1024];
    sprintf (s, "[Ball-%d,ts=%10.10f,pos=%s,vel=%s,cc=%d]"
        , id, timestamp, position.str ().c_str (), velocity.str ().c_str (), collisionCounter);

    return s;
  }


  void update (const Vec2& newVel, const double time) {

    assert (time > timestamp && "Time update in the past?");

    if (time < timestamp) {
      std::cerr << "Time update in the past" << std::endl;
      abort ();
    }

    Vec2 newPos = this->pos (time); 


    position = FPutils::truncate (newPos);
    velocity = FPutils::truncate (newVel);

    timestamp = FPutils::truncate (time);
  }

  const Vec2& pos () const { return position; }

  Vec2 pos (const double t) const {

    assert (t >= timestamp);
    return (position + velocity * (t - timestamp)); 
  }


  const Vec2& vel () const { return velocity; }

  double mass () const { return m_mass; }

  double time () const { return timestamp; }

  double radius () const { return m_radius; }

  Vec2 mom (const Vec2& _vel) const { return (mass () * (_vel )); }

  Vec2 mom () const { return mom (this->vel ()); }

  double ke (const Vec2& _vel) const { return (_vel.magSqrd () * mass ())/2.0; }

  double ke () const { return ke (this->vel ()); }
  

};





#endif //  _BALL_H_

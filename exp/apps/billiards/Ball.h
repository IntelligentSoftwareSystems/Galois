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

  unsigned m_id;

  Vec2 m_pos;
  Vec2 m_vel;

  double m_mass;
  double m_radius;
  double m_timestamp;

  unsigned m_collCntr;


public:
  Ball (
      const unsigned id,
      const Vec2& pos,
      const Vec2& vel,
      double mass, 
      double radius,
      double time=0.0):

    m_id (id),
    m_pos (pos),
    m_vel (vel),
    m_mass (mass), 
    m_radius (radius), 
    m_timestamp (time),
    m_collCntr (0) {

      assert (mass > 0.0);
      assert (radius > 0.0);
      assert (time >= 0.0);
      
      truncateAll ();
    }


private:

  void truncateAll () {
    m_pos = FPutils::truncate (m_pos);
    m_vel = FPutils::truncate (m_vel);
    m_mass = FPutils::truncate (m_mass);
    m_radius = FPutils::truncate (m_radius);
    m_timestamp = FPutils::truncate (m_timestamp);
  }

public:

  virtual bool isStationary () const { return false; }

  virtual unsigned collCounter () const { 
    return m_collCntr;
  }

  virtual unsigned getID () const { return m_id; }

  virtual void incrCollCounter () {
    ++m_collCntr;
  }

  virtual std::string str () const {
    char s [1024];
    sprintf (s, "[Ball-%d,ts=%10.10f,pos=%s,vel=%s,cc=%d]"
        , m_id, m_timestamp, m_pos.str ().c_str (), m_vel.str ().c_str (), m_collCntr);

    return s;
  }


  void update (const Vec2& newVel, const double time) {

    assert (time > m_timestamp && "Time update in the past?");

    if (time < m_timestamp) {
      std::cerr << "Time update in the past" << std::endl;
      abort ();
    }

    Vec2 newPos = this->pos (time); 


    m_pos = FPutils::truncate (newPos);
    m_vel = FPutils::truncate (newVel);

    m_timestamp = FPutils::truncate (time);
  }

  const Vec2& pos () const { return m_pos; }

  Vec2 pos (const double t) const {

    assert (t >= m_timestamp);
    return (m_pos + m_vel * (t - m_timestamp)); 
  }


  const Vec2& vel () const { return m_vel; }

  double mass () const { return m_mass; }

  double time () const { return m_timestamp; }

  double radius () const { return m_radius; }

  Vec2 mom (const Vec2& _vel) const { return (mass () * (_vel )); }

  Vec2 mom () const { return mom (this->vel ()); }

  double ke (const Vec2& _vel) const { return (_vel.magSqrd () * mass ())/2.0; }

  double ke () const { return ke (this->vel ()); }
  

};





#endif //  _BALL_H_

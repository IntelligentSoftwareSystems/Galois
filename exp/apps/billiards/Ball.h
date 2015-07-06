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

#include "GeomUtils.h"
#include "FPutils.h"
#include "CollidingObject.h"

#include "Galois/FlatSet.h"


#include <iostream>
#include <string>

#include <cassert>

class Sector;

class Ball: public CollidingObject {

  unsigned m_id;

  Vec2 m_pos;
  Vec2 m_vel;

  FP m_mass;
  FP m_radius;
  FP m_timestamp;

  unsigned m_collCntr;

  Galois::FlatSet<Sector*> sectors;

  using SectorIterator = typename Galois::FlatSet<Sector*>::const_iterator;


public:
  Ball (
      const unsigned id,
      const Vec2& pos,
      const Vec2& vel,
      const FP& mass, 
      const FP& radius,
      const FP& time=0.0):

    m_id (id),
    m_pos (pos),
    m_vel (vel),
    m_mass (mass), 
    m_radius (radius), 
    m_timestamp (time),
    m_collCntr (0) {

      assert (mass > FP (0.0));
      assert (radius > FP (0.0));
      assert (time >= FP (0.0));
      
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
        , m_id, double (m_timestamp), m_pos.str ().c_str (), m_vel.str ().c_str (), m_collCntr);

    return s;
  }

  virtual void simulate (const Event& e);

  void addSector (Sector* s) {
    assert (s != nullptr);
    sectors.insert (s);
    assert (sectors.contains (s));
  }

  void removeSector (Sector* s) {
    assert (sectors.contains (s));
    sectors.erase (s);
    assert (!sectors.contains (s));
  }

  bool hasSector (const Sector* s) const {
    assert (s);
    return sectors.contains (const_cast<Sector*> (s));
  }

  std::pair<SectorIterator, SectorIterator> sectorRange (void) const {
    return std::make_pair (sectors.begin (), sectors.end ());
  }

  void update (const Vec2& newVel, const FP& time) {


    if (time < m_timestamp) {
      if (!FPutils::almostEqual (time, m_timestamp)) {
        assert (time >= m_timestamp && "Time update in the past?");
        std::cerr << "Time update in the past" << std::endl;
        abort ();
      }
    }

    Vec2 newPos = this->pos (time); 


    m_pos = FPutils::truncate (newPos);
    m_vel = FPutils::truncate (newVel);

    m_timestamp = FPutils::truncate (time);
  }

  const Vec2& pos () const { return m_pos; }

  Vec2 pos (const FP& t) const {

    if (t < m_timestamp) {
      if (!FPutils::almostEqual (t, m_timestamp)) {
        assert (t >= m_timestamp);
        std::cerr << "Time in the past" << std::endl;
        abort ();
      }
    }
    return (m_pos + m_vel * (t - m_timestamp)); 
  }


  const Vec2& vel () const { return m_vel; }

  const FP& mass () const { return m_mass; }

  const FP& time () const { return m_timestamp; }

  const FP& radius () const { return m_radius; }

  Vec2 mom (const Vec2& _vel) const { return (mass () * (_vel )); }

  Vec2 mom () const { return mom (this->vel ()); }

  FP ke (const Vec2& _vel) const { return (_vel.magSqrd () * mass ())/FP (2.0); }

  FP ke () const { return ke (this->vel ()); }
  

};





#endif //  _BALL_H_

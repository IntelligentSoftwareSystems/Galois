/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef _BALL_H_
#define _BALL_H_

#include "GeomUtils.h"
#include "FPutils.h"
#include "CollidingObject.h"

#include "galois/FlatSet.h"
#include "galois/gIO.h"

#include <boost/iterator/transform_iterator.hpp>

#include <iostream>
#include <string>

#include <cassert>

class Sector;

class Ball : public CollidingObject {

  unsigned m_id;

  Vec2 m_pos;
  Vec2 m_vel;

  FP m_mass;
  FP m_radius;
  FP m_timestamp;

  unsigned m_collCntr;

protected:
  void checkMonotony(const FP& t) const {
    if (t < m_timestamp) {
      if (!FPutils::almostEqual(t, m_timestamp)) {
        assert(t >= m_timestamp);
        std::cerr << "Time in the past" << std::endl;
        abort();
      }
    }
  }

public:
  Ball(const unsigned id, const Vec2& pos, const Vec2& vel, const FP& mass,
       const FP& radius, const FP& time = 0.0)
      :

        m_id(id), m_pos(pos), m_vel(vel), m_mass(mass), m_radius(radius),
        m_timestamp(time), m_collCntr(0) {

    assert(mass > FP(0.0));
    assert(radius > FP(0.0));
    assert(time >= FP(0.0));
  }

public:
  virtual bool isStationary() const { return false; }

  virtual unsigned collCounter() const { return m_collCntr; }

  virtual unsigned getID() const { return m_id; }

  virtual void incrCollCounter() { ++m_collCntr; }

  virtual std::string str() const {
    char s[1024];
    sprintf(s, "[Ball-%d,ts=%10.10f,pos=%s,vel=%s,cc=%d]", m_id,
            double(m_timestamp), m_pos.str().c_str(), m_vel.str().c_str(),
            m_collCntr);

    return s;
  }

  virtual void simulate(const Event& e);

  void update(const Vec2& newVel, const FP& time) {

    checkMonotony(time);
    Vec2 newPos = this->pos(time);

    m_pos       = newPos;
    m_vel       = newVel;
    m_timestamp = time;
  }

  const Vec2& pos() const { return m_pos; }

  Vec2 pos(const FP& t) const {

    checkMonotony(t);
    return (m_pos + m_vel * t - m_vel * m_timestamp);
  }

  const Vec2& vel() const { return m_vel; }

  const FP& mass() const { return m_mass; }

  const FP& time() const { return m_timestamp; }

  const FP& radius() const { return m_radius; }

  Vec2 mom(const Vec2& _vel) const { return (mass() * (_vel)); }

  Vec2 mom() const { return mom(this->vel()); }

  FP ke(const Vec2& _vel) const { return (_vel.magSqrd() * mass()) / FP(2.0); }

  FP ke() const { return ke(this->vel()); }

  // TODO: move to runtime

  const Ball* readWeak(const Event& e) const { return this; }

  const Ball* getWrapper(void) const { return this; }

  // TODO: fix this anomaly
  const FP& ghostTime(void) const { return time(); }

  const Vec2& ghostPos(void) const { return pos(); }
};

class BallSectored : public Ball {

  using SectorSet      = galois::FlatSet<Sector*>;
  using SectorIterator = typename SectorSet::const_iterator;

  Vec2 m_ghost_pos;
  FP m_ghost_ts;

  SectorSet sectors;

public:
  BallSectored(const unsigned id, const Vec2& pos, const Vec2& vel,
               const FP& mass, const FP& radius, const FP& time = 0.0)
      :

        Ball(id, pos, vel, mass, radius, time), m_ghost_pos(pos),
        m_ghost_ts(time)

  {}

  void addSector(Sector* s) {
    assert(s != nullptr);
    sectors.insert(s);
    assert(sectors.contains(s));
  }

  void removeSector(Sector* s) {
    assert(sectors.contains(s));
    sectors.erase(s);
    assert(!sectors.contains(s));
  }

  void removeAllSectors(void) { sectors.clear(); }

  bool hasSector(const Sector* s) const {
    assert(s);
    return sectors.contains(const_cast<Sector*>(s));
  }

  std::pair<SectorIterator, SectorIterator> sectorRange(void) const {
    return std::make_pair(sectors.begin(), sectors.end());
  }

  void updateGhostPos(const FP& time) {
    checkMonotony(time);
    m_ghost_pos = this->pos(time);
    m_ghost_ts  = time;
  }

  void update(const Vec2& newVel, const FP& time) {

    Ball::update(newVel, time);

    m_ghost_pos = Ball::pos();
    m_ghost_ts  = Ball::time();
  }

  const Vec2& ghostPos(void) const { return m_ghost_pos; }

  const FP& ghostTime(void) const { return m_ghost_ts; }
};

template <typename B, typename E = Event>
class BallOptimWrapper {

  using BallAlloc = galois::FixedSizeAllocator<B>;
  using CheckP    = std::pair<E, B*>;
  using StateLog  = galois::gstl::List<CheckP>;

  using dbg = galois::debug<0>;

  B* m_ball;
  BallAlloc m_alloc;
  StateLog m_hist;

public:
  BallOptimWrapper(B* b) : m_ball(b) {}

  bool hasEmptyHistory(void) const { return m_hist.empty(); }

  B* checkpoint(const E& e) {

    dbg::print("checkpoint called on ", m_ball->str());

    B* bcpy = m_alloc.allocate(1);
    m_alloc.construct(bcpy, static_cast<B&>(*m_ball));

    m_hist.push_back(CheckP(e, bcpy));

    return bcpy;
  }

  void restore(B* bcpy) {
    dbg::print("restore called on ", m_ball->str(), ", restoring from ",
               bcpy->str());

    assert(bcpy);
    assert(m_ball->getID() == bcpy->getID());

    *m_ball = *bcpy;

    assert(!m_hist.empty());
    CheckP& tail = m_hist.back();
    m_hist.pop_back();

    assert(tail.second == bcpy);

    m_alloc.destroy(bcpy);
    m_alloc.deallocate(bcpy, 1);
  }

  void reclaim(const E& e, B* b) {
    assert(b);

    dbg::print("reclaim called on ", m_ball->str(),
               ", removing copy: ", b->str());

    assert(!m_hist.empty());
    CheckP& head = m_hist.front();

    assert(head.first == e);
    assert(m_ball->getID() == b->getID());
    assert(head.second == b);

    if (head.second != b || m_ball->getID() != b->getID()) {
      std::abort();
    }

    m_hist.pop_front();

    m_alloc.destroy(b);
    m_alloc.deallocate(b, 1);
  }

  struct BallReader : std::unary_function<const CheckP, const B*> {

    const B* operator()(const CheckP& p) const { return p.second; }
  };

  using BallHistIter =
      boost::transform_iterator<BallReader, typename StateLog::const_iterator>;

  BallHistIter ballHistBeg(void) const {
    return boost::make_transform_iterator(m_hist.begin(), BallReader());
  }

  BallHistIter ballHistEnd(void) const {
    return boost::make_transform_iterator(m_hist.end(), BallReader());
  }

  const B* readWeak(const E& e) const {

    typename E::Comparator cmp;

    for (const CheckP& p : m_hist) {
      if (cmp(e, p.first)) { // first entry larger than e
        return p.second;
      }
    }

    return m_ball;
  }
};

template <typename B = Ball>
class BallOptim : public B {

public:
  using Wrapper = BallOptimWrapper<BallOptim>;

private:
  using WrapperPtr = std::shared_ptr<Wrapper>;

  WrapperPtr wrap;

public:
  BallOptim(const unsigned id, const Vec2& pos, const Vec2& vel, const FP& mass,
            const FP& radius, const FP& time = 0.0)
      :

        B(id, pos, vel, mass, radius, time), wrap(new Wrapper(this)) {}

  Wrapper* getWrapper(void) { return wrap.get(); }

  const Wrapper* getWrapper(void) const { return wrap.get(); }
};

#endif //  _BALL_H_

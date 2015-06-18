/** Event  -*- C++ -*-
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
 * Event.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef _EVENT_H_
#define _EVENT_H_

#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>

#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "GeomUtils.h"
#include "FPutils.h"

#include "CollidingObject.h"
#include "Cushion.h"
#include "Ball.h"
#include "Sector.h"

// We use EventKind to identify
// different types of Event. A common alternative
// is to use virtual functions and hierarchy, but that
// requires Event's being allocated on the heap.


class Event {
  friend class Table;

public:
  enum EventKind {
    BALL_COLLISION = 0,
    CUSHION_COLLISION = 1,
    SECTOR_ENTRY = 2,
    SECTOR_LEAVE = 3
  };

private:
  EventKind kind;

  Ball* ball;
  CollidingObject* otherObj;

  double time;

  // collision counter is used to maintain
  // the versions of state of the object. If
  // at the time of processing the event, the
  // collision counters do not match, then the
  // Event (corresponding collision) is based on
  // outdated information and must not be simulated
  unsigned collCounterA;
  unsigned collCounterB;

  Event (
      const EventKind& kind, 
      Ball* ball, 
      CollidingObject* otherObj, 
      const double time) 
    : 
    kind (kind),
    ball (ball),
    otherObj (otherObj),
    time (time) {

      assert (time >= 0.0);

      collCounterA = this->ball->collCounter ();
      collCounterB = this->otherObj->collCounter ();

      switch (kind) {
        case BALL_COLLISION:
          assert (dynamic_cast<Ball*> (otherObj) != nullptr);
          break;

        case CUSHION_COLLISION:
          assert (dynamic_cast<Cushion*> (otherObj) != nullptr);
          break;

        case SECTOR_ENTRY:
        case SECTOR_LEAVE:
          assert (dynamic_cast<Sector*> (otherObj) != nullptr);
          break;

        default:
          GALOIS_DIE ("shouldn't reach here");
      }

    }

public:

  template <typename T>
  static Event makeEvent (const EventKind& kind, const Ball* ball, const T* collObj, const double time) {

    assert (&ball != nullptr);
    assert (&collObj != nullptr);

    return Event (kind, const_cast<Ball*> (ball), const_cast<T*> (collObj), time);
  }

  // static Event makeCushionCollision (const Ball* ball, const Cushion* c, const double time) {
// 
    // assert (&ball != nullptr);
    // assert (&c != nullptr);
// 
    // return Event (CUSHION_COLLISION, const_cast<Ball*> (ball), const_cast<Cushion*> (c), time);
// 
  // }
// 
  // static Event makeSectorEntry (const Ball* ball, const Sector* sector, const double time) {
// 
    // assert (&ball != nullptr);
    // assert (&sector != nullptr);
// 
    // return Event (SECTOR_ENTRY, const_cast<Ball*> (ball), const_cast<Sector*> (sector), time);
  // }
// 
  // static Event makeSectorLeave (const Ball* ball, const Sector* sector, const double time) {
// 
    // assert (&ball != nullptr);
    // assert (&sector != nullptr);
// 
    // return Event (SECTOR_LEAVE, const_cast<Ball*> (ball), const_cast<Sector*> (sector), time);
    // 
  // }

  void simulate () {
    assert (notStale ());

    if (notStale ()) {
      // update collision counters, such that event remains valid
      ball->incrCollCounter ();
      otherObj->incrCollCounter ();

      this->collCounterA = ball->collCounter ();
      this->collCounterB = otherObj->collCounter ();

      otherObj->simulate (*this);
    }
  }

  double getTime () const { return time; }

  Ball* getBall () const { 
    assert (ball != nullptr);
    return ball; 
  }

  bool notStale () const { 
    return (ball->collCounter () == this->collCounterA && 
        otherObj->collCounter () == this->collCounterB);
  }

  bool firstBallChanged () const { 
    return (ball->collCounter () != this->collCounterA);
  }

  bool otherBallChanged () const {
    if (kind == BALL_COLLISION) {
      return (otherObj->collCounter () != this->collCounterB);

    } else { 
      return false;
    }
  }

  void updateFirstBall (const Ball& b) {
    *(this->ball) = b;
    this->collCounterA = b.collCounter ();
  }

  void updateOtherBall (const Ball& b) {
    assert (kind == BALL_COLLISION);
    Ball* ob = downCast<Ball> (otherObj);
    *ob = b;
    this->collCounterB = b.collCounter ();
  }

  Ball* getOtherBall () const { 
    assert (kind == BALL_COLLISION);

    return downCast<Ball> (otherObj);
  }

  Cushion* getCushion () const {
    assert (kind == CUSHION_COLLISION);

    return downCast<Cushion> (otherObj);
  }

  Sector* getSector () const {
    assert (kind == SECTOR_ENTRY || kind == SECTOR_LEAVE);
    
    return downCast<Sector> (otherObj);
  }

  EventKind getKind () const { return kind; }


  std::string str () const {
    const char* kindName = nullptr;

    std::string objBstr;
    switch (kind) {
      case BALL_COLLISION:
        kindName = "BALL_COLLISION";
        objBstr = getOtherBall ()->str ();
        break;

      case CUSHION_COLLISION:
        kindName = "CUSHION_COLLISION";
        objBstr = getCushion ()->str ();
        break;

      default:
        abort ();

    }

    std::ostringstream s;

    s.precision (10);
    s << "[time=" << std::fixed <<  time << ", kind=" << kindName;

    s << std::setw (20) << std::setfill (' ') << "ball=" << ball->str ();

    s << std::setw (20) << std::setfill (' ') << "otherObj=" << objBstr;

    s << "]" << std::endl;

    return s.str ();

  }

  friend std::ostream& operator << (std::ostream& out, const Event& e) {
    out << e.str ();
    return out;
  }

private:

  template <typename T>
  static T* downCast (CollidingObject* obj) {
    T* ptr = dynamic_cast<T*> (obj);
    assert (ptr != nullptr);
    return ptr;
  }

public:
  struct Comparator {

    // TODO: fix for almostEqual events
    static int compare (const Event& e1, const Event& e2) {

      int cmp = 0;

      if (FPutils::almostEqual (e1.time, e2.time)) {
        cmp = e1.ball->getID () - e2.ball->getID ();

        if (cmp == 0) {
          unsigned othid1 = (e1.otherObj == nullptr) ? 0 : e1.otherObj->getID ();
          unsigned othid2 = (e2.otherObj == nullptr) ? 0 : e2.otherObj->getID ();

          cmp = othid1 - othid2;

          if (cmp == 0) {
            cmp = int (e1.getKind ()) - int (e2.getKind ());
          }
        }

      } else if (e1.time < e2.time) {
        cmp = -1;

      } else if (e1.time > e2.time) {
        cmp = 1;

      } else {
        abort ();
      }

      return cmp;
    }

    bool operator () (const Event& e1, const Event& e2) const {
      return (compare (e1, e2) < 0);
    }

  };

  // We want to sort events in increasing order of time,
  // but since stl heap is a max heap, we use the opposite sense
  // of comparison
  struct ReverseComparator: Comparator {

    bool operator () (const Event& e1, const Event& e2) const {
      return (compare (e1, e2) > 0);
    }
  };

  bool operator < (const Event& that) const {
    return Comparator::compare (*this, that) < 0;
  }

  bool operator > (const Event& that) const {
    return Comparator::compare (*this, that) > 0;
  }

  bool operator == (const Event& that) const { 
    return Comparator::compare (*this, that) == 0;
  }

  bool operator != (const Event& that) const {
    return Comparator::compare (*this, that) != 0;
  }

  bool operator >= (const Event& that) const { 
    return Comparator::compare (*this, that) >= 0; 
  }

  bool operator <= (const Event& that) const {
    return Comparator::compare (*this, that) <= 0;
  }

};

#endif //  _EVENT_H_

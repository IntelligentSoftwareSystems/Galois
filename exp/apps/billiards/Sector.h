/** A spatial partition  -*- C++ -*-
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
 * A spatial partition .
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef SECTOR_H_
#define SECTOR_H_

#include "Galois/FlatSet.h"
#include "Galois/optional.h"

#include "GeomUtils.h"
#include "CollidingObject.h"
#include "Ball.h"
#include "Cushion.h"

class Sector: public CollidingObject {

  unsigned id;

  // sides are created in clockwise direction,
  // starting from the bottom left
  std::vector<LineSegment> sides;

  // a neighbor associated with each side
  std::vector<Sector*> neighbors;

  BoundingBox boundbox;

  Galois::FlatSet<Cushion*> cushions;
  Galois::FlatSet<Ball*> balls;

  void init (const Vec2& bottomLeft, const FP& sectorSize) {

    Vec2 delX (sectorSize, 0.0);
    Vec2 delY (0.0, sectorSize);

    Vec2 topLeft = bottomLeft + delY;

    Vec2 topRight = topLeft + delX;

    Vec2 bottomRight = bottomLeft + delX;

    sides.clear ();
    assert (sides.empty ());
    sides.emplace_back (bottomLeft, topLeft);
    sides.emplace_back (topLeft, topRight);
    sides.emplace_back (topRight, bottomRight);
    sides.emplace_back (bottomRight, bottomLeft);


    neighbors.resize (sides.size (), nullptr);

    boundbox.update (bottomLeft);
    boundbox.update (topLeft);
    boundbox.update (topRight);
    boundbox.update (bottomRight);
  }

public:

  Sector (unsigned id, const Vec2& bottomLeft, const FP& sectorSize)
    : CollidingObject (), id (id) 
  {
    init (bottomLeft, sectorSize);
  }

  virtual ~Sector (void) {
  }

  virtual unsigned collCounter (void) const { 
    return 0;
  }

  virtual void incrCollCounter (void) {
  }

  virtual bool isStationary (void) const {
    return true;
  }

  virtual unsigned getID (void) const { 
    return id;
  }

  virtual std::string str (void) const {
    char s[256];

    std::sprintf (s, "Sector-%d: %s", id, boundbox.str ().c_str ());

    return s;
  }


  void addNeighbor (const RectSide& sd, Sector* sec) {
    assert (sec != nullptr);
    assert (int (sd) < int (sides.size ()));

    assert (neighbors.at (int (sd)) == nullptr);
    neighbors[int (sd)] = sec;
  }

  void addCushion (Cushion* c) {
    assert (c != nullptr);

    cushions.insert (c);
  }

  void addBall (Ball* b) {
    assert (b != nullptr);

    balls.insert (b);
    assert (balls.contains (b));
  }

  void removeBall (Ball* b) {
    assert (balls.contains (b));
    balls.erase (b);

    assert (!balls.contains (b));
  }

  bool hasBall (const Ball* b) const { 
    assert (b);
    return balls.contains (const_cast<Ball*> (b));
  }


  bool intersects (const Ball* ball) const {

    const Vec2& p = ball->pos ();

    if (boundbox.isInside (p)) {
      return true;

    }

    // measure distance from ball to all sides of the sector
    // if distance < radius, return true
    for (const LineSegment& l: sides) {
      if (l.distanceFrom (p) < ball->radius ()) {
        return true;
      }
    }


    return false;
  }

  virtual void simulate (const Event& e);

  Galois::optional<Event> computeEarliestEvent (const Ball* ball, const FP& endtime) const;


  Galois::optional<Event> earliestSectorEntry (const Ball* ball, const FP& endtime) const;


  Galois::optional<Event> earliestSectorLeave (const Ball* b, const FP& endtime) const;


};




#endif // SECTOR_H_

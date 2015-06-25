/** Billiards Table  -*- C++ -*-
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
 * Billiards Table.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef _TABLE_SECTORED_H_
#define _TABLE_SECTORED_H_

#include "Table.h"

#include "FPutils.h"
#include "GeomUtils.h"
#include "Cushion.h"
#include "Ball.h"
#include "Collision.h"
#include "Event.h"

#include "Galois/optional.h"

#include "Galois/Runtime/ll/gio.h"

#include <vector>

#include <cstdlib>
#include <cstdio>
#include <ctime>

class TableSectored: public Table {

private:
  
  unsigned sectorSize;
  unsigned xSectors;
  unsigned ySectors;

  std::vector<std::vector<Sector*> > sectors;

  Table& operator = (const Table& that) { abort (); return *this; }

public:

  TableSectored (unsigned numBalls, unsigned sectorSize, unsigned xSectors, unsigned ySectors) 
    :
      Table (numBalls, (sectorSize * xSectors), (sectorSize * ySectors)),
      sectorSize (sectorSize),
      xSectors (xSectors),
      ySectors (ySectors)
  {
    init ();
  }

  template <typename I>
  TableSectored (const I ballsBeg, const I ballsEnd, unsigned sectorSize, unsigned xSectors, unsigned ySectors) 
    :
      Table (ballsBeg, ballsEnd, (sectorSize * xSectors), (sectorSize * ySectors)),
      sectorSize (sectorSize),
      xSectors (xSectors),
      ySectors (ySectors)
  {
    init ();

  }


  TableSectored (const TableSectored& that) 
    : Table (that), sectorSize (that.sectorSize), xSectors (that.xSectors), ySectors (that.ySectors) {

      sectors.resize (that.sectors.size ());

      assert (that.sectors.size () == that.xSectors);
      for (size_t i = 0; i < that.sectors.size (); ++i) {
        Table::copyVecPtr (that.sectors[i], this->sectors[i]);
      }

  }

  ~TableSectored () {
    assert (sectors.size () == xSectors);
    for (size_t i = 0; i < sectors.size (); ++i) {
      Table::freeVecPtr (sectors [i]);
    }
  }


  void genInitialEvents (std::vector<Event>& initEvents, const double endtime) {

    initEvents.clear ();

    for (std::vector<Ball*>::iterator i = balls.begin (), ei = balls.end ();
        i != ei; ++i) {

      addEventsForOneBall (initEvents, *i, endtime);

    }
  }


  //! @param addList
  //! @param b1, ball involved in a collision just simualtedrevious collision 
  //! @param b2, other object involved in a collision just simualted
  //!
  //! The idea is to avoid generating a collision event with previous object. Because
  //! the collision physics is going to generate a new collision with the same object the
  //! ball just collided with, but we have already simulated a collision and ball should be moving
  //! in another direction and pick a different collision

  template <typename C>
  void addNextEvents (const Event& e, C& addList, const double endtime) const {

    switch (e.getKind ()) {
      case Event::BALL_COLLISION:
        addEventsForTwoBalls (addList, e.getBall (), e.getOtherBall (), endtime, &e);
        break;

      case Event::CUSHION_COLLISION:
        addEventsForOneBall (addList, e.getBall (), endtime, &e);
        break;

      case Event::SECTOR_ENTRY:
        addEventsForOneBall (addList, e.getBall (), endtime, &e);
        break;

      case Event::SECTOR_LEAVE:
        addEventsForOneBall (addList, e.getBall (), endtime, &e);
        break;

      default:
        GALOIS_DIE ("unkown event kind");
    }
  }

private:

  void init (void) {
    createSectors ();
    addCushionsToSectors ();
    addBallsToSectors ();
  }

  void createSectors (void) {
    unsigned numSectors = xSectors * ySectors;
    assert (numSectors != 0);

    sectors.clear ();
    sectors.resize (xSectors, std::vector<Sector*> (ySectors, nullptr));

    unsigned idCntr = 0;

    for (size_t i = 0; i < xSectors; ++i) {
      for (size_t j = 0; j < ySectors; ++j) {

        // bottom left corner of sector
        Vec2 bl (double (i*sectorSize), double (j*sectorSize));

        sectors [i][j] = new Sector (idCntr, bl, double (sectorSize)); 
        idCntr++;
      }
    }


    // add adjacency information

    for (int i = 0; i < xSectors; ++i) {
      for (int j = 0; j < ySectors; ++j) {

        if ((i - 1) >= 0) {
          sectors [i][j]->addNeighbor (RectSide::LEFT, sectors [i-1][j]);
        }

        if ((j - 1) >= 0) {
          sectors [i][j]->addNeighbor (RectSide::BOTTOM, sectors [i][j-1]);
        }

        if ((i + 1) < xSectors) {
          sectors [i][j]->addNeighbor (RectSide::RIGHT, sectors [i+1][j]);
        }

        if ((j + 1) < ySectors) {
          sectors [i][j]->addNeighbor (RectSide::TOP, sectors [i][j+1]);
        }

      }
    }
  }

  void addCushionsToSectors () {

    // add cusions to sectors
    for (size_t i = 0; i < xSectors; ++i) {
      for (size_t j = 0; j < ySectors; ++j) {

        if (i == 0) {
          sectors[i][j]->addCushion (Table::getCushion (RectSide::LEFT));
        }

        if (i == (xSectors - 1)) {
          assert (xSectors > 0);

          sectors[i][j]->addCushion (Table::getCushion (RectSide::RIGHT));
        }

        if (j == 0) {
          sectors[i][j]->addCushion (Table::getCushion (RectSide::BOTTOM));
        }

        if (j == (ySectors - 1)) {
          assert (ySectors > 0);
          sectors[i][j]->addCushion (Table::getCushion (RectSide::TOP));
        }
      }
    }


  }


  void addBallsToSectors () {

    // add balls to sectors
    for (Ball* b: balls) {

      for (size_t i = 0; i < xSectors; ++i) {
        for (size_t j = 0; j < ySectors; ++j) {

          if (sectors[i][j]->intersects (b)) {
            sectors[i][j]->addBall (b);
            b->addSector (sectors[i][j]);
          }
        }
      }
    } // end for balls
  }


  Galois::optional<Event> computeEarliestEvent (Ball* ball, const double endtime) const {

    auto range = ball->sectorRange ();

    Galois::optional<Event> minEvent;
    Event::Comparator cmp;

    for (auto i = range.first; i != range.second; ++i) {
      Galois::optional<Event> e = (*i)->computeEarliestEvent (ball, endtime);

      if (e) {
        if (!minEvent || cmp (*e, *minEvent)) {
          minEvent = e;
        }
      }
    }

    return minEvent;
  }

  template <typename C>
  void addEventsForTwoBalls (C& addList, Ball* b1, Ball* b2, const double endtime, const Event* prevEvent) const {

    assert (prevEvent != nullptr);
    assert (prevEvent->getKind () == Event::BALL_COLLISION);

    if (!prevEvent->firstBallChanged ()) {

      Galois::optional<Event> minE1 = computeEarliestEvent (b1, endtime);
      if (minE1) {
        addList.push_back (*minE1);

        if (prevEvent->notStale ()) {
          assert (*prevEvent != *minE1);
        }
      }
    }

    if (!prevEvent->otherBallChanged ()) {

      Galois::optional<Event> minE2 = computeEarliestEvent (b2, endtime);
      if (minE2) {
        addList.push_back (*minE2);

        if (prevEvent->notStale ()) {
          assert (*prevEvent != *minE2);
        }
      }
    }

  }

  template <typename C>
  void addEventsForOneBall (C& addList, Ball* b, const double endtime, const Event* prevEvent=nullptr) const {

    if (prevEvent && prevEvent->firstBallChanged ()) {
      assert (prevEvent->getKind () != Event::BALL_COLLISION);
      return; // prevEvent involved a single ball and was invalid,
      // which means ball underwent some other valid event, which should have scheduled a correct new event
      // after updating the ball. 
    }

    Galois::optional<Event> minE = computeEarliestEvent (b, endtime);
    if (minE) {
      addList.push_back (*minE);

      if (prevEvent && prevEvent->notStale ()) {
        assert (*prevEvent != *minE);
      }
    }
  }

};


#endif //  _TABLE_SECTORED_H_
